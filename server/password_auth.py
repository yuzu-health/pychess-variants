"""Username/password accounts gated by a shared registration secret.

Password hashes live in their own ``user_password`` collection (keyed by username)
so they never travel with user documents (profile caches, personal-data exports,
admin views).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import re
import secrets
from datetime import UTC, datetime
from typing import Any

import aiohttp_jinja2
import aiohttp_session
from aiohttp import web
from client_ip import get_client_ip
from const import reserved
from login import USERNAME_LOWER_FIELD, normalized_username, username_exists
from pychess_global_app_state_utils import get_app_state
from pymongo.errors import DuplicateKeyError
from request_protection import SlidingWindowLimiter
from security_evasion import (
    collect_client_signals,
    is_signup_blocked_by_signals,
    remember_user_signals,
)
from session_security import auth_version_from_user_document, authenticate_session
from settings import REGISTRATION_SECRET
from typedefs import REQUEST_NEW_SESSION_KEY
from typing_defs import UserDocument
from user_stats import DEFAULT_USER_COUNT
from views import get_user_context

log = logging.getLogger(__name__)

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,20}$")
PASSWORD_MIN_LENGTH = 8
# Bound the work an attacker can force through the KDF.
PASSWORD_MAX_LENGTH = 256

# scrypt parameters: N=2**15, r=8 -> 32 MiB per hash, ~50-100 ms on a modern core.
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_MAXMEM = 64 * 1024 * 1024
SCRYPT_DKLEN = 32

LOGIN_IP_LIMIT = (20, 300.0)  # attempts per IP per 5 minutes
LOGIN_USER_LIMIT = (10, 300.0)  # attempts per username per 5 minutes
SIGNUP_IP_LIMIT = (10, 600.0)  # attempts per IP per 10 minutes

_limiter = SlidingWindowLimiter()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        maxmem=SCRYPT_MAXMEM,
        dklen=SCRYPT_DKLEN,
    )


def hash_password_sync(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password_sync(password: str, encoded: str) -> bool:
    try:
        algo, n, r, p, salt_b64, digest_b64 = encoded.split("$")
        if algo != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = _scrypt(password, salt, int(n), int(r), int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


# Verifying against a dummy hash for unknown usernames keeps response time uniform,
# so login timing does not reveal which usernames exist.
_DUMMY_HASH = hash_password_sync(secrets.token_urlsafe(16))


async def hash_password(password: str) -> str:
    return await asyncio.to_thread(hash_password_sync, password)


async def verify_password(password: str, encoded: str) -> bool:
    return await asyncio.to_thread(verify_password_sync, password, encoded)


def registration_secret_matches(supplied: str) -> bool:
    return hmac.compare_digest(supplied.encode("utf-8"), REGISTRATION_SECRET.encode("utf-8"))


def username_error(username: str) -> str | None:
    if not USERNAME_RE.match(username):
        return "Username must be 3-20 characters: letters, numbers, _ and -"
    if reserved(username):
        return "Username is reserved"
    return None


def password_error(password: str, password2: str) -> str | None:
    if len(password) < PASSWORD_MIN_LENGTH:
        return f"Password must be at least {PASSWORD_MIN_LENGTH} characters"
    if len(password) > PASSWORD_MAX_LENGTH:
        return f"Password must be at most {PASSWORD_MAX_LENGTH} characters"
    if password != password2:
        return "Passwords do not match"
    return None


def _allowed(bucket: str, key: str, limit: tuple[int, float]) -> bool:
    return _limiter.allow(f"{bucket}:{key}", limit[0], limit[1])


def _reject_cross_site(request: web.Request) -> None:
    # The CSRF middleware skips cookieless requests; still refuse cross-site form
    # posts so another site cannot sign a visitor into an attacker-chosen account.
    if request.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
        raise web.HTTPForbidden(text="Cross-site request refused")


def _form_str(form: Any, name: str) -> str:
    value = form.get(name, "")
    return value if isinstance(value, str) else ""


async def _render(
    request: web.Request,
    template: str,
    *,
    error: str = "",
    username: str = "",
    status: int = 200,
) -> web.Response:
    user, context = await get_user_context(request)
    if not user.anon:
        raise web.HTTPFound("/")
    context["view_css"] = "faq.css"
    context["auth_error"] = error
    context["auth_username"] = username
    return await aiohttp_jinja2.render_template_async(template, request, context, status=status)


async def login_page(request: web.Request) -> web.StreamResponse:
    return await _render(request, "password_login.html")


async def signup_page(request: web.Request) -> web.StreamResponse:
    return await _render(request, "password_signup.html")


async def _find_user(app_state: Any, username: str) -> UserDocument | None:
    return await app_state.db.user.find_one(
        {
            "$or": [
                {"_id": username},
                {USERNAME_LOWER_FIELD: {"$eq": normalized_username(username), "$type": "string"}},
            ]
        }
    )


async def login_post(request: web.Request) -> web.StreamResponse:
    _reject_cross_site(request)
    form = await request.post()
    username = _form_str(form, "username").strip()
    password = _form_str(form, "password")

    ip = get_client_ip(request) or "unknown"
    if not _allowed("pw_login_ip", ip, LOGIN_IP_LIMIT) or not _allowed(
        "pw_login_user", normalized_username(username), LOGIN_USER_LIMIT
    ):
        return await _render(
            request,
            "password_login.html",
            error="Too many login attempts. Try again in a few minutes.",
            username=username,
            status=429,
        )

    invalid = "Invalid username or password"
    if not username or not password or len(password) > PASSWORD_MAX_LENGTH:
        return await _render(
            request, "password_login.html", error=invalid, username=username, status=400
        )

    app_state = get_app_state(request.app)
    user_doc = await _find_user(app_state, username)
    cred = (
        await app_state.db.user_password.find_one({"_id": user_doc["_id"]})
        if user_doc is not None
        else None
    )
    encoded = cred.get("hash") if cred else None
    ok = await verify_password(password, encoded if isinstance(encoded, str) else _DUMMY_HASH)
    if not ok or user_doc is None or cred is None:
        log.info("Failed password login for %r", username)
        return await _render(
            request, "password_login.html", error=invalid, username=username, status=400
        )

    if not user_doc.get("enabled", True):
        return await _render(
            request,
            "password_login.html",
            error="This account is closed. Contact a site admin.",
            username=username,
            status=403,
        )

    session = await aiohttp_session.get_session(request)
    authenticate_session(session, user_doc["_id"], auth_version_from_user_document(user_doc))
    request[REQUEST_NEW_SESSION_KEY] = True
    await remember_user_signals(app_state.db, user_doc["_id"], collect_client_signals(request))
    log.info("+++ password login: %s", user_doc["_id"])
    return web.HTTPFound("/")


async def signup_post(request: web.Request) -> web.StreamResponse:
    _reject_cross_site(request)
    form = await request.post()
    username = _form_str(form, "username").strip()
    password = _form_str(form, "password")
    password2 = _form_str(form, "password2")
    secret = _form_str(form, "secret")

    def fail(error: str, status: int = 400) -> Any:
        return _render(
            request, "password_signup.html", error=error, username=username, status=status
        )

    ip = get_client_ip(request) or "unknown"
    if not _allowed("pw_signup_ip", ip, SIGNUP_IP_LIMIT):
        return await fail("Too many signup attempts. Try again in a few minutes.", 429)

    if not registration_secret_matches(secret):
        log.warning("Signup with wrong registration secret from %s (username %r)", ip, username)
        return await fail("Wrong challenge secret", 403)

    error = username_error(username) or password_error(password, password2)
    if error:
        return await fail(error)

    app_state = get_app_state(request.app)
    if await username_exists(app_state, username):
        return await fail("Username is already taken")

    signals = collect_client_signals(request)
    blocked, match_reason = await is_signup_blocked_by_signals(app_state.db, signals)
    if blocked:
        log.warning("Blocked signup for %s (ban-evasion %s match)", username, match_reason)
        return await fail("Signup blocked. Contact a site admin.", 403)

    password_hash = await hash_password(password)
    now = datetime.now(UTC)
    try:
        await app_state.db.user.insert_one(
            {
                "_id": username,
                USERNAME_LOWER_FIELD: normalized_username(username),
                "title": "",
                "createdAt": now,
                "perfs": {},
                "pperfs": {},
                "count": dict(DEFAULT_USER_COUNT),
                "enabled": True,
                "shadowban": False,
            }
        )
    except DuplicateKeyError:
        return await fail("Username is already taken")

    await app_state.db.user_password.replace_one(
        {"_id": username},
        {"_id": username, "hash": password_hash, "updatedAt": now},
        upsert=True,
    )

    session = await aiohttp_session.get_session(request)
    authenticate_session(session, username, 0)
    request[REQUEST_NEW_SESSION_KEY] = True
    await remember_user_signals(app_state.db, username, signals)
    log.info("Created new user %s via password signup", username)
    return web.HTTPFound("/")
