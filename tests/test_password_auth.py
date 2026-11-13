import re
import unittest
from unittest.mock import patch

import test_logger
from aiohttp.test_utils import AioHTTPTestCase
from mongomock_motor import AsyncMongoMockClient
from password_auth import (
    hash_password_sync,
    registration_secret_matches,
    verify_password_sync,
)
from pychess_global_app_state_utils import get_app_state
from request_protection import SlidingWindowLimiter

from server import make_app

test_logger.init_test_logger()

PASSWORD = "correct-horse-battery"


class PasswordHashTestCase(unittest.TestCase):
    def test_hash_round_trip(self):
        encoded = hash_password_sync(PASSWORD)
        self.assertTrue(encoded.startswith("scrypt$"))
        self.assertTrue(verify_password_sync(PASSWORD, encoded))
        self.assertFalse(verify_password_sync(PASSWORD + "x", encoded))

    def test_hashes_are_salted(self):
        self.assertNotEqual(hash_password_sync(PASSWORD), hash_password_sync(PASSWORD))

    def test_malformed_hash_is_rejected(self):
        self.assertFalse(verify_password_sync(PASSWORD, "not-a-hash"))
        self.assertFalse(verify_password_sync(PASSWORD, "md5$1$2$3$xx$yy"))

    def test_registration_secret(self):
        with patch("password_auth.REGISTRATION_SECRET", "letmein"):
            self.assertTrue(registration_secret_matches("letmein"))
            self.assertFalse(registration_secret_matches("letmein "))
            self.assertFalse(registration_secret_matches(""))


class PasswordAuthRouteTestCase(AioHTTPTestCase):
    async def get_application(self):
        return make_app(db_client=AsyncMongoMockClient(tz_aware=True), simple_cookie_storage=True)

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.secret_patch = patch("password_auth.REGISTRATION_SECRET", "letmein")
        self.secret_patch.start()
        self.limiter_patch = patch("password_auth._limiter", SlidingWindowLimiter())
        self.limiter_patch.start()

    async def asyncTearDown(self):
        self.limiter_patch.stop()
        self.secret_patch.stop()
        await super().asyncTearDown()

    async def signup(self, username="alice", password=PASSWORD, password2=None, secret="letmein"):
        return await self.client.post(
            "/signup",
            data={
                "username": username,
                "password": password,
                "password2": password if password2 is None else password2,
                "secret": secret,
            },
            allow_redirects=False,
        )

    async def login(self, username="alice", password=PASSWORD):
        return await self.client.post(
            "/login",
            data={"username": username, "password": password},
            allow_redirects=False,
        )

    def alert(self, html: str) -> str:
        match = re.search(r'role="alert"><strong>([^<]*)', html)
        return match.group(1) if match else ""

    async def test_signup_requires_secret(self):
        response = await self.signup(secret="wrong")
        self.assertEqual(response.status, 403)
        self.assertEqual(self.alert(await response.text()), "Wrong challenge secret")
        app_state = get_app_state(self.app)
        self.assertIsNone(await app_state.db.user.find_one({"_id": "alice"}))

    async def test_signup_validates_input(self):
        for kwargs, error in (
            ({"username": "a b"}, "Username must be 3-20 characters"),
            ({"password": "short"}, "Password must be at least 8 characters"),
            ({"password2": "different-password"}, "Passwords do not match"),
        ):
            with self.subTest(kwargs=kwargs):
                response = await self.signup(**kwargs)
                self.assertEqual(response.status, 400)
                self.assertTrue(self.alert(await response.text()).startswith(error))

    async def test_signup_then_login(self):
        response = await self.signup()
        self.assertEqual(response.status, 302)

        app_state = get_app_state(self.app)
        user_doc = await app_state.db.user.find_one({"_id": "alice"})
        self.assertIsNotNone(user_doc)
        self.assertNotIn("hash", user_doc)
        cred = await app_state.db.user_password.find_one({"_id": "alice"})
        self.assertTrue(verify_password_sync(PASSWORD, cred["hash"]))

        self.client.session.cookie_jar.clear()
        response = await self.login(username="ALICE")
        self.assertEqual(response.status, 302)
        self.assertEqual(response.headers.get("Location"), "/")

    async def test_case_only_duplicate_is_rejected(self):
        await self.signup()
        self.client.session.cookie_jar.clear()
        response = await self.signup(username="Alice")
        self.assertEqual(response.status, 400)
        self.assertEqual(self.alert(await response.text()), "Username is already taken")

    async def test_wrong_password_and_unknown_user_look_the_same(self):
        await self.signup()
        self.client.session.cookie_jar.clear()
        wrong = await self.login(password="wrong-password")
        unknown = await self.login(username="nobody")
        self.assertEqual(wrong.status, 400)
        self.assertEqual(unknown.status, 400)
        self.assertEqual(self.alert(await wrong.text()), self.alert(await unknown.text()))

    async def test_cross_site_post_is_refused(self):
        response = await self.client.post(
            "/login",
            data={"username": "alice", "password": PASSWORD},
            headers={"Sec-Fetch-Site": "cross-site"},
            allow_redirects=False,
        )
        self.assertEqual(response.status, 403)

    async def test_oauth_routes_are_disabled(self):
        for path in ("/login/lichess", "/oauth/lichess"):
            with self.subTest(path=path):
                response = await self.client.get(path, allow_redirects=False)
                self.assertEqual(response.status, 404)


if __name__ == "__main__":
    unittest.main()
