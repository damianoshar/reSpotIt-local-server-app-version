"""Tests for the Spotify authorization flow (PKCE and the client secret).

Mocked database and a mocked token endpoint: no browser, no network.

Run from the server directory:

    python -m pytest tests/test_spotify_oauth.py -q
"""
import base64
import hashlib
import unittest
import urllib.parse
from unittest.mock import Mock, patch

from api.services import auth_service

USER_ID = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"


def _challenge_for(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


class _FakePost:
    """Stands in for requests.post and records the form that was sent."""

    def __init__(self, payload=None):
        self.payload = payload or {"access_token": "a", "refresh_token": "r"}
        self.data = None

    def __call__(self, url, data=None, **kwargs):
        self.data = data or {}
        response = Mock()
        response.json.return_value = self.payload
        response.raise_for_status.return_value = None
        return response


class _FakeDb:
    """The smallest session that can hold one OAuth state row."""

    def __init__(self):
        self.rows = {}
        self.wanted = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def add(self, row):
        self.rows[row.state] = row

    def query(self, _model):
        return self

    def filter(self, *_args):
        return self

    def first(self):
        return self.rows.get(self.wanted)

    def delete(self, row=None):
        if row is not None:
            self.rows.pop(row.state, None)
        return 0

    def commit(self):
        return None


class _StateTestCase(unittest.TestCase):
    """Shared setup: a fake session and a fake schema check."""

    def setUp(self):
        self.db = _FakeDb()
        self.patches = [
            patch.object(auth_service, "_ensure_schema", lambda: None),
            patch.object(auth_service, "as_uuid", lambda value: value),
            patch.object(auth_service, "SessionLocal", return_value=self.db),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in self.patches:
            patcher.stop()

    def _stored(self, state):
        self.db.wanted = state
        return self.db.first()


class PkceModeTests(_StateTestCase):
    """Without a client secret the flow proves itself with PKCE."""

    def setUp(self):
        super().setUp()
        self.secret_patch = patch.object(auth_service, "CLIENT_SECRET", "")
        self.secret_patch.start()

    def tearDown(self):
        self.secret_patch.stop()
        super().tearDown()

    def test_the_authorize_url_carries_a_pkce_challenge(self):
        url = auth_service.build_auth_url(USER_ID)
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        self.assertEqual(params["code_challenge_method"], ["S256"])
        self.assertEqual(len(params["code_challenge"][0]), 43)
        state = params["state"][0]
        verifier = self._stored(state).code_verifier
        self.assertEqual(params["code_challenge"][0], _challenge_for(verifier))

    def test_the_exchange_sends_the_verifier_and_no_secret(self):
        post = _FakePost()
        with patch.object(auth_service.requests, "post", post):
            auth_service.exchange_code("the-code", "the-verifier")
        self.assertEqual(post.data.get("code_verifier"), "the-verifier")
        self.assertNotIn("client_secret", post.data)

    def test_an_empty_secret_is_never_sent(self):
        post = _FakePost()
        with patch.object(auth_service.requests, "post", post):
            auth_service.exchange_code("the-code", "the-verifier")
        self.assertNotIn("client_secret", post.data)
        self.assertEqual(post.data.get("client_id"), auth_service.CLIENT_ID)

    def test_the_state_carries_the_verifier_to_the_callback(self):
        state, verifier = auth_service._store_oauth_state(USER_ID)
        self.db.wanted = state
        grant = auth_service.validate_oauth_state(state)
        self.assertEqual(grant.user_id, USER_ID)
        self.assertEqual(grant.code_verifier, verifier)

    def test_a_state_can_be_used_once(self):
        state, _verifier = auth_service._store_oauth_state(USER_ID)
        self.db.wanted = state
        self.assertIsNotNone(auth_service.validate_oauth_state(state))
        # The row is gone, so the second use finds nothing.
        self.db.wanted = state
        self.assertIsNone(auth_service.validate_oauth_state(state))


class ClientSecretModeTests(unittest.TestCase):
    """With a client secret the flow keeps the confidential exchange."""

    def setUp(self):
        self.secret_patch = patch.object(auth_service, "CLIENT_SECRET", "s3cret")
        self.secret_patch.start()

    def tearDown(self):
        self.secret_patch.stop()

    def test_the_exchange_sends_the_secret(self):
        post = _FakePost()
        with patch.object(auth_service.requests, "post", post):
            auth_service.exchange_code("the-code", "the-verifier")
        self.assertEqual(post.data.get("client_secret"), "s3cret")

    def test_the_authorize_url_has_no_challenge(self):
        with patch.object(auth_service, "_store_oauth_state", return_value=("s", "v")):
            url = auth_service.build_auth_url(USER_ID)
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        self.assertNotIn("code_challenge", params)


if __name__ == "__main__":
    unittest.main()
