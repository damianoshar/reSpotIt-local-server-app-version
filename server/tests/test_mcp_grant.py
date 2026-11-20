"""Tests for seeding the assistant's Spotify MCP grant from the app's link.

Mocked database and a temp token-file directory: no MCP, no network.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api.services import auth_service, mcp_grant_service

USER_A = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"
GRANT = {"access_token": "app-access", "refresh_token": "app-refresh"}


class GrantSeedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"SPOTIFY_MCP_DATA_DIR": self.tmp.name})
        self.env.start()
        os.environ.pop("SPOTIFY_MCP_GRANT_SEED", None)
        os.environ.pop("SPOTIFY_MCP_GRANT_USER", None)
        self.linked = [USER_A]
        self.patches = [
            patch.object(mcp_grant_service, "_linked_user_ids", side_effect=lambda: list(self.linked)),
            patch.object(
                auth_service,
                "load_token",
                side_effect=lambda uid: dict(GRANT) if uid in self.linked else None,
            ),
            patch.object(auth_service, "get_valid_token", side_effect=lambda uid: f"fresh-{uid[:4]}"),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in self.patches:
            patcher.stop()
        self.env.stop()
        self.tmp.cleanup()

    @property
    def grant_file(self) -> Path:
        return Path(self.tmp.name) / "tokens.json"

    def read(self) -> dict:
        return json.loads(self.grant_file.read_text(encoding="utf-8"))

    def test_seed_writes_the_fields_the_mcp_validates(self):
        status = mcp_grant_service.seed()
        self.assertEqual(status["status"], "written")
        data = self.read()
        self.assertEqual(data["access_token"], f"fresh-{USER_A[:4]}")
        self.assertEqual(data["refresh_token"], "app-refresh")
        self.assertIsInstance(data["expires_at"], int)
        self.assertTrue(data["scope"])
        self.assertEqual(data["user_id"], USER_A)

    def test_seed_keeps_what_the_mcp_wrote_back(self):
        mcp_grant_service.seed()
        data = self.read()
        data["access_token"] = "mcp-rotated-this"
        data["expires_at"] = 0
        self.grant_file.write_text(json.dumps(data), encoding="utf-8")

        status = mcp_grant_service.seed()
        self.assertEqual(status["status"], "unchanged")
        self.assertEqual(self.read()["access_token"], "mcp-rotated-this")

    def test_seed_rewrites_when_the_recorded_user_is_missing(self):
        mcp_grant_service.seed()
        data = self.read()
        del data["user_id"]
        self.grant_file.write_text(json.dumps(data), encoding="utf-8")

        self.assertEqual(mcp_grant_service.seed()["status"], "written")
        self.assertEqual(self.read()["user_id"], USER_A)

    def test_force_rewrites_even_when_it_matches(self):
        mcp_grant_service.seed()
        self.assertEqual(mcp_grant_service.seed(force=True)["status"], "written")

    def test_seed_skips_when_nobody_has_linked_spotify(self):
        self.linked = []
        status = mcp_grant_service.seed()
        self.assertEqual(status["status"], "skipped")
        self.assertFalse(self.grant_file.exists())

    def test_seed_can_be_disabled(self):
        with patch.dict(os.environ, {"SPOTIFY_MCP_GRANT_SEED": "0"}):
            self.assertEqual(mcp_grant_service.seed()["status"], "disabled")
        self.assertFalse(self.grant_file.exists())

    def test_recorded_user_wins_over_the_earliest_linked_user(self):
        self.linked = [USER_A, USER_B]
        self.grant_file.write_text(json.dumps({"user_id": USER_B}), encoding="utf-8")
        self.assertEqual(mcp_grant_service.target_user_id(), USER_B)

    def test_configured_grant_user_pins_the_account(self):
        self.linked = [USER_A, USER_B]
        self.grant_file.write_text(json.dumps({"user_id": USER_B}), encoding="utf-8")
        with patch.dict(os.environ, {"SPOTIFY_MCP_GRANT_USER": USER_A}):
            self.assertEqual(mcp_grant_service.target_user_id(), USER_A)

    def test_unknown_users_fall_through_to_the_earliest_link(self):
        self.linked = [USER_A]
        self.assertEqual(mcp_grant_service.target_user_id(USER_B), USER_A)

    def test_revoke_ignores_another_user(self):
        mcp_grant_service.seed()
        status = mcp_grant_service.revoke(USER_B)
        self.assertEqual(status["status"], "ignored")
        self.assertTrue(self.grant_file.exists())

    def test_revoke_removes_the_followed_users_file(self):
        mcp_grant_service.seed()
        self.assertEqual(mcp_grant_service.revoke(USER_A)["status"], "removed")
        self.assertFalse(self.grant_file.exists())

    def test_revoke_without_a_file_is_absent(self):
        self.assertEqual(mcp_grant_service.revoke()["status"], "absent")


if __name__ == "__main__":
    unittest.main()
