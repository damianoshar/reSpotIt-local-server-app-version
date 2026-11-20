-- migrations/016_oauth_pkce.sql
-- PKCE verifier for the Spotify authorization flow.
--
-- A deployment without a client secret is a public client: it proves the
-- code exchange with a verifier (RFC 7636) instead of a secret. The verifier
-- lives with the state row until the callback consumes it.
--
-- The column is nullable, so states created before this migration stay
-- usable: they are exchanged with the client secret, if one is configured.

ALTER TABLE oauth_states ADD COLUMN IF NOT EXISTS code_verifier TEXT;
