"""Artist filtering on GET /api/v1/albums (repeated `artist` parameters).

The library endpoint accepts one `artist` query parameter per selected name and
matches an album credited to any of them, including collaborator credits.
SQLite in-memory keeps this a pure query test: no Postgres and no Spotify.
"""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_auth.db")
os.environ.setdefault("JWT_SECRET", "test-secret")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.models import Album, Artist, Base, User
from api.routers.albums import _apply_filters


def _session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    user = User(email="albums@test.com", password_hash="x")
    db.add(user)
    db.flush()

    lanegan = Artist(user_id=user.id, name="Mark Lanegan")
    campbell = Artist(user_id=user.id, name="Isobel Campbell")
    radiohead = Artist(user_id=user.id, name="Radiohead")
    db.add_all([lanegan, campbell, radiohead])

    ballads = Album(user_id=user.id, spotify_id="a1", name="Ballad of the Broken Seas")
    ballads.artists = [lanegan, campbell]
    ok = Album(user_id=user.id, spotify_id="a2", name="OK Computer")
    ok.artists = [radiohead]
    db.add_all([ballads, ok])
    db.commit()
    return db


def _names(db, artists):
    q = _apply_filters(
        db.query(Album), db, None, None, None, None, None, artists, None, None,
    )
    return sorted(album.name for album in q.all())


def test_repeated_artists_match_collaborator_credits():
    db = _session()
    try:
        assert _names(db, ["Isobel Campbell", "Radiohead"]) == [
            "Ballad of the Broken Seas", "OK Computer",
        ]
    finally:
        db.close()


def test_artist_filter_does_not_match_partial_names():
    db = _session()
    try:
        assert _names(db, ["Mark"]) == []
    finally:
        db.close()
