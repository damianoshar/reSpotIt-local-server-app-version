"""tool_registry.py - compact, searchable index over the MCP tool surface.

The assistant has 169 MCP tools. Sending every JSON schema costs ~42k prompt
tokens, which alone overflows a 16k-context local model (measured: a 46,190
token request against n_ctx 16384). The registry keeps every tool callable but
advertises only a small surface:

    L0  ten hot playback tools + three meta-tools   (~3.3k tokens, 12.4x less)
        tool_search(query)    -> ranked compact cards
        tool_describe(name)   -> the full argument schema for one tool
        tool_call(name, args) -> execute it on the owning MCP server

Ranking is deterministic field-weighted BM25 over name (x3), tags (x2) and the
first part of the description (x1), with intent boosts: "build me a playlist"
prefers create_playlist, a request without delete/remove never surfaces
playlist_delete, and a match that only exists in unrelated prose is demoted.

Nothing here talks to the network and nothing is sent whole: a card is ~70
characters, so the model reads tool *names and purposes* and asks for a schema
only when it is about to call one.

See docs/AI-TOOL-ROUTING-PLAN.md.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

# Tools advertised natively on every request: the calls an assistant makes
# without thinking. Names are matched against the routed public names, so a
# server that had to be prefixed (``spotify_search``) still counts.
HOT_TOOLS: tuple[str, ...] = (
    "search",
    "get_now_playing",
    "play",
    "play_from_search",
    "pause",
    "skip_next",
    "set_volume",
    "get_queue",
    "add_to_queue",
    "get_devices",
)

META_NAMES: frozenset[str] = frozenset({"tool_search", "tool_describe", "tool_call"})

SUMMARY_CHARS = 140
CARD_CHARS = 180
MAX_RESULTS = 12
DEFAULT_RESULTS = 8

# Where a query term matched matters more than how often it matched. A tool
# called ``playlist_create`` is about creating playlists even though its
# description only says "playlist" once.
FIELD_WEIGHTS = {"name": 3.0, "tags": 2.0, "summary": 1.0}

# The first ~800 characters of a description are searched; the tail of the
# longest ones is usually examples and caveats that dilute the length norm.
DESC_CHARS = 800

# A request phrased as an action should prefer a tool that performs it.
CREATE_VERBS = frozenset(
    "create make build generate new compose curate start setup add save follow "
    "import".split()
)
DESTROY_VERBS = frozenset(
    "delete remove clear drop purge wipe unfollow clean cancel reset destroy "
    "erase".split()
)
# Name fragments that mark a tool as destructive.
_DESTRUCTIVE = (
    "delete", "remove", "clear", "purge", "wipe", "unfollow", "cancel",
    "clean", "drop", "reset", "erase",
)
# "Create" is the canonical verb for "build/make me a ...". "Add" and "save"
# are actions too, but they modify an existing thing, so they earn less.
_STRONG_CREATIVE = ("create", "make", "new", "compose", "build", "generate")
_WEAK_CREATIVE = ("add", "save", "follow", "import", "start")

STRONG_CREATIVE_BOOST = 3.0
WEAK_CREATIVE_BOOST = 1.5
DESTRUCTIVE_PENALTY = 4.0
EXACT_NAME_BOOST = 5.0
ALL_NAME_TOKENS_BOOST = 4.0
HOT_BOOST = 0.25
COUNT_BOOST = 2.0

# The user's saved library is a different place from the Spotify catalog. A
# request that names it ("search the library", "my saved albums") must not be
# answered by the catalog `search` tool just because its name contains
# "search". The catalog tools are demoted and the library tools are lifted;
# the tool descriptions carry the same warning for the model.
_LIBRARY_INTENT = re.compile(
    r"\b(?:library|libraries|saved|collection|my\s+own|i\s+own|"
    r"do\s+i\s+have|have\s+i\s+got)\b"
)
LIBRARY_BOOST = 4.0
LIBRARY_LIST_BOOST = 2.0
LIBRARY_DEFAULT_ALBUM_BOOST = 3.0
CATALOG_SEARCH_PENALTY = 8.0
MEDIA_TYPE_MISMATCH_PENALTY = 7.0
_CATALOG_SEARCH_TOOLS = frozenset(
    {"search", "spotify_search", "catalog_search", "search_deep", "play_from_search"}
)
_LIBRARY_LIST_TOOLS = frozenset(
    {
        "library_list_albums",
        "library_list_artists",
        "library_stats",
        "library_genres",
        "library_genre_report",
        "library_export_csv",
        "library_get_album",
        "library_get_artist",
        "library_album_tracks",
        "library_hygiene",
        "get_saved_albums",
        "get_saved_tracks",
        "get_saved_counts",
        "search_saved_albums",
        "search_saved_tracks",
        "search_saved_shows",
        "search_saved_episodes",
        "search_saved_audiobooks",
    }
)
_MEDIA_TYPES = frozenset(
    {"album", "artist", "track", "song", "show", "episode", "audiobook", "playlist"}
)
_MEDIA_TYPE_ALIASES = {"song": "track"}
_MEDIA_TYPE_TOOLS = {
    "library_list_albums": "album",
    "library_list_artists": "artist",
    "get_saved_albums": "album",
    "get_saved_tracks": "track",
    "search_saved_albums": "album",
    "search_saved_tracks": "track",
    "search_saved_shows": "show",
    "search_saved_episodes": "episode",
    "search_saved_audiobooks": "audiobook",
}

# Appended to the advertised schema only: a small model reads the hot `search`
# description and calls it for "search the library". Naming the saved-library
# tools there is what makes it choose them.
_DESCRIPTION_NOTES: dict[str, str] = {
    "search": (
        " This searches the whole Spotify catalog, not the user's saved "
        "library. For the user's own library use library_list_albums, "
        "library_list_artists, search_saved_albums or search_saved_tracks."
    ),
    "spotify_search": (
        " This searches the whole Spotify catalog, not the user's saved "
        "library. For the user's own library use library_list_albums, "
        "library_list_artists, search_saved_albums or search_saved_tracks."
    ),
    "catalog_search": (
        " This searches the whole Spotify catalog, not the user's saved "
        "library."
    ),
    "play_from_search": (
        " This searches the whole Spotify catalog, not the user's saved "
        "library."
    ),
}
# A description-only match is weak evidence: it is how "makes a trimmed
# surface usable" used to outrank every playlist tool for "make a playlist".
DESCRIPTION_ONLY_SCALE = 0.45
_COUNT_NAMES = ("stats", "stat", "count", "total")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words carry no signal anywhere: they are dropped from tool names,
# descriptions and queries alike.
_FUNCTION_STOP = frozenset(
    """
    a an the to of for and or in on with without my me i is are was it that this
    be by at from as please can could would you your we our some any then than
    """.split()
)

# Extra words dropped from the *query*: question scaffolding. They never name
# a tool. Tool names keep them, because playlist_list and get_saved_albums are
# named that way.
_QUERY_STOP = _FUNCTION_STOP | frozenset(
    """
    how many much do does did have has had there what which who whom whose where
    when why
    """.split()
)

# Generic request verbs are kept in the query but count for little: "list my
# playlists" needs "list" to find playlist_list, yet "show" must not drag in
# every TV-show tool.
_WEAK_QUERY_TERMS = frozenset(
    "show find list give tell look need want display".split()
)
WEAK_TERM_WEIGHT = 0.3


def _tokens(text: str, stop: frozenset[str] = _FUNCTION_STOP) -> list[str]:
    """Lowercase word tokens, with a crude plural strip.

    ``get_now_playing`` becomes ``get now playing`` because underscores are
    separators, so "what is playing now" matches the tool name directly.
    """
    out: list[str] = []
    for raw in _TOKEN_RE.findall(str(text).replace("_", " ").lower()):
        if raw in stop or len(raw) < 2:
            continue
        if len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss"):
            raw = raw[:-1]
        out.append(raw)
    return out


@dataclass(frozen=True)
class ToolCard:
    """One tool, reduced to what a model needs in order to decide."""

    name: str
    server: str
    summary: str
    tags: tuple[str, ...]
    required: tuple[str, ...]

    def render(self) -> str:
        line = f"{self.name} [{self.server}] {self.summary}"
        if self.required:
            line += f" (args: {', '.join(self.required)})"
        return line[:CARD_CHARS]


@dataclass(frozen=True)
class Registry:
    """Cards plus the full schemas, indexed for field-weighted BM25 search."""

    cards: tuple[ToolCard, ...]
    by_name: dict[str, ToolCard]
    schemas: dict[str, dict[str, Any]]
    # name -> term -> weighted frequency (name x3, tags x2, summary x1)
    tf: dict[str, dict[str, float]]
    lengths: dict[str, float]
    df: dict[str, int]
    avg_len: float

    def __len__(self) -> int:
        return len(self.cards)


def _summary(description: str) -> str:
    text = " ".join(str(description or "").split())
    if not text:
        return ""
    for mark in (". ", " - ", "; "):
        idx = text.find(mark)
        if 0 < idx <= SUMMARY_CHARS:
            text = text[:idx]
            break
    if len(text) > SUMMARY_CHARS:
        text = text[: SUMMARY_CHARS - 1].rstrip() + "\u2026"
    return text


_TAG_GROUPS = (
    "library", "playlist", "catalog", "search", "queue", "device",
    "artist", "album", "track", "recent", "genre", "user",
)


def _tags(name: str, description: str) -> tuple[str, ...]:
    """Cheap grouping tags derived from the name and description."""
    lowered = f"{name} {description}".lower()
    return tuple(group for group in _TAG_GROUPS if group in lowered)


def _guess_server(name: str) -> str:
    """Fallback owning-server label, for display only."""
    for prefix in ("library_", "playlist_", "catalog_", "spotify_status"):
        if name.startswith(prefix):
            return "respotit"
    return "spotify"


def build(
    tools: Iterable[dict[str, Any]],
    server_of: Mapping[str, str] | None = None,
) -> Registry:
    """Build the registry from OpenAI function-calling tool dicts."""
    tool_list = [t for t in tools if isinstance(t, dict)]
    server_map = dict(server_of or {})

    cards: list[ToolCard] = []
    by_name: dict[str, ToolCard] = {}
    schemas: dict[str, dict[str, Any]] = {}
    tf: dict[str, dict[str, float]] = {}
    lengths: dict[str, float] = {}
    df: dict[str, int] = {}

    for tool in tool_list:
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        description = str(fn.get("description") or "")
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        required = params.get("required")
        card = ToolCard(
            name=name,
            server=server_map.get(name) or _guess_server(name),
            summary=_summary(description),
            tags=_tags(name, description),
            required=tuple(str(r) for r in required) if isinstance(required, list) else (),
        )
        cards.append(card)
        by_name[name] = card
        note = _DESCRIPTION_NOTES.get(name)
        if note:
            noted_fn = dict(fn)
            noted_fn["description"] = f"{description}{note}"
            schemas[name] = {**tool, "function": noted_fn}
        else:
            schemas[name] = tool

        weights: dict[str, float] = {}
        for term in _tokens(name):
            weights[term] = weights.get(term, 0.0) + FIELD_WEIGHTS["name"]
        for term in card.tags:
            weights[term] = weights.get(term, 0.0) + FIELD_WEIGHTS["tags"]
        for term in _tokens(description[:DESC_CHARS]):
            weights[term] = weights.get(term, 0.0) + FIELD_WEIGHTS["summary"]
        tf[name] = weights
        lengths[name] = sum(weights.values())
        for term in weights:
            df[term] = df.get(term, 0) + 1

    avg = (sum(lengths.values()) / len(lengths)) if lengths else 0.0
    return Registry(
        cards=tuple(cards),
        by_name=by_name,
        schemas=schemas,
        tf=tf,
        lengths=lengths,
        df=df,
        avg_len=avg,
    )


def wants_library(query: str) -> bool:
    """True when the request names the user's saved library, not the catalog.

    ``search`` is a hot tool, and its name alone used to win "search the
    library" even when the library tools were advertised next to it. The
    ranking and the advertised surface both use this one test.
    """
    return bool(_LIBRARY_INTENT.search(str(query or "").lower()))


def is_library_tool(name: str) -> bool:
    """True for a tool that reads the user's saved library."""
    return str(name or "") in _LIBRARY_LIST_TOOLS


def rank(reg: Registry, query: str, limit: int = DEFAULT_RESULTS) -> list[tuple[float, ToolCard]]:
    """Rank cards against a plain-language need (BM25, deterministic).

    Returns ``(score, card)`` pairs. ``search()`` drops the score; the pre-reveal
    in ``llm_service`` needs it to tell a strong match from a stray keyword.
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = DEFAULT_RESULTS
    limit = max(1, min(limit, MAX_RESULTS))

    terms = _tokens(query, _QUERY_STOP)
    if not terms:
        return [(0.0, card) for card in reg.cards[:limit]]

    term_set = set(terms)
    wants_create = bool(term_set & CREATE_VERBS)
    wants_destroy = bool(term_set & DESTROY_VERBS)
    raw_query = str(query).lower()
    wants_library_intent = wants_library(raw_query)
    requested_types = term_set & _MEDIA_TYPES
    requested_types = {_MEDIA_TYPE_ALIASES.get(term, term) for term in requested_types}
    wants_count = any(
        key in raw_query
        for key in ("how many", "count", "total", "stats", "statistic", "number of")
    )
    # Verbs say what to do; the remaining terms say what it is done to.
    content_terms = term_set - CREATE_VERBS - DESTROY_VERBS - _WEAK_QUERY_TERMS

    k1, b = 1.2, 0.75
    total = max(1, len(reg.cards))
    lowered_query = str(query).lower().replace(" ", "_")
    scored: list[tuple[float, ToolCard]] = []

    for card in reg.cards:
        weights = reg.tf.get(card.name) or {}
        length = reg.lengths.get(card.name) or 1.0

        score = 0.0
        for term in terms:
            freq = weights.get(term, 0.0)
            if freq <= 0:
                continue
            df = reg.df.get(term, 0)
            idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * length / (reg.avg_len or 1.0))
            weight = WEAK_TERM_WEIGHT if term in _WEAK_QUERY_TERMS else 1.0
            score += weight * idf * (freq * (k1 + 1)) / denom
        if score <= 0:
            continue

        name = card.name.lower()
        name_terms = set(_tokens(card.name))
        anchor_terms = name_terms | set(card.tags)
        # "make a playlist": a match that only exists in prose about something
        # else must not outrank a tool actually named after the object.
        if content_terms and not (content_terms & anchor_terms):
            score *= DESCRIPTION_ONLY_SCALE
        # An exact name hit should beat a description that merely mentions it.
        if name in lowered_query:
            score += EXACT_NAME_BOOST
        # "skip to next track" should find skip_next: every name token present.
        if name_terms and name_terms <= term_set:
            score += ALL_NAME_TOKENS_BOOST
        # "build me a calm playlist" should prefer a create tool over one that
        # merely appends to an existing playlist.
        if wants_create and any(frag in name for frag in _STRONG_CREATIVE):
            score += STRONG_CREATIVE_BOOST
        elif wants_create and any(frag in name for frag in _WEAK_CREATIVE):
            score += WEAK_CREATIVE_BOOST
        # "how many albums do I have" wants the stats tool, not a list.
        if wants_count and any(frag in name for frag in _COUNT_NAMES):
            score += COUNT_BOOST
        # Never surface playlist_delete for a request that did not ask to
        # delete anything.
        if not wants_destroy and any(frag in name for frag in _DESTRUCTIVE):
            score -= DESTRUCTIVE_PENALTY
        # "search the library" names the saved library: lift the tools that
        # read it and push the catalog search tools down. Without this the hot
        # `search` tool wins on its name alone and answers from all of Spotify.
        if wants_library_intent:
            if card.name in _LIBRARY_LIST_TOOLS:
                score += LIBRARY_BOOST + LIBRARY_LIST_BOOST
                # A generic search of the saved album library should use the
                # album list, which can match both album titles and credited
                # artists. The artist list is the right choice only when the
                # user specifically asks for artists.
                if not requested_types and card.name == "library_list_albums":
                    score += LIBRARY_DEFAULT_ALBUM_BOOST
            if card.name in _CATALOG_SEARCH_TOOLS:
                score -= CATALOG_SEARCH_PENALTY
            tool_type = _MEDIA_TYPE_TOOLS.get(card.name)
            if requested_types and tool_type and tool_type not in requested_types:
                score -= MEDIA_TYPE_MISMATCH_PENALTY
        if card.name in HOT_TOOLS:
            score += HOT_BOOST

        if score > 0:
            scored.append((score, card))

    scored.sort(key=lambda pair: (-pair[0], pair[1].name))
    return scored[:limit]


def search(reg: Registry, query: str, limit: int = DEFAULT_RESULTS) -> list[ToolCard]:
    """Cards for a plain-language need, best first (see ``rank``)."""
    return [card for _, card in rank(reg, query, limit)]


def describe(reg: Registry, name: str) -> dict[str, Any] | None:
    """The full OpenAI tool dict for one name, or None."""
    return reg.schemas.get(str(name or "").strip())


def hot_tools(reg: Registry) -> list[dict[str, Any]]:
    """The native schemas for HOT_TOOLS that actually exist on this server."""
    out: list[dict[str, Any]] = []
    for hot in HOT_TOOLS:
        for candidate in (hot, f"spotify_{hot}"):
            schema = reg.schemas.get(candidate)
            if schema is not None:
                out.append(schema)
                break
    return out


def render_search(query: str, cards: list[ToolCard], total: int) -> str:
    """The text a tool_search call returns to the model."""
    if not cards:
        return (
            f'No tool matched "{query}" ({total} tools are registered). '
            "Try different words, or answer without a tool."
        )
    lines = [f'{len(cards)} of {total} tools for "{query}":']
    for card in cards:
        lines.append(f"- {card.render()}")
    lines.append(
        "Call tool_call(name, arguments) to run one; tool_describe(name) "
        "returns its full argument schema."
    )
    return "\n".join(lines)


META_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "tool_search",
            "description": (
                "Find assistant tools by plain-language need. Returns compact "
                "cards (name, purpose, required args). Use it whenever no "
                "advertised tool fits the request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What you need, e.g. 'queue a song', "
                            "'add to playlist', 'top genres'."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum cards to return (1-12).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_describe",
            "description": (
                "Return the full argument schema for one tool name found with "
                "tool_search. Call this before tool_call when the arguments "
                "are not obvious from the card."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Tool name."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_call",
            "description": (
                "Run a tool found with tool_search. Pass arguments matching "
                "its schema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Tool name."},
                    "arguments": {
                        "type": "object",
                        "description": "Arguments for the tool; omit for none.",
                    },
                },
                "required": ["name"],
            },
        },
    },
]
