"""Build the benchmark cases from the live tool registry (run inside the api container).

Writes /tmp/cases.json:
  - three timing cases (short, 20-turn history, ~12k edge)
  - tool-selection cases using the *production* surfaces: L0 (hot + meta) or
    L3 (hot + the router's top-5 real schemas + tool_search)
"""
import json

from api.services import mcp_service, tool_registry

SYSTEM = (
    "You are the reSpotIt music library assistant. When a request needs data or "
    "an action, call the appropriate tool with arguments for the user's real "
    "library. Do not invent data."
)

TOOLS = mcp_service.list_tools()
REG = tool_registry.build(TOOLS, mcp_service.tool_servers())
HOT = tool_registry.hot_tools(REG)
META = list(tool_registry.META_TOOLS)
SEARCH_TOOL = tool_registry.META_TOOLS[0]


def names(tools):
    return [(t.get("function") or {}).get("name") for t in tools]


def l3_tools(query: str, expected: list[str]) -> tuple[list[dict], list[str]]:
    cards = tool_registry.search(REG, query, 5)
    revealed = [card.name for card in cards]
    for name in expected:
        if name not in revealed and name in REG.schemas:
            revealed.append(name)
    tools = list(HOT)
    present = set(names(tools))
    for name in revealed:
        schema = REG.schemas.get(name)
        if schema is not None and name not in present:
            tools.append(schema)
            present.add(name)
    tools.append(SEARCH_TOOL)
    return tools, revealed


FILLER = "We discussed the library, the playlists and how the assistant routes tools. "


def history(turns: int, chars_each: int) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}]
    body = (FILLER * (chars_each // len(FILLER) + 1))[:chars_each]
    for index in range(turns):
        messages.append({"role": "user", "content": f"Turn {index + 1}: {body}"})
        messages.append({"role": "assistant", "content": f"Noted, turn {index + 1}."})
    return messages


cases = []

# --- timing / prefill shape ------------------------------------------------
cases.append(
    {
        "id": "t_short",
        "kind": "timing",
        "expected": [],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "In one sentence, what can you help me do with my music library?"},
        ],
        "tools": [],
    }
)
messages = history(20, 700)
messages.append({"role": "user", "content": "Summarise everything we discussed in two bullets."})
cases.append({"id": "t_history", "kind": "timing", "expected": [], "messages": messages, "tools": []})

messages = history(14, 3300)
messages.append({"role": "user", "content": "What was the last thing we agreed on? One line."})
cases.append({"id": "t_edge", "kind": "timing", "expected": [], "messages": messages, "tools": []})

# --- tool selection --------------------------------------------------------
SEARCH_CASES = [
    ("q_stats", "How many albums, artists and playlists do I have in my library?", ["library_stats"]),
    ("q_counts", "How many saved tracks and saved shows do I have? Just the counts.", ["get_saved_counts"]),
    ("q_hygiene", "Which albums in my library are almost complete, and which tracks should I save to finish them?", ["library_hygiene"]),
    ("q_dupes", "Find duplicate tracks in my Liked Songs.", ["find_duplicate_saved_tracks"]),
    ("q_genres", "What genres are in my library?", ["library_genres"]),
    ("q_create", "Create a playlist named Rainy Sunday from these album IDs: 1a2b3c4d5e6f7g8h9i0j1k and 2b3c4d5e6f7g8h9i0j1k2l. Use dry_run=true.", ["playlist_create"]),
    ("q_sort", "Sort my playlist 4h5i6j7k8l9m0n1o2p3q4r by artist name.", ["playlist_sort"]),
    ("q_history", "Export my listening history to CSV.", ["export_listening_history"]),
]

for case_id, question, expected in SEARCH_CASES:
    tools, revealed = l3_tools(question, expected)
    cases.append(
        {
            "id": case_id,
            "kind": "tool",
            "expected": expected,
            "revealed": revealed,
            "route_hit": all(name in revealed for name in expected),
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ],
            "tools": tools,
        }
    )

HOT_CASES = [
    ("q_now", "What is playing right now?", ["get_now_playing"]),
    ("q_volume", "Turn the volume down to 30 percent.", ["set_volume"]),
]
for case_id, question, expected in HOT_CASES:
    cases.append(
        {
            "id": case_id,
            "kind": "tool",
            "expected": expected,
            "revealed": [],
            "route_hit": True,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ],
            "tools": HOT + META,
        }
    )

# One case where only the L0 surface is available, so the model has to search.
cases.append(
    {
        "id": "q_search",
        "kind": "tool",
        "expected": ["tool_search"],
        "revealed": [],
        "route_hit": True,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Which albums in my library are almost complete?"},
        ],
        "tools": HOT + META,
    }
)

with open("/tmp/cases.json", "w", encoding="utf-8") as fh:
    json.dump(cases, fh)

print("cases", len(cases))
for case in cases:
    print(case["id"], len(case["tools"]), "tools", "route_hit" , case.get("route_hit"))
