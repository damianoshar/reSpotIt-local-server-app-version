"""Routing eval: does the assistant pick the right MCP tool for a question?

Runs the same questions through the live API, so it measures the whole ladder
(L0 hot tools, L1a pre-reveal, L3 search) as the user experiences it.

Mint a token first (the API's own JWT, not an Open WebUI session):

    docker exec -w /app respotit-local-server-app-api-1 python -c \
      "from api.services import user_auth_service as u; \
       print(u.create_access_token('<user-uuid>', '<email>'))" > token.txt

    python scripts/route_eval.py --token-file token.txt

Exit code 0 when every question routed correctly, 1 otherwise. See
docs/AI-TOOL-ROUTING-PLAN.md section 3.2 for what the ladder is meant to do.
"""
import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request

# (id, kind, question, expected tool)
# "library" questions may legitimately be answered through tool_search; catalog
# and playback questions must reach the named tool directly.
CASES = [
    ("lib_stats", "library", "How many albums are in my library?", "library_stats"),
    ("lib_counts", "library", "How many saved tracks and shows do I have?", "get_saved_counts"),
    ("lib_playlists", "library", "List my playlists.", "playlist_list"),
    ("lib_genres", "library", "What genres are in my library?", "library_genres"),
    ("lib_dupes", "library", "Find duplicate tracks in my Liked Songs.", "find_duplicate_saved_tracks"),
    ("lib_hygiene", "library", "Which albums in my library are almost complete?", "library_hygiene"),
    ("cat_search", "catalog", "Search Spotify for albums by Radiohead.", "search"),
    ("cat_album", "catalog", "Find the album Blue Train on Spotify.", "search"),
    ("play_pause", "playback", "Pause the music.", "pause"),
    ("play_volume", "playback", "Turn the volume to 30 percent.", "set_volume"),
]


def ask(url: str, token: str, question: str) -> dict:
    body = {
        "messages": [{"role": "user", "content": question}],
        "use_history": False,
        "persist": False,
        "use_tools": True,
        # Greedy decoding: this measures the router, not sampling variety.
        # (At the API's default 0.7 a direct command occasionally comes back
        # with no tool call at all; see AI-TOOL-ROUTING-PLAN.md section 8.)
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/api/v1/llm/chat",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(request, timeout=600, context=context) as response:
            return json.loads(response.read().decode()).get("data", {})
    except urllib.error.HTTPError as exc:
        # A router that cannot find a tool can loop until the budget guard
        # fires; record that as a miss instead of aborting the run.
        return {"_error": exc.code, "_detail": exc.read()[:200].decode("utf-8", "replace")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://127.0.0.1:18000")
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--json", help="write per-case results here")
    args = parser.parse_args()

    with open(args.token_file, encoding="ascii") as handle:
        token = handle.read().strip()

    rows = []
    for case_id, kind, question, expected in CASES:
        data = ask(args.url, token, question)
        if "_error" in data:
            rows.append({"id": case_id, "expected": expected, "calls": [], "ok": False, "error": data["_error"]})
            print(f"{case_id:<14} expected={expected:<26} HTTP {data['_error']} MISS", flush=True)
            continue
        calls = [call.get("name") for call in (data.get("tool_calls") or [])]
        first = calls[0] if calls else None
        route = data.get("tool_route") or {}
        ok = first == expected or (kind == "library" and first == "tool_search")
        rows.append({"id": case_id, "expected": expected, "calls": calls, "ok": ok})
        print(
            f"{case_id:<14} expected={expected:<26} calls={calls} "
            f"pre={route.get('pre_revealed')} adv={route.get('advertised')} "
            f"tok={route.get('estimated_tokens')} {'OK' if ok else 'MISS'}",
            flush=True,
        )

    hits = sum(1 for row in rows if row["ok"])
    print(f"\n{hits}/{len(rows)} routed correctly")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=1)
    return 0 if hits == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
