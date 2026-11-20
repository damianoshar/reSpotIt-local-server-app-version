"""Compare answer quality on prose/grounding prompts (no tools)."""
import json
import sys
import urllib.request

PORT = int(sys.argv[1])
LABEL = sys.argv[2]

ALBUMS = [
    {"title": "Blue Train", "artist": "John Coltrane", "genre": "jazz", "year": 1958},
    {"title": "Random Access Memories", "artist": "Daft Punk", "genre": "electronic", "year": 2013},
    {"title": "Kind of Blue", "artist": "Miles Davis", "genre": "jazz", "year": 1959},
    {"title": "Immunity", "artist": "Jon Hopkins", "genre": "electronic", "year": 2013},
    {"title": "Aja", "artist": "Steely Dan", "genre": "rock", "year": 1977},
    {"title": "Heligoland", "artist": "Massive Attack", "genre": "trip hop", "year": 2010},
]

DUPES = {
    "exact_groups": [
        {"uri": "spotify:track:1", "title": "Teardrop", "appears": 3},
        {"uri": "spotify:track:2", "title": "Angel", "appears": 2},
    ],
    "near_groups": [
        {"title": "Teardrop", "variants": ["Teardrop", "Teardrop - Remastered 2006"]},
    ],
    "scanned_tracks": 1240,
}

CASES = [
    (
        "p_advice",
        "My local library has 6,258 albums (I counted). In three sentences, suggest how to make it "
        "easier to discover music without deleting anything.",
    ),
    (
        "p_playlist",
        "Here are six albums as JSON:\n"
        + json.dumps(ALBUMS, indent=1)
        + "\nBuild a 25-minute playlist using at most five tracks from these albums, one track per "
        "album maximum. Reply as a numbered list: track, artist, and one short sentence on why it "
        "fits the arc of the playlist.",
    ),
    (
        "p_ground",
        "Summarise this duplicate-detection result for a user in at most four sentences. Only use "
        "the numbers given; if something is not in the data, say so.\n"
        + json.dumps(DUPES, indent=1),
    ),
]

out = []
for case_id, prompt in CASES:
    body = {
        "model": "local",
        "messages": [
            {"role": "system", "content": "You are the reSpotIt music library assistant. Be concrete and concise."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": 500,
        "reasoning_effort": "none",
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer local", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read())
    message = data["choices"][0]["message"]
    timings = data.get("timings", {})
    out.append(
        {
            "id": case_id,
            "content": message.get("content") or "",
            "finish": data["choices"][0].get("finish_reason"),
            "prefill_s": round(timings.get("prompt_ms", 0) / 1000, 2),
            "decode_tps": round(timings.get("predicted_per_second", 0), 1),
        }
    )
    print(f"\n=== {case_id} ({out[-1]['prefill_s']}s prefill, {out[-1]['decode_tps']} tps) ===")
    print(out[-1]["content"])

json.dump(out, open(f"prose_{LABEL}.json", "w", encoding="utf-8"), indent=1)
