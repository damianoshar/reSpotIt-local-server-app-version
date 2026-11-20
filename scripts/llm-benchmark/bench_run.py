"""Run the local-model benchmark against a llama-server instance.

Usage:
  python bench_run.py --port 8080 --model <gguf path> --label ornith9b \
      --cases cases.json --out results_ornith9b.json
"""
import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request

import psutil


def gpu_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def server_pid(port: int) -> int | None:
    for conn in psutil.net_connections(kind="inet"):
        if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port:
            return conn.pid
    return None


class Sampler(threading.Thread):
    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.port = port
        self.stop = threading.Event()
        self.peak_vram = 0
        self.peak_rss = 0
        self.pid = server_pid(port)
        self.proc = psutil.Process(self.pid) if self.pid else None

    def run(self):
        while not self.stop.is_set():
            self.peak_vram = max(self.peak_vram, gpu_used_mib())
            if self.proc is not None:
                try:
                    self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
                except psutil.Error:
                    pass
            self.stop.wait(0.4)


def wait_health(port: int, api_key: str, timeout: float) -> float:
    started = time.time()
    while time.time() - started < timeout:
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/health",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read() or b"{}")
            if body.get("status") in ("ok", "no slot available"):
                return time.time() - started
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit(f"server on {port} did not become healthy in {timeout}s")


def post_case(port: int, api_key: str, model: str, case: dict, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": case["messages"],
        "stream": False,
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
    }
    if case["tools"]:
        body["tools"] = case["tools"]
        body["tool_choice"] = "auto"
    started = time.time()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return {
            "id": case["id"],
            "error": exc.read().decode()[:300],
            "wall_s": round(time.time() - started, 2),
        }
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = [
        ((call.get("function") or {}).get("name"))
        for call in (message.get("tool_calls") or [])
    ]
    expected = case.get("expected") or []
    expected = [expected] if isinstance(expected, str) else expected
    return {
        "id": case["id"],
        "kind": case.get("kind"),
        "expected": expected,
        "called": calls,
        "ok": bool(expected) and bool(set(expected) & set(calls)),
        "route_hit": case.get("route_hit"),
        "wall_s": round(time.time() - started, 2),
        "usage": data.get("usage"),
        "timings": data.get("timings"),
        "finish_reason": choice.get("finish_reason"),
        "content": (message.get("content") or "")[:400],
        "arguments": [
            (call.get("function") or {}).get("arguments")
            for call in (message.get("tool_calls") or [])
        ][:2],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cases", default="cases.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--health-timeout", type=float, default=900)
    args = parser.parse_args()

    load_s = wait_health(args.port, args.api_key, args.health_timeout)
    sampler = Sampler(args.port)
    sampler.start()

    cases = json.load(open(args.cases, encoding="utf-8"))
    results = {
        "label": args.label,
        "model": args.model,
        "load_s": round(load_s, 1),
        "pid": sampler.pid,
        "cases": [],
    }
    for case in cases:
        result = post_case(args.port, args.api_key, args.model, case, args.max_tokens)
        results["cases"].append(result)
        timings = result.get("timings") or {}
        print(
            f"{result['id']:<12} prompt_n={timings.get('prompt_n')} "
            f"cache_n={timings.get('cache_n')} prefill_s={round((timings.get('prompt_ms') or 0)/1000, 2)} "
            f"decode_tps={round(timings.get('predicted_per_second') or 0, 1)} "
            f"called={result.get('called')} ok={result.get('ok')} wall_s={result.get('wall_s')}",
            flush=True,
        )
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)

    sampler.stop.set()
    sampler.join(timeout=5)
    results["peak_vram_mib"] = sampler.peak_vram
    results["peak_rss_gb"] = round(sampler.peak_rss / 1024**3, 2)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(
        f"done {args.label}: load_s={results['load_s']} "
        f"peak_vram={results['peak_vram_mib']} MiB peak_rss={results['peak_rss_gb']} GB"
    )


if __name__ == "__main__":
    main()
