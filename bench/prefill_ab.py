#!/usr/bin/env python3
"""prefill_ab.py: fresh-prefill A/B against a running server; the KV prefix cache is always missed.
A = a slice of CORPUS no earlier run used (its Engram rows partly cold on SSD).
B = the same text behind a random nonce (KV miss again, its Engram rows now in the page cache).
Each run takes its own slices: ARM selects them (ARM k uses slices k*COUNT .. k*COUNT+COUNT-1), so
configs compared in one session read equally new text.
Env: API_KEY, PORT (8000), MODEL, CORPUS (required: large text the server has not seen, shuffled so slices are
alike: bench/make_corpus_py.py), N (35000 tokens), ARM (0), COUNT (6)."""
import json, os, statistics as st, time, urllib.request, uuid
PORT = os.environ.get("PORT", "8000"); MODEL = os.environ.get("MODEL", "deepseek-v4.1-flash"); KEY = os.environ["API_KEY"]
T = open(os.environ["CORPUS"], errors="replace").read()
N, ARM, COUNT = int(os.environ.get("N", "35000")), int(os.environ.get("ARM", "0")), int(os.environ.get("COUNT", "6"))
SLICE = int(N * 4.6)   # chars; ~4.6 chars/token on this Python source
def req(p):
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": 1, "temperature": 0,
            "chat_template_kwargs": {"thinking": False}}
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t = time.perf_counter(); d = json.load(urllib.request.urlopen(r, timeout=3600)); dt = time.perf_counter() - t
    return d["usage"]["prompt_tokens"], dt
a_rates, b_rates = [], []
for j in range(COUNT):
    k = ARM * COUNT + j
    if (k + 1) * SLICE > len(T):
        raise SystemExit(f"corpus too small for ARM {ARM}: need {(k + 1) * SLICE} chars, have {len(T)}")
    p = T[k * SLICE:(k + 1) * SLICE] + "\n\nExplain what this code does, step by step."
    na, ta = req(p)
    nb, tb = req(f"[{uuid.uuid4().hex}]\n" + p)
    a_rates.append(na / ta); b_rates.append(nb / tb)
    print(f"  slice {k}: A {na} tok {ta:.2f}s {na/ta:.0f} tok/s | B {nb} tok {tb:.2f}s {nb/tb:.0f} tok/s | SSD cost {ta-tb:+.2f}s", flush=True)
print(f"ARM {ARM}: A new text median {st.median(a_rates):.0f} tok/s (mean {st.mean(a_rates):.0f}) | "
      f"B Engram warm median {st.median(b_rates):.0f} tok/s", flush=True)
