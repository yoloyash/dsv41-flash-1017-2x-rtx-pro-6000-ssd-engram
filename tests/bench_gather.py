"""Both Engram layers, one rank, check off: the per-step prestage cost at decode/prefill sizes.
Uniform-random ids = worst case (mostly cold pages); the same ids again = page-cache warm.
Run from the repo root:
    VP=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41
    docker run --rm --gpus device=0 -v $PACK:$PACK:ro -v $PWD/overlay/engram_ssd.py:$VP/common/engram_ssd.py:ro \
      -v $PWD/overlay/engram.py:$VP/common/engram.py:ro -v $PWD/tests:/t:ro -e DSV41_ENGRAM_INT4=1 \
      --entrypoint python3 dsv41-flash-1017-exl3-3bpw-sm120 /t/bench_gather.py $PACK"""
import sys, time, types, torch
from vllm.models.deepseek_v41.common import engram_ssd as S
pack = sys.argv[1]
lh, heads, dev = 12, 24, torch.device("cuda", 0)
tabs = []
for layer, n_emb in ((1, 384006168), (14, 384016682)):
    emb = types.SimpleNamespace(dim=256, block_size=32, part_n_hash_cols=lh, head_start=0, n_hash_cols=heads,
                                vocab_start_idx=0, vocab_end_idx=n_emb // 2, _num_sms=188)
    t = S.SsdEngramTable(emb, layer, 16384, dev, model_dir=pack); t.check = False
    tabs.append((t, torch.empty(16384, lh, 256, dtype=torch.bfloat16, device=dev), n_emb // 2))
g = torch.Generator().manual_seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
def step(hs):
    st = [t.start(h) for (t, _, _), h in zip(tabs, hs)]
    for (t, buf, _), s in zip(tabs, st): t.finish(s, buf)
    torch.cuda.synchronize()
step([torch.zeros(1, heads, dtype=torch.int32)] * 2)
for T, reps in ((6, 50), (48, 50), (2048, 3), (16384, 2)):
    cold, warm = [], []
    for _ in range(reps):
        hs = [torch.randint(0, hi, (T, heads), generator=g, dtype=torch.int32) for _, _, hi in tabs]
        for lst in (cold, warm):
            t0 = time.perf_counter(); step(hs); lst.append(time.perf_counter() - t0)
    med = lambda v: sorted(v)[len(v) // 2] * 1e3
    print(f"T={T:5d} rows/layer={T*lh:6d}  2 layers: random/cold p50 {med(cold):8.2f} ms   warm p50 {med(warm):8.2f} ms")
