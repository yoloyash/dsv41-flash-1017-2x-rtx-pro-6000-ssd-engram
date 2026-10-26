"""Unit test: SsdEngramTable.prestage == stock int4 lookup kernel over the same rows, and == CPU dequant of file bytes.
Run inside the serving image with the overlay mounted, one GPU, no model load.
Run from the repo root:
    VP=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41
    docker run --rm --gpus device=0 -v $PACK:$PACK:ro -v $PWD/overlay/engram_ssd.py:$VP/common/engram_ssd.py:ro \
      -v $PWD/overlay/engram.py:$VP/common/engram.py:ro -v $PWD/overlay/engram_gather.cpp:$VP/common/engram_gather.cpp:ro \
      -v $PWD/tests:/t:ro -e DSV41_ENGRAM_INT4=1 \
      --entrypoint python3 dsv41-flash-1017-exl3-3bpw-sm120 /t/test_engram_ssd.py $PACK <layer 1|14> <rows 384006168|384016682> <tp rank 0|1>
Add -e DSV41_ENGRAM_SSD_NATIVE=0 for the Python gather pool, -e DSV41_ENGRAM_SSD_RESIDENT_SCALES=1 for resident scales."""
import sys, time, types, torch
from vllm.models.deepseek_v41.common import engram_ssd as S
from vllm.models.deepseek_v41.common.engram import _engram_lookup_int4_kernel

pack, layer_id, n_emb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
heads, tp_rank, tp = 24, int(sys.argv[4]), 2
lh = heads // tp
# head sizes are not needed: owned range is [vs, ve); emulate rank's range as a contiguous half
vs = 0 if tp_rank == 0 else n_emb // 2
ve = n_emb // 2 if tp_rank == 0 else n_emb
emb = types.SimpleNamespace(dim=256, block_size=32, part_n_hash_cols=lh, head_start=tp_rank * lh, n_hash_cols=heads,
                            vocab_start_idx=vs, vocab_end_idx=ve, _num_sms=torch.cuda.get_device_properties(0).multi_processor_count)
dev = torch.device("cuda", 0)
T = 4096
tab = S.SsdEngramTable(emb, layer_id, T, dev, model_dir=pack)
tab.check = True
staged = torch.empty(T, lh, 256, dtype=torch.bfloat16, device=dev)
g = torch.Generator().manual_seed(0)
for t in (1, 6, 48, 4096):
    h = torch.randint(0, n_emb, (t, heads), generator=g, dtype=torch.int32)
    h[torch.rand(t, heads, generator=g) < 0.05] = -1   # dead ids
    t0 = time.perf_counter(); tab.prestage(h, staged); torch.cuda.synchronize(); dt = time.perf_counter() - t0
    # reference: stock kernel reading a GPU copy of exactly the rows of the full table it would index
    loc = h[:, emb.head_start:emb.head_start + lh].reshape(-1).long()
    own = (loc >= vs) & (loc < ve)
    uniq, inv = torch.unique(torch.where(own, loc, 0), return_inverse=True)
    w = tab.w.table[uniq].to(dev); sc = tab.s.table[uniq].to(dev).view(torch.float16)
    ids = torch.where(own, inv, -1).view(t, lh).to(torch.int32).to(dev)
    # put local heads at the right column offset with full width, like the stock call
    full = torch.full((t, heads), -1, dtype=torch.int32, device=dev); full[:, emb.head_start:emb.head_start + lh] = ids
    ref = torch.empty(t, lh, 256, dtype=torch.bfloat16, device=dev)
    _engram_lookup_int4_kernel[(64,)](w, sc, full, ref, 0, uniq.numel(), t * lh, full.stride(0), full.stride(1),
        HEAD_START=emb.head_start, LOCAL_HEADS=lh, TOTAL_HEADS=heads, DIM=256, QUANT_BLOCK=32, BLOCK_R=16, GRID=64)
    torch.cuda.synchronize()
    same = torch.equal(ref.view(torch.int16), staged[:t].view(torch.int16))
    print(f"T={t:5d} rows={t*lh:6d} total {dt*1e3:8.2f} ms  bit-identical to stock kernel: {same}")
    assert same
print("OK")
