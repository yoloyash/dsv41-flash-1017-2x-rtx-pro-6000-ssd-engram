# DeepSeek-V4.1-Flash-1017 on 2x RTX PRO 6000 with 128 GB of host RAM: Engram tables on SSD

A serving recipe for [diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)
on a host with **128 GB of RAM** instead of ~256 GB. No weights and no fork: it builds diffbot's image with diffbot's scripts and
runs diffbot's `serve.sh`, plus four bind-mounted vLLM files that keep the two int4 Engram tables (~111 GB) in the pack's shards
on NVMe instead of pinned host RAM. The rows the model sees are bit-identical.

| | diffbot recipe | this repo |
|---|---|---|
| Host RAM | ~230 GiB pinned | ~82 GiB + page cache (container capped at 120g) |
| 1M context, vision, tool calls, DSpark | yes | yes |

## Results

Test host below; diffbot's numbers are from their model card on a PCIe 5.0 / DDR5 host, so the gap is SSD and host together.
Raw output for every number: [`results/raw-logs.txt`](results/raw-logs.txt).

**Decode**, tok/s, DSpark k=5, diffbot's `bench-fixed.py`, warm page cache

| | diffbot | this repo |
|---|---:|---:|
| 1 stream, 2K prompt | 156.9 | **147.8 – 157.3** |
| 1 stream, 46K prompt | 172.8 | **156.0 – 156.1** |
| 4 streams, 46K prompts, aggregate | 362.9 | **269.0 – 269.7** |

**Prefill**, tok/s, ~40K-token prompt, nothing in the KV cache

| | diffbot | this repo |
|---|---:|---:|
| text the server has never seen | 11,157 – 11,467 | **7,263 – 7,457** |
| text whose Engram rows are already cached | — | 9,394 |

**Quality and long context**

| | diffbot | this repo |
|---|---:|---:|
| GSM8K-200, thinking off | 98.5% | **98.5%** |
| HumanEval pass@1, thinking off | 97.0% | **97.6%** |
| ~1.04M-token needle | found, 110.4 s | **found, 104.0 s** |
| peak VRAM during it | 96.6 / 97.0 GB | 95.5 / 95.5 GB |

Notes:
- Decode: two runs (restart per run).
- Prefill is [`bench/prefill_ab.py`](bench/prefill_ab.py): median of 8 never-seen slices of Python source (38K-60K tokens),
  measured in the first minutes after a restart (cold cache), two runs. A server that has been up for a while does better.
- Right after a boot, decode is slower until the page cache warms up (80.8 / 114.9 / 218.8 on an earlier version).
- DSpark off: 84.3 / 87.3 / 205.2 tok/s. Leave it on.
- Quality, the needle and DSpark-off were measured before prefetch / resident scales / eviction / per-layer wait. Those don't
  change any row the model sees, only speed.

### What each change is worth

Prefill on new text, median tok/s, 8 fresh prompts and a restart per row (run-to-run spread about ±10%):

| | tok/s |
|---|---:|
| prestage only (start / end of the session) | 4,132 / 4,293 |
| + evict loader pages from the page cache | 5,129 / 4,523 |
| + evict + prefetch | 5,977 |
| + evict + resident scales | 5,380 |
| + evict + prefetch + resident scales | 6,925 / 6,959 / 6,775 |
| **+ per-layer wait (default)** | **7,457 / 7,263** |

The last row and the last two values above it are one session, run off / on / on / off with fresh text each time.

Decode (1 stream 2K / 1 stream 46K / 4 streams), tok/s:

| | tok/s |
|---|---|
| prestage only | 132.7 / 132.8 / 249.3 |
| + evict + prefetch + resident scales | 150.1 / 143.5 / 249.3 |
| same, a later session (start / end) | 143.9 / 144.6 / 258.2, 154.1 / 138.6 / 271.3 |
| **+ native row gather (default)**, same session | **157.3 / 156.1 / 269.0, 147.8 / 156.0 / 269.7** |

The native gather is worth ~+10% on one 46K stream (both runs above both runs without it); at 2K and at four streams the runs
overlap. It leaves prefill where it was (7,561 / 7,419 without, 7,523 / 8,131 with). The per-layer wait only changes prefill steps
(512+ tokens, which run outside the CUDA graphs); decode steps take the same path as before.

### Test host

Threadripper PRO 5955WX (Zen 3, no AVX-512, so diffbot's GPU-only path), 8x 16 GB DDR4-2933, 2x RTX PRO 6000 Workstation (600 W)
on PCIe 4.0 x16, Engram shards on a WD_BLACK SN7100 4 TB (~195K random 4 KiB reads/s across the tables' 110 GB, fio), Linux 7.0.

## Quick start

Needs 2x RTX PRO 6000 (96 GB, sm_120), an NVIDIA driver with CUDA ≥ 13, Docker with the NVIDIA runtime, ~128 GB RAM and ~330 GB on
a fast NVMe.

```bash
# 1. diffbot's pack (weights + recipe/), on the NVMe
hf download diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 --local-dir /models/dsv41-1017
export PACK=/models/dsv41-1017

# 2. diffbot's image and kernels, with diffbot's scripts (~30-60 min)
(cd $PACK/recipe && ./docker/build.sh && ./kernels/build.sh)

# 3. This repo
git clone https://github.com/yoloyash/dsv41-flash-1017-2x-rtx-pro-6000-ssd-engram && cd dsv41-flash-1017-2x-rtx-pro-6000-ssd-engram

# 4. Optional: parity mode (eager, every step checked against the shard bytes). Send a few requests, check the log.
API_KEY=change-me ENGRAM_SSD_CHECK=1 EAGER=1 WARMUP=0 serve/serve-ssd.sh

# 5. Serve
API_KEY=change-me serve/serve-ssd.sh
```

Every knob of diffbot's `serve.sh` still works (`PORT`, `SPEC`, `MAX_NUM_SEQS`, `MAX_MODEL_LEN`, `VISION`, ...). This repo adds:

| setting | default | |
|---|---|---|
| `ENGRAM_SSD` | `1` | `0` = diffbot's pinned-RAM path |
| `ENGRAM_SSD_PREFETCH` | `16384` | tokens of the next prefill chunk to read ahead; `0` = off |
| `ENGRAM_SSD_RESIDENT_SCALES` | `1` | keep the Engram scales in RAM (11.4 GiB); `0` gives that back |
| `ENGRAM_SSD_EVICT` | `1` | drop the loader's leftover shard pages from the page cache after load |
| `ENGRAM_SSD_DEFER` | `1` | prefill: wait for each Engram layer's rows only when the forward reaches that layer; `0` = all before |
| `ENGRAM_SSD_NATIVE` | `1` | row gather on a C++ thread pool (built at load); `0` = the Python pool |
| `ENGRAM_SSD_THREADS` | `128` | row-gather threads |
| `ENGRAM_SSD_CHECK` | `0` | parity mode |
| `CONTAINER_MEM` | `120g` | container cap; page cache counts against it, so it is also the Engram cache size |
| `SSD_PLAN` | `plan-1017-agent30-c512-p375.json` | host-expert plan from `plans/`; empty = diffbot's default |

## How it works

DeepSeek-V4.1 looks up 48 hashed rows per token in two ~384M-row Engram tables. diffbot pins them in host RAM and the GPU reads
them inside the CUDA graphs. Here:

1. **Prestage** (nktlabs' design, ported): before each step, the runner hashes the batch, gathers its rows from an mmap of the
   shard on 128 threads, and dequantizes them on the GPU with diffbot's own int4 kernel into fixed buffers. The CUDA-graphed
   forward only reads those buffers. Anything that doesn't line up raises instead of running on stale rows.
2. **Prefetch**: a long prompt is prefilled in 16K-token chunks. While one chunk computes, the next chunk's rows are read into
   the page cache, so its gather takes ~20 ms instead of ~1.2 s.
3. **Resident scales**: each row is 128 B of weights plus 16 B of scales stored separately, so a cold row was two SSD reads. The
   scales now live in RAM: one read.
4. **Eviction**: loading leaves ~35 GB of other shards in the page cache, and the kernel was evicting Engram pages first
   (`MADV_RANDOM` mappings get no recency since Linux 6.3). Before: 6.6 of 37.9 GiB cached was Engram. After: nearly all of it.
5. **Per-layer wait** ([Project Maya](https://github.com/mw00/project-maya)'s ordering): prefill chunks run outside the CUDA
   graphs, so they don't wait for every row before the forward. Layer 1's rows are read first, alone; layer 14's start once
   layer 1's are in and load while layers 1-13 compute. A new prompt's Engram wait goes from ~1.24 s to ~0.88 s.
6. **Native row gather**: the rows are copied by a small C++ thread pool ([`overlay/engram_gather.cpp`](overlay/engram_gather.cpp),
   built into the torch extension cache at load, ~15 s the first time), a few rows at a time to whichever thread is free. A decode
   step needs a few hundred rows, mostly cached, split into ~50 Python tasks per layer whose interpreter overhead was most of the
   cost: the gather between decode steps (the GPU waits for it) took 2.7-5.5 ms, p99 12-17 ms, at four streams. Now 1.1-2.2 ms,
   p99 ~3.5 ms.
7. **p375 plan**: diffbot's default plan with its 375 hottest host experts moved back to VRAM (+10% at 4 streams).

Where the remaining gap is:
- a new prompt still waits ~0.9 s on cold reads: ~0.6 s for layer 1's rows of the first chunk (nothing runs before them),
  ~0.25 s on the second chunk, whose prefetch now starts at layer 14. The SSD's random-read rate is the limit
- decode still waits ~1-2 ms per step between steps (hash, gather, copy, dequant), with the GPU idle
- with every row cached, prefill is 9.4K vs 11.2K; not isolated, most likely PCIe 4.0 vs 5.0 for the host experts

Tried and dropped: vLLM's adaptive DSpark verification (`enable_adaptive_verification`: it needs a variable-length indexer
kernel that exists for sm_90 and sm_100 only, so vLLM refuses to start with it on sm_120), an io_uring reader (the SSD tops out
at ~200K random reads/s, which the threads already reach), prefetching two chunks ahead (same as one), and the per-layer wait
with both layers read at once (it moves ~0.4 s from the first chunk to the second: no gain).

## Fidelity

- [`tests/test_engram_ssd.py`](tests/test_engram_ssd.py): staged rows bit-identical to diffbot's kernel, both layers, both GPUs,
  with and without resident scales.
- Parity mode: each step re-reads sampled rows with `pread` and compares bit for bit, checks the hashes against the stock path, and
  checks that prefetch hashes the right tokens. 699 steps (long prompts, concurrent prefills, 16 decode streams, images): zero
  mismatches. Two more runs with the per-layer wait on (a 46K prompt, four 24K-34K prompts at once, decode streams): zero. One
  with the native gather on (a 50K prompt, four 24K-30K prompts at once, decode streams; 199 hash-checked steps per rank): zero.
  The bit-identity test passes with the native gather on and off.
- GSM8K / HumanEval match diffbot's figures. Fidelity to the original weights is diffbot's (their `recipe/kl/`).

## Files

```
overlay/engram_ssd.py     new: shard reader, gather, prefetch, resident scales, eviction, parity check
overlay/engram_gather.cpp new: the native row gather (built at load with torch.utils.cpp_extension)
overlay/engram.py         vLLM files from diffbot's image + hooks; .diff files show the changes
overlay/model.py
overlay/model_state.py
serve/serve-ssd.sh        wrapper around diffbot's serve.sh
serve/ab.sh               restart with a given config and benchmark it
serve/memlog.sh           log the container's memory every 10 s
bench/                    prefill benchmark + its corpus builder
plans/                    p375 host-expert plan + its generator
tests/                    bit-identity test, gather benchmark, page-cache residency (see each docstring)
results/raw-logs.txt      every number above
```

## Credits

**diffbot** (the pack, quantization, sm_120 vLLM overlay, kernels, plans, benchmarks), **nktlabs** (prestage design, parity mode,
fail-loud guards), **local-inference-lab** (the prefetch and resident-scales ideas, b12x #360 / vLLM #736), **Project Maya** (the
per-layer Engram wait), **sfxnz** (the Engram-on-disk lineage), **turboderp** (ExLlamaV3), **coolbho3k** (calibration method),
**vLLM**, **DeepSeek**. Details and licences: [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md), [`NOTICE`](NOTICE). Code here is MIT except the three modified vLLM files in
`overlay/`, which stay Apache-2.0.
