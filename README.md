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
| 1 stream, 2K prompt | 156.9 | **150.1** |
| 1 stream, 46K prompt | 172.8 | **143.5** |
| 4 streams, 46K prompts, aggregate | 362.9 | **249.3** |

**Prefill**, tok/s, ~40K-token prompt, nothing in the KV cache

| | diffbot | this repo |
|---|---:|---:|
| text the server has never seen | 11,157 – 11,467 | **6,925** |
| text whose Engram rows are already cached | — | 9,394 |

**Quality and long context**

| | diffbot | this repo |
|---|---:|---:|
| GSM8K-200, thinking off | 98.5% | **98.5%** |
| HumanEval pass@1, thinking off | 97.0% | **97.6%** |
| ~1.04M-token needle | found, 110.4 s | **found, 104.0 s** |
| peak VRAM during it | 96.6 / 97.0 GB | 95.5 / 95.5 GB |

Notes:
- Prefill is [`bench/prefill_ab.py`](bench/prefill_ab.py): median of 8 never-seen slices of Python source, measured in the first
  minutes after a restart (cold cache). A server that has been up for a while does better.
- Right after a boot, decode is slower until the page cache warms up (80.8 / 114.9 / 218.8 on an earlier version).
- DSpark off: 84.3 / 87.3 / 205.2 tok/s. Leave it on.
- Quality, the needle and DSpark-off were measured before prefetch / resident scales / eviction. Those don't change any row the
  model sees, only speed.

### What each change is worth

Prefill on new text, median tok/s, 8 fresh prompts and a restart per row (run-to-run spread about ±10%):

| | tok/s |
|---|---:|
| prestage only (start / end of the session) | 4,132 / 4,293 |
| + evict loader pages from the page cache | 5,129 / 4,523 |
| + evict + prefetch | 5,977 |
| + evict + resident scales | 5,380 |
| **+ evict + prefetch + resident scales (default)** | **6,925** |

Decode went from 132.7 / 132.8 / 249.3 to 150.1 / 143.5 / 249.3 with all three on.

### Test host

Threadripper PRO 5955WX (Zen 3, no AVX-512, so diffbot's GPU-only path), 8x 16 GB DDR4-2933, 2x RTX PRO 6000 Workstation (600 W)
on PCIe 4.0 x16, Engram shards on a WD_BLACK SN7100 4 TB (~227K random 4 KiB reads/s measured with fio), Linux 7.0.

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
5. **p375 plan**: diffbot's default plan with its 375 hottest host experts moved back to VRAM (+10% at 4 streams).

Where the remaining gap is:
- the first 16K chunk of a new prompt still waits ~1.2 s on cold reads (nothing runs before it to overlap with)
- decode pays ~3 ms per step for the gather between steps
- with every row cached, prefill is 9.4K vs 11.2K; not isolated, most likely PCIe 4.0 vs 5.0 for the host experts

Tried and dropped: an io_uring reader (the SSD tops out at ~227K reads/s, which the threads already reach), and prefetching
two chunks ahead (same as one).

## Fidelity

- [`tests/test_engram_ssd.py`](tests/test_engram_ssd.py): staged rows bit-identical to diffbot's kernel, both layers, both GPUs,
  with and without resident scales.
- Parity mode: each step re-reads sampled rows with `pread` and compares bit for bit, checks the hashes against the stock path, and
  checks that prefetch hashes the right tokens. 699 steps (long prompts, concurrent prefills, 16 decode streams, images): zero
  mismatches.
- GSM8K / HumanEval match diffbot's figures. Fidelity to the original weights is diffbot's (their `recipe/kl/`).

## Files

```
overlay/engram_ssd.py     new: shard reader, gather, prefetch, resident scales, eviction, parity check
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
fail-loud guards), **local-inference-lab** (the prefetch and resident-scales ideas, b12x #360 / vLLM #736), **sfxnz** (the
Engram-on-disk lineage), **turboderp** (ExLlamaV3), **coolbho3k** (calibration method), **vLLM**, **DeepSeek**. Details and
licences: [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md), [`NOTICE`](NOTICE). Code here is MIT except the three modified vLLM files in
`overlay/`, which stay Apache-2.0.
