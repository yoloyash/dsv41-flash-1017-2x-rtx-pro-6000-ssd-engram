# DeepSeek-V4.1-Flash-1017 on 2x RTX PRO 6000 with 128 GB of host RAM: Engram tables on SSD

A serving recipe, not a model. It runs [diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)
(EXL3 3-bit routed experts, int4 Engram, DSpark drafter, vision, 1M context) on a host with **128 GB of RAM** instead of the
~256 GB the pack's own recipe needs. No weights here: you download diffbot's pack, build diffbot's image and kernels with
diffbot's scripts, and run diffbot's `serve.sh`. This repo adds one thing on top: the two int4 Engram tables (~111 GB) stay in the
pack's safetensors shards on an NVMe SSD instead of pinned host RAM.

| | diffbot recipe (pinned RAM) | this repo (Engram on SSD) |
|---|---|---|
| Host RAM the container uses | ~230 GiB | **~82 GiB** anonymous (11.4 GiB of it the Engram scales) + whatever page cache the cap allows (`CONTAINER_MEM=120g`) |
| Output | — | bit-identical Engram rows (checked; see Fidelity) |
| Context, vision, tool calls, DSpark | 1M, yes, yes, yes | 1M, yes, yes, yes |

## Results

Same harness as the pack's model card for decode, the needle and quality: diffbot's `recipe/bench/bench-fixed.py` (fixed prompts
from the ExLlamaV3 corpus, temperature 0, thinking off, `ignore_eos`), `needle-test.py` and `quality-probe.py`, unchanged. Prefill
is `bench/prefill_ab.py` (below). diffbot's numbers are from their model card and `recipe/results/` on their hardware; ours are on
different hardware (below), so the gap is SSD + host together.

| decode tok/s, DSpark k=5 | diffbot, pinned RAM | this repo, warm page cache |
|---|---:|---:|
| 1 stream, 2K prompt | 156.9 | **150.1** |
| 1 stream, 46K prompt (median) | 172.8 | **143.5** |
| 4 streams, 46K prompts (aggregate) | 362.9 | **249.3** (62 per stream) |

| prefill tok/s, ~40K-token prompt, nothing in the KV cache | diffbot, pinned RAM | this repo |
|---|---:|---:|
| text the server has never seen (median of 8, right after a restart) | 11,157 – 11,467 | **6,925** (6,025 – 7,379) |
| same text again, prefix cache missed on purpose (its Engram rows now cached) | — | 9,394 |
| text never seen, before the prefetch / page-cache changes in this README | — | 4,132 – 4,293 |

| | diffbot | this repo |
|---|---:|---:|
| GSM8K-200 (thinking off) | 98.5% | **98.5%** (197/200) |
| HumanEval pass@1 (thinking off) | 97.0% | **97.6%** (160/164) |
| ~1.04M-token needle | found, 110.4 s | **found, 104.0 s** (1,032,034 tokens) |
| peak VRAM in that prefill | 96.6 / 97.0 GB (+2.5 GB other process) | 95.5 / 95.5 GB of 97.9 |

Quality, the needle and the DSpark-off figures were measured before the prefetch / page-cache changes; those changes leave every
staged row bit-identical (Fidelity), so only speed can differ. DSpark off (`SPEC=none`): 84.3 (2K) / 87.3 (46K) tok/s single
stream, 205.2 aggregate at 4 streams. Leave it on. `plans/plan-1017-agent30-c512-p375.json` was +10% at 4 streams over diffbot's
default plan and inside the noise at 1 stream, so it is the recommended plan. Raw output: `results/raw-logs.txt`.

**Read these numbers with their caveats:**
- *Warm* means the hot Engram rows are in the page cache. Right after a boot or restart the cache is cold: the first run after
  boot of an earlier version decoded at 80.8 (2K) / 114.9 (46K) / 218.8 (4x46K) tok/s.
- Prefill: `bench/prefill_ab.py` sends each prompt twice. **A** is a slice of a 28 MiB corpus of Python sources
  (`bench/make_corpus_py.py`) that no earlier run used, so some of its Engram rows are cold on the SSD. **B** is the same text
  behind a random prefix, which misses the KV cache again but finds the rows A just read. Every configuration got its own 8
  slices, a fresh restart, and was measured in the minutes after it, so these are cold-start numbers; the cache only fills
  from there. A server that has been running a while does better on A.
- What is left of the prefill gap: the first 16K-token chunk of a new prompt still reads its cold rows before it can start
  (~1.2 s per GPU on this SSD; later chunks are prefetched, below). The rest (B, 9.4K, vs diffbot's 11.2K) is there with every row
  cached, so it is not the SSD; we have not isolated it, the obvious candidate is the host experts crossing PCIe 4.0 instead of 5.0.
  The 1M needle is repetitive filler whose rows stay hot, which is why its prefill matches diffbot's.
- The 4-stream gap is larger than the 1-stream gap because the row read happens once per step, between steps (see below).

### Test host

AMD Threadripper PRO 5955WX (16 cores, Zen 3, **no AVX-512**, so diffbot's `CPU_EXPERTS=1` mode is unavailable; this is their
GPU-only path), 8x 16 GB DDR4-2933, 2x RTX PRO 6000 Blackwell Workstation (600 W) on **PCIe 4.0** x16, Engram shards on a WD_BLACK
SN7100 4 TB (DRAM-less; fio: ~227K random 4 KiB reads/s over the two shards, the same at queue depth 128 and 512), Linux 7.0.
diffbot's host: Threadripper PRO 9995WX, DDR5-4800, PCIe 5.0, Max-Q cards. The cold experts both setups keep in host RAM are read
over PCIe every step, so part of the gap is Gen4 vs Gen5, not the SSD.

## How it works

DeepSeek-V4.1's Engram layers (1 and 14) look up 48 hashed rows per token in two ~384M-row tables. diffbot stores them as int4
(128 B of nibbles + 8 fp16 scales per 256-dim row) in pinned host RAM and the GPU reads them over UVA inside the CUDA graphs.

Reading them from disk *inside* the forward pass forces eager mode (the `DSV41_ENGRAM_DISK` path: 26.1 tok/s measured by nktlabs,
31.5 reported by diffbot; vLLM PR #62238 is likewise eager / piecewise only).
[nktlabs](https://huggingface.co/nktlabs/dsv41-flash-engram-disk-prestage) showed the read does not have to be there: the
model runner's `prepare_inputs` already runs eagerly right before the graph replay. This repo ports that design to diffbot's int4
tables and current vLLM:

1. `prepare_inputs` runs the stock n-gram hash kernel eagerly and copies the hashes to the host (one sync).
2. Each rank gathers the rows of its own hash heads from an `mmap` of the shard (`MADV_RANDOM`) into pinned buffers, on 128
   threads (a cold row is a page fault that blocks its thread, so many small tasks keep the SSD queue deep).
3. One H2D copy, then **the stock int4 lookup kernel** dequantizes the compact rows into the layer's persistent `staged_rows`.
4. The captured forward reads `staged_rows` and never touches the table. It raises if the staged token count does not match the
   step, and the stock lookup raises if it is ever reached in this mode: no silent fallback onto stale rows.
5. **Prefetch** (`ENGRAM_SSD_PREFETCH=16384`): a long prompt is prefilled in 16K-token chunks, and all of its tokens are already on
   the GPU. Once step 2 has its own rows, the pre-step hashes the next chunk of every prefilling request with the same kernel and
   faults those rows into the page cache on a second thread pool while the forward runs, so the next chunk's gather finds them
   there. Measured per GPU on a new ~40K-token prompt with the defaults: chunk 1 gather 1.15 s, chunk 2 21 ms, chunk 3 19 ms.
   Prefetching two chunks ahead measured the same as one. Submitting the prefetch *before* the step's own gather finished
   measured no better than no prefetch: both compete for the same saturated SSD.

What it costs: steps 1–3 sit between two decode steps (~3 ms p50 per step warm, logged as `engram prestage ... row gather+stage`),
and the host sync gives up some of the runner's CPU/GPU overlap. That is most of the remaining decode gap; closing it needs the
read to overlap the forward itself (in-graph waits), which this repo does not attempt.

### Host memory: what the page cache holds

The tables' only RAM here is the page cache, and two things kept the Engram rows out of it:
- **The loader's leftovers.** Loading reads ~200 GB of other shards through the cache, and the last ~35 GB of them stay. Worse,
  `MADV_RANDOM` (needed: without it every 128-byte row fault reads 450 KiB of readahead on a default 512 KiB `read_ahead_kb`) also
  tells the kernel to ignore accessed bits on the mapping (`vma_has_recency()`, Linux 6.3+), so Engram pages are never promoted
  and lose every eviction to the loader's pages. Measured before the fix: 37.9 GiB of page cache in the container, 6.6 GiB of it
  Engram (`tests/page_cache.py`). `ENGRAM_SSD_EVICT=1` drops every checkpoint byte outside the Engram tensors from the cache once loading is
  done (`posix_fadvise(DONTNEED)`, which only drops clean, unmapped pages). After it, the container's cache is almost all Engram.
- **Two reads per cold row.** Each row is 128 B of nibbles plus 16 B of scales in a separate tensor, so a cold row costs two
  random reads. `ENGRAM_SSD_RESIDENT_SCALES=1` reads each rank's scales into host RAM at start (11.4 GiB total, ~3 s per layer)
  and drops them from the cache: one read per cold row. On its own it measured inside the noise; with prefetch it is what lets
  the next chunk's reads fit in the current chunk's forward (chunk 2 gather 185 ms → 21 ms). Decode with all three on: 150.1 /
  143.5 tok/s (2K / 46K) against 132.7 / 132.8 before; not split per change.

Prefill, ~40K-token new prompts, median tok/s over 8 slices per arm (`results/raw-logs.txt`; run-to-run spread about ±10%):

| config | new text (A) | Engram warm (B) |
|---|---:|---:|
| none of the three (start / end of the session) | 4,132 / 4,293 | 8,600 / 9,020 |
| evict | 5,129 / 4,523 | 9,635 / 9,437 |
| evict + resident scales | 5,380 | 9,352 |
| evict + prefetch 16K / 32K | 5,977 / 6,017 | 9,373 / 9,353 |
| **evict + prefetch 16K + resident scales (default)** | **6,925** | 9,394 |

Not here because they would not help on this host: an io_uring reader like local-inference-lab's b12x one (the SSD itself tops out
at ~227K random reads/s, and the 128 fault threads already reach that), and dropping `MADV_RANDOM` to get recency back (~115x more
bytes read per row).

Two other changes, both configuration:
- `CONTAINER_MEM=120g` (diffbot: 260g). Page cache counts against the container, so the cap is also the Engram cache size.
- `plans/plan-1017-agent30-c512-p375.json`: diffbot's default host-expert plan with 375 of its hottest host experts moved back
  to VRAM. diffbot sizes their plans for ~2.5 GB/GPU used by another process; with the cards to yourself that is free.
  `plans/make_promoted_plan.py` regenerates it from the plans in the pack.

Everything else is diffbot's recipe as shipped: image, patches, kernels, `serve.sh` and all its defaults.

### Fidelity

Same bytes, same dequant kernel, so the WKV input is bit-identical to the pinned-RAM path:
- `tests/test_engram_ssd.py`: staged rows vs. the stock kernel over the same rows, both TP ranks, 1–4096 tokens, with and without
  resident scales: bit-identical.
- Parity mode (`ENGRAM_SSD_CHECK=1 EAGER=1`): every step re-reads 64 sampled rows with plain `pread`, dequantizes on the CPU and
  compares bit for bit, and compares the prestaged hashes with the stock in-forward hashes. With prefetch on it also rebuilds each
  prefill chunk with the prefetch's own input builder and compares its hashes with the step's, so a prefetch that reads the wrong
  rows fails loudly instead of only being slow. Run with all three changes on: 699 steps covering a 44K prompt, 3 long prompts
  prefilling at once, 16 decode streams and `test-image.py`: zero mismatches. (The first version: 500 steps, also zero.)
- GSM8K / HumanEval match the pack's published figures (above). The model's fidelity to the original weights is diffbot's
  (KL ~0.03–0.05 nats, their `recipe/kl/`); nothing here changes it.

## Quick start

Needs: 2x RTX PRO 6000 (96 GB, sm_120), NVIDIA driver with CUDA ≥ 13, Docker with the NVIDIA runtime, ~128 GB RAM, ~330 GB on a
fast NVMe for the pack.

```bash
# 1. The pack (weights + diffbot's recipe/), on the NVMe
hf download diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 --local-dir /models/dsv41-1017
export PACK=/models/dsv41-1017

# 2. diffbot's image and kernels, with diffbot's scripts (~30-60 min, mostly the ExLlamaV3 build)
(cd $PACK/recipe && ./docker/build.sh && ./kernels/build.sh)

# 3. This repo
git clone https://github.com/yoloyash/dsv41-flash-1017-2x-rtx-pro-6000-ssd-engram && cd dsv41-flash-1017-2x-rtx-pro-6000-ssd-engram

# 4. Parity first (eager, every step checked against the shard bytes). Send it a few requests, check the log for errors.
API_KEY=change-me ENGRAM_SSD_CHECK=1 EAGER=1 WARMUP=0 serve/serve-ssd.sh

# 5. Serve
API_KEY=change-me SSD_PLAN=plan-1017-agent30-c512-p375.json serve/serve-ssd.sh
```

`serve/serve-ssd.sh` bind-mounts the four files in `overlay/` over the image's vLLM DeepSeek-V4.1 code and then `exec`s
`$PACK/recipe/serve/serve.sh`, so every knob there works as documented in the pack (`PORT`, `SPEC`, `SPEC_K`, `MAX_NUM_SEQS`,
`MAX_MODEL_LEN`, `VISION`, `THINKING`, ...). Its own knobs are at the top of the script: `ENGRAM_SSD` (0 = stock pinned path),
`ENGRAM_SSD_CHECK`, `ENGRAM_SSD_THREADS`, `ENGRAM_SSD_MIN_CHUNK`, `ENGRAM_SSD_EVICT`, `ENGRAM_SSD_PREFETCH`,
`ENGRAM_SSD_RESIDENT_SCALES`, `CONTAINER_MEM`, `SSD_PLAN`, `RECIPE`.

Tips:
- Drop caches before the first boot only if you want the cold numbers; otherwise the page cache is your friend.
- Lower `CONTAINER_MEM` if your host runs other things; Engram cache shrinks with it, decode speed with that. Short of RAM,
  `ENGRAM_SSD_RESIDENT_SCALES=0` gives back 11.4 GiB.
- `python3 tests/page_cache.py $PACK` (host, no GPU) shows how much of each Engram tensor is cached right now.
- `serve/memlog.sh` logs the container's anon / file / shmem every 10 s. `serve/ab.sh` restarts with a given env, runs
  `bench/prefill_ab.py` on its own fresh slices (`ARM=<n>`) and `bench-fixed.py` twice (warm-up + measured; only its decode
  figures are kept, its "fresh prefill" is a prefix-cache hit on the second pass). `tests/bench_gather.py` times the row gather
  alone at decode and prefill sizes.

## Files

```
overlay/engram_ssd.py        new: shard mmap, threaded gather, staging, stock-kernel dequant, prefetch, resident scales,
                             post-load page-cache eviction, parity check
overlay/engram.py            vLLM common/engram.py from diffbot's image + DSV41_ENGRAM_SSD hooks   (engram.diff)
overlay/model.py             vLLM nvidia/model.py + engram_prestage / prefetch / dummy / hash parity / timing (model.diff)
overlay/model_state.py       vLLM nvidia/model_state.py + the prepare_inputs hook, prefetch planning + its check (model_state.diff)
serve/serve-ssd.sh           wrapper around $PACK/recipe/serve/serve.sh
serve/ab.sh, serve/memlog.sh benchmarking helpers
bench/prefill_ab.py          fresh-prefill A/B (new text vs. same text with the KV cache missed)
bench/make_corpus_py.py      its corpus: the host's Python sources, deduplicated, in a seeded random order
plans/                       p375 host-expert plan + its generator
tests/                       bit-identity test, gather benchmark (inside the image; see each docstring), page_cache.py (host)
results/raw-logs.txt         every number in this README
```

The `.diff` files are against the files in diffbot's image (`docker/build.sh` at the pack's revision); regenerate them with
`docker cp` from your own image before trusting them on a different vLLM base.

## Credits

This is a small layer on other people's work: **diffbot** (the pack, its quantization, the sm_120 vLLM overlay, kernels, plans,
benchmarks, `serve.sh`), **nktlabs** (the prestage design, parity mode and fail-loud guards this ports),
**local-inference-lab** (bounded NVMe prefetch and resident Engram scales in b12x #360 / vLLM #736: the two ideas, not code, behind
`ENGRAM_SSD_PREFETCH` and `ENGRAM_SSD_RESIDENT_SCALES`), **sfxnz** (the DGX Spark
recipe and Engram-on-disk lineage diffbot's builds on), **turboderp** (ExLlamaV3 / EXL3), **coolbho3k** (the calibration method),
the **vLLM** project, and **DeepSeek** (the model). Details and licences: `ATTRIBUTIONS.md`, `NOTICE`.

Code in this repo is MIT (`LICENSE`) except the three modified vLLM files in `overlay/`, which stay Apache-2.0 (`NOTICE`).
