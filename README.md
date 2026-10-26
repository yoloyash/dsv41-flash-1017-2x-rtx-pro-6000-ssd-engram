# DeepSeek-V4.1-Flash-1017 on 2x RTX PRO 6000 with 128 GB of host RAM: Engram tables on SSD

A serving recipe, not a model. It runs [diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)
(EXL3 3-bit routed experts, int4 Engram, DSpark drafter, vision, 1M context) on a host with **128 GB of RAM** instead of the
~256 GB the pack's own recipe needs. No weights here: you download diffbot's pack, build diffbot's image and kernels with
diffbot's scripts, and run diffbot's `serve.sh`. This repo adds one thing on top: the two int4 Engram tables (~111 GB) stay in the
pack's safetensors shards on an NVMe SSD instead of pinned host RAM.

| | diffbot recipe (pinned RAM) | this repo (Engram on SSD) |
|---|---|---|
| Host RAM the container uses | ~230 GiB | **~71 GiB** anonymous + whatever page cache the cap allows (`CONTAINER_MEM=120g`) |
| Output | — | bit-identical Engram rows (checked; see Fidelity) |
| Context, vision, tool calls, DSpark | 1M, yes, yes, yes | 1M, yes, yes, yes |

## Results

Same harness as the pack's model card: diffbot's `recipe/bench/bench-fixed.py` (fixed prompts from the ExLlamaV3 corpus,
temperature 0, thinking off, `ignore_eos`), `needle-test.py` and `quality-probe.py`, unchanged. diffbot's numbers are from their
model card and `recipe/results/` on their hardware; ours are on different hardware (below), so the gap is SSD + host together.

| decode tok/s, DSpark k=5 | diffbot, pinned RAM | this repo, warm page cache | this repo, first run after boot |
|---|---:|---:|---:|
| 1 stream, 2K prompt | 156.9 | **132.6** | 80.8 |
| 1 stream, 46K prompt (median) | 172.8 | **132.8 – 139.7** | 114.9 |
| 4 streams, 46K prompts (aggregate) | 362.9 | **249.3** (62 per stream) | 218.8 |
| fresh prefill, 40K prompt | 11,157 – 11,467 tok/s | — | 3,872 – 5,908 tok/s |

| | diffbot | this repo |
|---|---:|---:|
| GSM8K-200 (thinking off) | 98.5% | **98.5%** (197/200) |
| HumanEval pass@1 (thinking off) | 97.0% | **97.6%** (160/164) |
| ~1.04M-token needle | found, 110.4 s | **found, 104.0 s** (1,032,034 tokens) |
| peak VRAM in that prefill | 96.6 / 97.0 GB (+2.5 GB other process) | 95.5 / 95.5 GB of 97.9 |

DSpark off (`SPEC=none`), same setup: 84.3 (2K) / 87.3 (46K) tok/s single stream, 205.2 aggregate at 4 streams. Leave it on.

The two 1-stream 46K figures are the two configurations measured (`plans/plan-1017-agent30-c512-p375.json` and diffbot's default
plan); their difference is inside the ~5% run-to-run spread diffbot reports for that test, and p375 is +10% at 4 streams, so it is
the recommended one. Raw output: `results/raw-logs.txt`.

**Read these numbers with their caveats:**
- *Warm* means the hot Engram rows are in the page cache. The first minutes after a boot are slower until it fills; the last
  column is the first run after boot, measured with an earlier, untuned setting (64 threads, 112 GB cap, diffbot's plan).
- Fresh prefill of new text is ~2–3x slower than pinned RAM: a 16K-token chunk needs ~400K random 144-byte rows per GPU, and the
  cold ones are SSD reads. Cached prefixes (agentic turns) are unaffected. The needle prompt is repetitive filler, so its rows
  stay hot; that is why its prefill time is close to diffbot's and fresh code prefill is not.
- The 4-stream gap is larger than the 1-stream gap because the row read happens once per step, between steps (see below).

### Test host

AMD Threadripper PRO 5955WX (16 cores, Zen 3, **no AVX-512**, so diffbot's `CPU_EXPERTS=1` mode is unavailable; this is their
GPU-only path), 8x 16 GB DDR4-2933, 2x RTX PRO 6000 Blackwell Workstation (600 W) on **PCIe 4.0** x16, Engram shards on a WD_BLACK
SN7100 4 TB. diffbot's host: Threadripper PRO 9995WX, DDR5-4800, PCIe 5.0, Max-Q cards. The cold experts both setups keep in host
RAM are read over PCIe every step, so part of the gap is Gen4 vs Gen5, not the SSD.

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

What it costs: steps 1–3 sit between two decode steps (~3 ms p50 per step warm, logged as `engram prestage ... row gather+stage`),
and the host sync gives up some of the runner's CPU/GPU overlap. That is most of the remaining gap; closing it needs the read to
overlap the forward itself (in-graph waits), which this repo does not attempt.

Two other changes, both configuration:
- `CONTAINER_MEM=120g` (diffbot: 260g). Page cache counts against the container, so the cap is also the Engram cache size.
- `plans/plan-1017-agent30-c512-p375.json`: diffbot's default host-expert plan with 375 of its hottest host experts moved back
  to VRAM. diffbot sizes their plans for ~2.5 GB/GPU used by another process; with the cards to yourself that is free.
  `plans/make_promoted_plan.py` regenerates it from the plans in the pack.

Everything else is diffbot's recipe as shipped: image, patches, kernels, `serve.sh` and all its defaults.

### Fidelity

Same bytes, same dequant kernel, so the WKV input is bit-identical to the pinned-RAM path:
- `tests/test_engram_ssd.py`: staged rows vs. the stock kernel over the same rows, both TP ranks, 1–4096 tokens: bit-identical.
- Parity mode (`ENGRAM_SSD_CHECK=1 EAGER=1`): every step re-reads 64 sampled rows with plain `pread`, dequantizes on the CPU and
  compares bit for bit, and compares the prestaged hashes with the stock in-forward hashes. 500 steps of short prompts, a 3.7K
  prefill, 8 concurrent streams and an image request: zero mismatches.
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
`ENGRAM_SSD_CHECK`, `ENGRAM_SSD_THREADS`, `ENGRAM_SSD_MIN_CHUNK`, `CONTAINER_MEM`, `SSD_PLAN`, `RECIPE`.

Tips:
- Drop caches before the first boot only if you want the cold numbers; otherwise the page cache is your friend.
- Lower `CONTAINER_MEM` if your host runs other things; Engram cache shrinks with it, decode speed with that.
- `serve/memlog.sh` logs the container's anon / file / shmem every 10 s; `serve/ab.sh` restarts with a given env and runs
  `bench-fixed.py` twice (warm-up + measured). `tests/bench_gather.py` times the row gather alone at decode and prefill sizes.

## Files

```
overlay/engram_ssd.py        new: shard mmap, threaded gather, staging, stock-kernel dequant, parity check
overlay/engram.py            vLLM common/engram.py from diffbot's image + DSV41_ENGRAM_SSD hooks   (engram.diff)
overlay/model.py             vLLM nvidia/model.py + engram_prestage / dummy / hash parity / timing (model.diff)
overlay/model_state.py       vLLM nvidia/model_state.py + the prepare_inputs hook                  (model_state.diff)
serve/serve-ssd.sh           wrapper around $PACK/recipe/serve/serve.sh
serve/ab.sh, serve/memlog.sh benchmarking helpers
plans/                       p375 host-expert plan + its generator
tests/                       bit-identity test, gather benchmark (run inside the image; see each file's docstring)
results/raw-logs.txt         every number in this README
```

The `.diff` files are against the files in diffbot's image (`docker/build.sh` at the pack's revision); regenerate them with
`docker cp` from your own image before trusting them on a different vLLM base.

## Credits

This is a small layer on other people's work: **diffbot** (the pack, its quantization, the sm_120 vLLM overlay, kernels, plans,
benchmarks, `serve.sh`), **nktlabs** (the prestage design, parity mode and fail-loud guards this ports), **sfxnz** (the DGX Spark
recipe and Engram-on-disk lineage diffbot's builds on), **turboderp** (ExLlamaV3 / EXL3), **coolbho3k** (the calibration method),
the **vLLM** project, and **DeepSeek** (the model). Details and licences: `ATTRIBUTIONS.md`, `NOTICE`.

Code in this repo is MIT (`LICENSE`) except the three modified vLLM files in `overlay/`, which stay Apache-2.0 (`NOTICE`).
