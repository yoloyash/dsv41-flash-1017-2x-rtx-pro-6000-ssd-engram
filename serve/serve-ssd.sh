#!/usr/bin/env bash
# Runs diffbot's recipe/serve/serve.sh unchanged, with overlay/ bind-mounted over the image's vLLM DeepSeek-V4.1 code so the
# int4 Engram tables stay on SSD. Every serve.sh knob still works (API_KEY, PORT, SPEC, MAX_NUM_SEQS, EAGER, ...). Plus:
#   PACK=<dir>                    required: diffbot's downloaded pack (build its image and kernels first, see README)
#   ENGRAM_SSD=1                  0 = diffbot's pinned-RAM path (needs ~230 GiB of RAM)
#   ENGRAM_SSD_PREFETCH=16384     tokens of the next prefill chunk to read ahead during the current one; 0 = off
#   ENGRAM_SSD_RESIDENT_SCALES=1  keep the Engram scales (11.4 GiB) in RAM: one SSD read per cold row instead of two
#   ENGRAM_SSD_EVICT=1            after load, drop the other shards from the page cache so it holds Engram rows
#   ENGRAM_SSD_DEFER=1            prefill: layer 14 rows read during layers 1-13, waited for only at layer 14; 0 = all before
#   ENGRAM_SSD_NATIVE=1           row gather on a native thread pool (overlay/engram_gather.cpp, built at load); 0 = Python pool
#   ENGRAM_SSD_THREADS=128        gather threads (a cold row is a page fault that blocks one); ENGRAM_SSD_MIN_CHUNK=8 rows/task
#   ENGRAM_SSD_CHECK=0            1 = parity mode: re-read sampled rows with pread and compare bit for bit every step
#   CONTAINER_MEM=120g            container cap (diffbot: 260g); page cache counts against it, so it caps the Engram cache too
#   SSD_PLAN=<file in plans/>     host-expert plan; default the p375 plan, SSD_PLAN= for diffbot's
#   HOST_CACHE=736                VRAM slots per GPU for copies of host experts (diffbot: 512). Sized for cards nothing else
#                                 uses: with a display or another process on them, HOST_CACHE=512 (~1.4 GiB/GPU less)
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${PACK:?set PACK to the downloaded diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 directory}"
export PACK
export CONTAINER_MEM=${CONTAINER_MEM:-120g}
export HOST_CACHE=${HOST_CACHE:-736}
# diffbot's recipe: the copy inside the pack (<pack>/recipe, as downloaded from Hugging Face) unless RECIPE points elsewhere
RECIPE=${RECIPE:-$PACK/recipe}
[[ -f $RECIPE/serve/serve.sh ]] || { echo "no serve/serve.sh in RECIPE=$RECIPE"; exit 1; }
ENGRAM_SSD=${ENGRAM_SSD:-1}
VP=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41
O=$ROOT/overlay
mounts=()
if [[ $ENGRAM_SSD == 1 ]]; then
  for f in common/engram.py common/engram_ssd.py common/engram_gather.cpp nvidia/model.py nvidia/model_state.py; do
    [[ -f $O/$(basename $f) ]] || { echo "missing $O/$(basename $f)"; exit 1; }
    mounts+=(-v "$O/$(basename $f):$VP/$f:ro")
  done
fi
SSD_PLAN=${SSD_PLAN-plan-1017-agent30-c512-p375.json}
if [[ -n $SSD_PLAN ]]; then
  [[ -f $ROOT/plans/$SSD_PLAN ]] || { echo "plans/$SSD_PLAN missing"; exit 1; }
  export EXL3_HOST_EXPERTS=/opt/ssd-plans/$SSD_PLAN
fi
mounts+=(-v "$ROOT/plans:/opt/ssd-plans:ro")
export DOCKER_EXTRA="${mounts[*]} -e DSV41_ENGRAM_SSD=$ENGRAM_SSD -e DSV41_ENGRAM_SSD_CHECK=${ENGRAM_SSD_CHECK:-0} \
-e DSV41_ENGRAM_SSD_THREADS=${ENGRAM_SSD_THREADS:-128} -e DSV41_ENGRAM_SSD_MIN_CHUNK=${ENGRAM_SSD_MIN_CHUNK:-8} -e DSV41_ENGRAM_SSD_INLINE_ROWS=${ENGRAM_SSD_INLINE_ROWS:-0} -e DSV41_ENGRAM_SSD_LOG_EVERY=${ENGRAM_SSD_LOG_EVERY:-500} \
-e DSV41_ENGRAM_SSD_EVICT=${ENGRAM_SSD_EVICT:-1} -e DSV41_ENGRAM_SSD_PREFETCH=${ENGRAM_SSD_PREFETCH:-16384} \
-e DSV41_ENGRAM_SSD_PREFETCH_THREADS=${ENGRAM_SSD_PREFETCH_THREADS:-${ENGRAM_SSD_THREADS:-128}} \
-e DSV41_ENGRAM_SSD_RESIDENT_SCALES=${ENGRAM_SSD_RESIDENT_SCALES:-1} -e DSV41_ENGRAM_SSD_DEFER=${ENGRAM_SSD_DEFER:-1} \
-e DSV41_ENGRAM_SSD_NATIVE=${ENGRAM_SSD_NATIVE:-1} ${DOCKER_EXTRA:-}"
echo "[serve-ssd] Engram on SSD: $([[ $ENGRAM_SSD == 1 ]] && echo "on (check ${ENGRAM_SSD_CHECK:-0})" || echo off), container memory $CONTAINER_MEM${SSD_PLAN:+, plan $SSD_PLAN (overrides the plan serve.sh prints below)}"
exec bash "$RECIPE/serve/serve.sh" "$@"
