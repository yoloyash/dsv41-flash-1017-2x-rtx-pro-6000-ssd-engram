#!/usr/bin/env bash
# diffbot's recipe/serve/serve.sh, run unchanged, plus the Engram-on-SSD overlay: the four files in overlay/ are bind-mounted
# over the image's vLLM DeepSeek-V4.1 model code and DSV41_ENGRAM_SSD=1 keeps the int4 Engram tables (~111 GB) in the pack's
# shards instead of pinned host RAM. Every serve.sh knob still works (API_KEY, PORT, SPEC, SPEC_K, MAX_NUM_SEQS, EAGER, ...).
#   ENGRAM_SSD=1          0 = stock pinned-RAM path (needs ~230 GiB of host RAM)
#   ENGRAM_SSD_CHECK=0    1 = every step re-reads a sample of staged rows with pread and compares bit for bit; with EAGER=1
#                         also compares the prestaged hashes with the stock in-forward hashes. Run this once first.
#   ENGRAM_SSD_THREADS=128 ENGRAM_SSD_MIN_CHUNK=8   gather threads / smallest rows per task: a cold row is a page fault that
#                         blocks its thread for ~100 us, so many small tasks keep the SSD queue deep
#   ENGRAM_SSD_EVICT=1    after load, drop the page cache of the checkpoint outside the Engram tables, which otherwise
#                         outlives the Engram rows in the cache (overlay/engram_ssd.py docstring)
#   ENGRAM_SSD_PREFETCH=16384  tokens: hash the prompt tokens later steps will prefill and fault their rows in during this
#                         step's forward (one chunk ahead; 0 = off); ENGRAM_SSD_PREFETCH_THREADS (default ENGRAM_SSD_THREADS)
#   ENGRAM_SSD_RESIDENT_SCALES=1  keep the fp16 scales (11.4 GiB total) in host RAM: one SSD read per cold row, not two
#   CONTAINER_MEM=120g    (stock: 260g). Page cache is charged to the container, so this also caps how much of the Engram
#                         tables stays cached; on a 128 GB host 120g leaves ~5 GB for the OS.
#   PACK=<dir>            required: the downloaded pack (its recipe/ is what runs; build its image and kernels first, README)
#   SSD_PLAN=<file>       host-expert plan from this repo's plans/ instead of the recipe's PLAN (keep HOST_CACHE=512 with the
#                         c512-pN plans: they are the stock c512 plan with N of its hottest host experts moved back to VRAM)
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${PACK:?set PACK to the downloaded diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 directory}"
export PACK
export CONTAINER_MEM=${CONTAINER_MEM:-120g}
# diffbot's recipe: the copy inside the pack (<pack>/recipe, as downloaded from Hugging Face) unless RECIPE points elsewhere
RECIPE=${RECIPE:-$PACK/recipe}
[[ -f $RECIPE/serve/serve.sh ]] || { echo "no serve/serve.sh in RECIPE=$RECIPE"; exit 1; }
ENGRAM_SSD=${ENGRAM_SSD:-1}
VP=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41
O=$ROOT/overlay
mounts=()
if [[ $ENGRAM_SSD == 1 ]]; then
  for f in common/engram.py common/engram_ssd.py nvidia/model.py nvidia/model_state.py; do
    [[ -f $O/$(basename $f) ]] || { echo "missing $O/$(basename $f)"; exit 1; }
    mounts+=(-v "$O/$(basename $f):$VP/$f:ro")
  done
fi
if [[ -n ${SSD_PLAN:-} ]]; then
  [[ -f $ROOT/plans/$SSD_PLAN ]] || { echo "plans/$SSD_PLAN missing"; exit 1; }
  export EXL3_HOST_EXPERTS=/opt/ssd-plans/$SSD_PLAN
fi
mounts+=(-v "$ROOT/plans:/opt/ssd-plans:ro")
export DOCKER_EXTRA="${mounts[*]} -e DSV41_ENGRAM_SSD=$ENGRAM_SSD -e DSV41_ENGRAM_SSD_CHECK=${ENGRAM_SSD_CHECK:-0} \
-e DSV41_ENGRAM_SSD_THREADS=${ENGRAM_SSD_THREADS:-128} -e DSV41_ENGRAM_SSD_MIN_CHUNK=${ENGRAM_SSD_MIN_CHUNK:-8} -e DSV41_ENGRAM_SSD_INLINE_ROWS=${ENGRAM_SSD_INLINE_ROWS:-0} -e DSV41_ENGRAM_SSD_LOG_EVERY=${ENGRAM_SSD_LOG_EVERY:-500} \
-e DSV41_ENGRAM_SSD_EVICT=${ENGRAM_SSD_EVICT:-1} -e DSV41_ENGRAM_SSD_PREFETCH=${ENGRAM_SSD_PREFETCH:-16384} \
-e DSV41_ENGRAM_SSD_PREFETCH_THREADS=${ENGRAM_SSD_PREFETCH_THREADS:-${ENGRAM_SSD_THREADS:-128}} \
-e DSV41_ENGRAM_SSD_RESIDENT_SCALES=${ENGRAM_SSD_RESIDENT_SCALES:-1} ${DOCKER_EXTRA:-}"
echo "[serve-ssd] Engram on SSD: $([[ $ENGRAM_SSD == 1 ]] && echo "on (check ${ENGRAM_SSD_CHECK:-0})" || echo off), container memory $CONTAINER_MEM${SSD_PLAN:+, plan $SSD_PLAN (overrides the plan serve.sh prints below)}"
exec bash "$RECIPE/serve/serve.sh" "$@"
