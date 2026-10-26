#!/usr/bin/env bash
# ab.sh <label> [ENV=VAL ...]: (re)start serve-ssd.sh with the given env and benchmark it.
#  decode:  diffbot's bench-fixed.py twice (the first pass warms the page cache, the second is the measurement). Only its
#           decode figures are kept: its "fresh prefill" is fresh only on the first pass, the second hits the prefix cache.
#  prefill: bench/prefill_ab.py on slices of PREFILL_CORPUS no earlier run used (ARM picks them; give every run its own ARM).
# Needs API_KEY and PACK; PORT defaults to 8002. bench-fixed.py needs its corpus: python3 $PACK/recipe/bench/make_corpus.py
# bench-work/corpus.txt (CORPUS overrides the path). PREFILL_CORPUS: large text the server has not seen, in an order that makes
# every slice a similar mix: python3 bench/make_corpus_py.py bench-work/corpus-py.txt (the host's Python sources, shuffled).
# Logs go to logs/.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
: "${API_KEY:?set API_KEY}" "${PACK:?set PACK}"
export PORT=${PORT:-8002} CORPUS=${CORPUS:-$ROOT/bench-work/corpus.txt}
PREFILL_CORPUS=${PREFILL_CORPUS:-$ROOT/bench-work/corpus-py.txt}
L=$1; shift; mkdir -p logs
docker rm -f "${NAME:-dsv41-flash-exl3}" >/dev/null 2>&1; sleep 3
env "$@" nohup serve/serve-ssd.sh > logs/serve-$L.log 2>&1 &
timeout 3600 bash -c "until grep -qE '\[warmup\] done|Traceback' logs/serve-$L.log; do sleep 10; done"
grep -q Traceback logs/serve-$L.log && { echo "$L: FAILED (logs/serve-$L.log)"; exit 1; }
if [[ -f $PREFILL_CORPUS ]]; then
  CORPUS=$PREFILL_CORPUS ARM=${ARM:-0} python3 bench/prefill_ab.py > logs/ab-$L-prefill.log 2>&1
fi
for pass in warm measured; do python3 "$PACK/recipe/bench/bench-fixed.py" > logs/ab-$L-$pass.log 2>&1; done
echo "$L [$*]: $(sed 's/ | fresh prefill.*//' logs/ab-$L-measured.log)"
[[ -f logs/ab-$L-prefill.log ]] && echo "  prefill: $(tail -1 logs/ab-$L-prefill.log)"
grep "engram prestage" logs/serve-$L.log | tail -2 | sed 's/.*steps, //'
