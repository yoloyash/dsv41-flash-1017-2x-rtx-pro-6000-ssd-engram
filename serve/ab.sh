#!/usr/bin/env bash
# ab.sh <label> [ENV=VAL ...]: (re)start serve-ssd.sh with the given env, run diffbot's bench-fixed.py twice (the first pass warms
# the page cache, the second is the measurement) and print the result plus the prestage timing line. Logs go to logs/.
# Needs API_KEY and PACK; PORT defaults to 8002. bench-fixed.py needs its corpus: python3 $PACK/recipe/bench/make_corpus.py
# bench-work/corpus.txt (CORPUS overrides the path).
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
: "${API_KEY:?set API_KEY}" "${PACK:?set PACK}"
export PORT=${PORT:-8002} CORPUS=${CORPUS:-$ROOT/bench-work/corpus.txt}
L=$1; shift; mkdir -p logs
docker rm -f "${NAME:-dsv41-flash-exl3}" >/dev/null 2>&1; sleep 3
env "$@" nohup serve/serve-ssd.sh > logs/serve-$L.log 2>&1 &
timeout 3600 bash -c "until grep -qE '\[warmup\] done|Traceback' logs/serve-$L.log; do sleep 10; done"
grep -q Traceback logs/serve-$L.log && { echo "$L: FAILED (logs/serve-$L.log)"; exit 1; }
for pass in warm measured; do python3 "$PACK/recipe/bench/bench-fixed.py" > logs/ab-$L-$pass.log 2>&1; done
echo "$L [$*]: $(cat logs/ab-$L-measured.log)"
grep "engram prestage" logs/serve-$L.log | tail -2 | sed 's/.*steps, //'
