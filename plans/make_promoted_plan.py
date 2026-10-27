#!/usr/bin/env python3
"""make_promoted_plan.py <pack>/recipe/plans <N> <out.json>: diffbot's default host-expert plan (plan-1017-agent30-c512.json) with N
of its hottest host experts moved back to VRAM, for cards with no other process on them.

diffbot's plans are sized for ~2.5 GB/GPU used by another process (plans/README.md there). The plan files carry expert ids, not
routing counts, but the experts the c512 plan holds in host RAM beyond plan-1017-agent30.json are by construction the least cold
of its host experts (the next ones it demoted to pay for the 512-slot cache). This promotes N of that band, spread evenly across
it. plan-1017-agent30-c512-p375.json in this folder is N=375 (4,921 host experts instead of 5,296): ~2.4 GB/GPU more VRAM; a
1.03M-token prefill peaked at 95.5 of 97.9 GB per card with HOST_CACHE=512, 97.0 with serve-ssd.sh's default 736 slots. The
whole band (N=557) with 640 slots measured the same as p375 with 736 and used ~0.6 GB/GPU more."""
import json, os, sys
d, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
a = json.load(open(os.path.join(d, "plan-1017-agent30.json")))
c = json.load(open(os.path.join(d, "plan-1017-agent30-c512.json")))
band = [(k, e) for k in c for e in c[k] if e not in set(a.get(k, []))]
assert 0 < n <= len(band), (n, len(band))
pick = {band[int(i * len(band) / n)] for i in range(n)}
p = {k: [e for e in v if (k, e) not in pick] for k, v in c.items()}
json.dump(p, open(out, "w"))
print(f"{out}: {sum(map(len, p.values()))} host experts (c512 has {sum(map(len, c.values()))}, band {len(band)}, promoted {n})")
