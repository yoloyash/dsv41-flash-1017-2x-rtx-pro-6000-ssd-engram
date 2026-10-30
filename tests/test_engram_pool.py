"""CPU-only regression for detached Torch tensor cleanup vs pool submission.

Run from the repo root, with PyTorch and a C++ compiler installed:
    MAX_JOBS=1 timeout 90s python3 tests/test_engram_pool.py overlay/engram_gather.cpp

Run in a separate process: the pre-fix extension can deadlock. Compilation is
included in the timeout, so allow more time on a slower build host. No GPU or
checkpoint is needed. Awaited gathers also check exact reference bytes.
"""
import argparse
import gc
from pathlib import Path
import time

import torch
from torch.utils.cpp_extension import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--jobs', type=int, default=10000)
    args = parser.parse_args()
    mod = load('dsv41_engram_pool_regression', [str(args.source.resolve())],
               extra_cflags=['-O3'])
    torch.set_num_threads(1)
    pool = mod.new_pool(8)
    table = torch.arange(4096 * 144, dtype=torch.int64).to(torch.uint8).view(4096, 144)
    begin = time.monotonic()
    for i in range(args.jobs):
        # Dropping Python owners while detached native jobs finish exercises the
        # tensor-decref/GIL edge responsible for the production lock inversion.
        ids = torch.randint(0, 4096, (64 + (i % 7) * 16,), dtype=torch.int64)
        out = torch.empty(ids.numel(), 144, dtype=torch.uint8)
        mod.start(pool, table, ids, out, None, None, None, 4, True)
        del ids, out
        if i % 50 == 0:
            idx = torch.tensor([0, 17, 4095], dtype=torch.int64)
            got = torch.empty(3, 144, dtype=torch.uint8)
            job = mod.start(pool, table, idx, got, None, None, None, 1, False)
            mod.wait(pool, job)
            assert torch.equal(got, table[idx])
            gc.collect()
        if i % 1000 == 0:
            print(f'jobs={i}', flush=True)
    print(f'PASS {args.jobs} detached jobs plus exact-byte awaited gathers '
          f'in {time.monotonic() - begin:.3f}s', flush=True)


if __name__ == '__main__':
    main()
