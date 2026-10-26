"""Int4 Engram tables read from the checkpoint shards on SSD (DSV41_ENGRAM_SSD=1).

The stock 2x recipe pins both int4 Engram tables (~111 GB) in host RAM and reads them over UVA
inside the CUDA graphs. Here they stay in the safetensors shards: each step the model runner
hashes the batch eagerly (model.py engram_prestage), this module gathers the rows the step needs
from an mmap of the shard into pinned host memory, copies them to the GPU once, and dequantizes
them into the layer's persistent `staged_rows` with the stock int4 lookup kernel. The graphed
forward then only reads `staged_rows`. The bytes and the dequant kernel are the stock ones, so
the WKV input is bit-identical to the pinned-RAM path.

Prefill chunks (DSV41_ENGRAM_SSD_DEFER=1) run eagerly, so they skip the wait before the forward:
layer 1's rows are read first and staged when the forward reaches layer 1, then layer 14's are
read while layers 1-13 compute and staged when it reaches layer 14 (Project Maya's ordering).

Port of nktlabs/dsv41-flash-engram-disk-prestage (MIT) to the int4 tables of
diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 and this vLLM nightly. Differences from
it: int4 rows (128 B nibbles + 16 B fp16 scales, separate tensors) instead of FP8, rows gathered
with a threaded index_select over an mmap instead of one preadv per row (a 16K-token prefill
chunk needs ~400K rows per rank), and dequant on the GPU with the stock kernel.

Rows are copied by engram_gather.cpp (DSV41_ENGRAM_SSD_NATIVE=1) on its own thread pool: on a Python
pool, a decode step's few hundred rows cost mostly interpreter overhead.

Page cache: MADV_RANDOM is what keeps a 128-byte row from costing a 512 KiB read-around, but
since Linux 6.3 (vma_has_recency) the kernel also ignores accessed bits on such mappings, so
Engram pages are never promoted and lose every eviction to the shard pages the loader read
once. evict_loaded_shards() drops those after load so the Engram rows get the cache.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_EMBED_SUFFIXES = (".engram.embed.weight", ".engram.embed.scale")


def engram_ssd_enabled() -> bool:
    return os.environ.get("DSV41_ENGRAM_SSD", "0") == "1"


def engram_ssd_check_enabled() -> bool:
    """Every step re-reads a sample of rows with plain pread, dequantizes them on the CPU
    and raises unless they equal the staged rows; in eager mode the in-forward hashes are
    also compared with the prestaged ones."""
    return os.environ.get("DSV41_ENGRAM_SSD_CHECK", "0") == "1"


def install_loader_skip() -> None:
    """Keep the weight iterator from reading the Engram tables (it would copy ~55 GB per rank)."""
    from vllm.model_executor.model_loader import weight_utils

    orig = weight_utils.should_skip_weight
    if getattr(orig, "_engram_ssd", False):
        return

    def should_skip_weight(name, *args, **kwargs):
        return name.endswith(_EMBED_SUFFIXES) or orig(name, *args, **kwargs)

    should_skip_weight._engram_ssd = True
    weight_utils.should_skip_weight = should_skip_weight


def _tensor_meta(path: str, name: str) -> tuple[int, tuple[int, ...], str]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n))[name]
    return 8 + n + meta["data_offsets"][0], tuple(meta["shape"]), meta["dtype"]


class _ShardView:
    """Read-only mmap of one tensor in a safetensors shard as a [rows, row_bytes] uint8 tensor."""

    def __init__(self, path: str, name: str, row_bytes: int) -> None:
        self.path = path
        offset, shape, self.dtype = _tensor_meta(path, name)
        self.rows = shape[0]
        nbytes = self.rows * row_bytes
        assert np.prod(shape) * {"U8": 1, "F16": 2}[self.dtype] == nbytes, (name, shape, row_bytes)
        page = offset - offset % mmap.ALLOCATIONGRANULARITY
        with open(path, "rb") as fh:
            self._mm = mmap.mmap(fh.fileno(), nbytes + offset - page, prot=mmap.PROT_READ, offset=page)
        # Rows are hashed: readahead only drags in neighbours nobody asked for.
        self._mm.madvise(mmap.MADV_RANDOM)
        arr = np.frombuffer(self._mm, dtype=np.uint8, count=nbytes, offset=offset - page)
        self.table = torch.from_numpy(arr.reshape(self.rows, row_bytes))
        self.offset, self.row_bytes = offset, row_bytes
        _KEEP.setdefault(os.path.realpath(path), []).append((offset, offset + nbytes))

    def read_range(self, lo: int, hi: int) -> torch.Tensor:
        """Rows [lo, hi) into anonymous host memory with plain reads, then drop them from the
        page cache: nothing reads them from the file again."""
        out = torch.empty(hi - lo, self.row_bytes, dtype=torch.uint8, device="cpu")
        buf, start = memoryview(out.numpy()).cast("B"), self.offset + lo * self.row_bytes
        fd = os.open(self.path, os.O_RDONLY)
        try:
            done = 0
            while done < len(buf):
                got = os.preadv(fd, [buf[done:done + (1 << 30)]], start + done)
                if got <= 0:
                    raise OSError(f"short read from {self.path}")
                done += got
            os.posix_fadvise(fd, start, len(buf), os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return out

    def pread_rows(self, rows: list[int]) -> torch.Tensor:
        """Independent reference read (check mode), bypassing the mmap."""
        out = torch.empty(len(rows), self.row_bytes, dtype=torch.uint8, device="cpu")
        with open(self.path, "rb", buffering=0) as fh:
            for i, r in enumerate(rows):
                fh.seek(self.offset + r * self.row_bytes)
                out[i] = torch.frombuffer(bytearray(fh.read(self.row_bytes)), dtype=torch.uint8)
        return out


# Byte ranges of the Engram tensors per shard: evict_loaded_shards() leaves them cached.
_KEEP: dict[str, list[tuple[int, int]]] = {}
_EVICTED = False


def evict_loaded_shards(model_dir: str) -> None:
    """Once per process, after load: drop every checkpoint byte outside the Engram tensors from
    the page cache. Loading leaves the last ~35 GB of shards cached and they outlive the Engram
    pages (see the module docstring). POSIX_FADV_DONTNEED only drops clean, unmapped pages, so
    it cannot affect correctness. DSV41_ENGRAM_SSD_EVICT=0 disables it."""
    global _EVICTED
    if _EVICTED or os.environ.get("DSV41_ENGRAM_SSD_EVICT", "1") != "1":
        return
    _EVICTED = True
    threading.Thread(target=_evict, args=(model_dir,), name="engram-evict", daemon=True).start()


def _evict(model_dir: str) -> None:
    t0, files = time.perf_counter(), 0
    for root, _, names in os.walk(model_dir):
        for name in names:
            if not name.endswith(".safetensors"):
                continue
            path = os.path.join(root, name)
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                lo, size = 0, os.fstat(fd).st_size
                for a, b in sorted(_KEEP.get(os.path.realpath(path), [])) + [(size, size)]:
                    if a > lo:
                        os.posix_fadvise(fd, lo, a - lo, os.POSIX_FADV_DONTNEED)
                    lo = max(lo, b)
                files += 1
            finally:
                os.close(fd)
    logger.info("Engram SSD: dropped the page cache of %d loaded checkpoint files outside the "
                "Engram tables in %.2f s", files, time.perf_counter() - t0)


_POOL: ThreadPoolExecutor | None = None
_PREFETCH_POOL: ThreadPoolExecutor | None = None
_NATIVE = None   # the engram_gather extension, False if disabled or unbuildable
_NATIVE_POOLS: dict[str, int] = {}


def _native():
    """DSV41_ENGRAM_SSD_NATIVE=1: engram_gather.cpp, built once into the torch extension cache."""
    global _NATIVE
    if _NATIVE is None:
        _NATIVE = False
        if os.environ.get("DSV41_ENGRAM_SSD_NATIVE", "1") == "1":
            from torch.utils.cpp_extension import load

            t0 = time.perf_counter()
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engram_gather.cpp")
            try:
                _NATIVE = load("dsv41_engram_gather", [src], extra_cflags=["-O3"])
                logger.info("Engram SSD: native row gather ready (%.1f s)", time.perf_counter() - t0)
            except Exception as e:  # no compiler, no source: the Python pool still works
                logger.warning("Engram SSD: native row gather unavailable, using the Python pool: %s", e)
    return _NATIVE


def _native_pool(name: str, threads: int) -> int:
    if name not in _NATIVE_POOLS:
        _NATIVE_POOLS[name] = _native().new_pool(threads)
    return _NATIVE_POOLS[name]


class _NativeJob:
    """A started native gather; result() waits for it like a future."""

    def __init__(self, pool: int, job: int) -> None:
        self.pool, self.job = pool, job

    def result(self) -> None:
        _native().wait(self.pool, self.job)


def _native_block(n: int, threads: int) -> int:
    """Rows per claim: a few for decode-sized steps, ~16 claims per thread for prefill chunks."""
    return max(4, -(-n // (16 * threads)))


def _pool(threads: int) -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(threads, thread_name_prefix="engram-ssd")
    return _POOL


def _prefetch_pool(threads: int) -> ThreadPoolExecutor:
    """Separate from _pool so a step's own reads never queue behind prefetches for later steps."""
    global _PREFETCH_POOL
    if _PREFETCH_POOL is None:
        _PREFETCH_POOL = ThreadPoolExecutor(threads, thread_name_prefix="engram-pf")
    return _PREFETCH_POOL


class SsdEngramTable:
    """One Engram layer's int4 table on SSD, for the heads this rank owns."""

    def __init__(self, emb, layer_id: int, max_tokens: int, device: torch.device, model_dir: str | None = None) -> None:
        if model_dir is None:
            from vllm.config import get_current_vllm_config

            model_dir = get_current_vllm_config().model_config.model
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as fh:
            weight_map = json.load(fh)["weight_map"]
        prefix = f"layers.{layer_id}.engram.embed"
        self.emb = emb
        self.layer_id = layer_id
        self.dim, self.block = emb.dim, emb.block_size
        self.w = _ShardView(os.path.join(model_dir, weight_map[f"{prefix}.weight"]), f"{prefix}.weight", self.dim // 2)
        self.s = _ShardView(os.path.join(model_dir, weight_map[f"{prefix}.scale"]), f"{prefix}.scale",
                            2 * (self.dim // self.block))
        assert self.w.dtype == "U8" and self.s.dtype == "F16", "DSV41_ENGRAM_SSD serves int4 Engram tables only"
        assert self.w.rows == self.s.rows >= emb.vocab_end_idx, (self.w.rows, emb.vocab_end_idx)
        self.local_heads = emb.part_n_hash_cols
        max_rows = max_tokens * self.local_heads
        pin = {"dtype": torch.uint8, "device": "cpu", "pin_memory": True}
        self.w_host = torch.empty(max_rows, self.w.row_bytes, **pin)
        self.s_host = torch.empty(max_rows, self.s.row_bytes, **pin)
        self.ids_host = torch.empty(max_rows, dtype=torch.int32, device="cpu", pin_memory=True)
        self.w_dev = torch.empty(max_rows, self.w.row_bytes, dtype=torch.uint8, device=device)
        self.s_dev = torch.empty(max_rows, self.s.row_bytes, dtype=torch.uint8, device=device)
        self.ids_dev = torch.empty(max_rows, dtype=torch.int32, device=device)
        self._copied = torch.cuda.Event()
        self._copied.record()
        self.threads = int(os.environ.get("DSV41_ENGRAM_SSD_THREADS", "64"))
        self.min_chunk = int(os.environ.get("DSV41_ENGRAM_SSD_MIN_CHUNK", "32"))
        self.inline_rows = int(os.environ.get("DSV41_ENGRAM_SSD_INLINE_ROWS", "0"))
        self.pf_threads = int(os.environ.get("DSV41_ENGRAM_SSD_PREFETCH_THREADS", str(self.threads)))
        self._pf_scratch: tuple[torch.Tensor, torch.Tensor] | None = None
        self.check = engram_ssd_check_enabled()
        self.model_dir = model_dir
        _native()   # build now, not on the first request
        # DSV41_ENGRAM_SSD_RESIDENT_SCALES=1: this rank's fp16 scales (1/9 of the table) live in
        # host RAM, so a cold row costs one SSD read instead of two.
        self.s_res: torch.Tensor | None = None
        if os.environ.get("DSV41_ENGRAM_SSD_RESIDENT_SCALES", "0") == "1":
            t0 = time.perf_counter()
            self.s_res = self.s.read_range(emb.vocab_start_idx, emb.vocab_end_idx)
            logger.info("Engram layer %d: %.2f GiB of scales resident in host RAM (%.1f s)",
                        layer_id, self.s_res.nbytes / 2**30, time.perf_counter() - t0)
        logger.info(
            "Engram layer %d on SSD: rows [%d, %d) of %s, %d local heads, staging %d rows (%.1f MiB pinned)",
            layer_id, emb.vocab_start_idx, emb.vocab_end_idx, os.path.basename(self.w.path),
            self.local_heads, max_rows, (self.w_host.nbytes + self.s_host.nbytes) / 2**20,
        )

    def _scale_ids(self, idx: torch.Tensor) -> torch.Tensor:
        """Table row ids -> rows of the scale source: the file (global ids) or s_res (local)."""
        if self.s_res is None:
            return idx
        return (idx - self.emb.vocab_start_idx).clamp_(min=0)

    @property
    def _scale_table(self) -> torch.Tensor:
        return self.s.table if self.s_res is None else self.s_res

    def _submit(self, idx: torch.Tensor, n: int) -> list:
        """Start w_host[:n] / s_host[:n] = the table rows idx (int64 CPU) on the shared pool.
        Cold rows are page faults served one at a time per thread, so cold reads want many
        chunks; warm ones want few, as each chunk costs a pool round trip."""
        sidx, stab = self._scale_ids(idx), self._scale_table
        if _native():
            pool = _native_pool("step", self.threads)
            job = _native().start(pool, self.w.table, idx, self.w_host, stab, sidx, self.s_host,
                                  _native_block(n, self.threads), False)
            return [_NativeJob(pool, job)]
        if n <= self.inline_rows:
            # Warm rows cost ~100 ns each; a pool round trip costs more than the gather.
            torch.index_select(self.w.table, 0, idx, out=self.w_host[:n])
            torch.index_select(stab, 0, sidx, out=self.s_host[:n])
            return []
        chunk = max(self.min_chunk, -(-n // self.threads))

        def work(lo: int) -> None:
            hi = min(lo + chunk, n)
            torch.index_select(self.w.table, 0, idx[lo:hi], out=self.w_host[lo:hi])
            torch.index_select(stab, 0, sidx[lo:hi], out=self.s_host[lo:hi])

        return [_pool(self.threads).submit(work, lo) for lo in range(0, n, chunk)]

    def prefetch(self, hashes: torch.Tensor) -> int:
        """Fault in, without waiting, the rows a later step will read (hashes: [T, n_hash_cols]
        int32 CPU, prompt tokens past this step), so that step's gather finds them in the page
        cache. The copies land in a scratch buffer nobody reads. Returns the rows submitted."""
        emb = self.emb
        local = hashes[:, emb.head_start: emb.head_start + self.local_heads].reshape(-1)
        idx = local[(local >= emb.vocab_start_idx) & (local < emb.vocab_end_idx)].to(torch.int64)
        n = idx.numel()
        if n == 0:
            return 0
        if self._pf_scratch is None or self._pf_scratch[0].shape[0] < n:
            self._pf_scratch = (torch.empty(n, self.w.row_bytes, dtype=torch.uint8, device="cpu"),
                                torch.empty(n, self.s.row_bytes, dtype=torch.uint8, device="cpu"))
        w_out, s_out = self._pf_scratch
        scales = self.s_res is None
        if _native():
            # Nobody waits on a prefetch: the job owns its tensors and is freed when done.
            pool = _native_pool("prefetch", self.pf_threads)
            block = _native_block(n, self.pf_threads)
            if scales:
                _native().start(pool, self.w.table, idx, w_out, self.s.table, idx, s_out, block, True)
            else:
                _native().start(pool, self.w.table, idx, w_out, None, None, None, block, True)
            return n
        chunk = max(self.min_chunk, -(-n // self.pf_threads))

        def work(lo: int) -> None:
            hi = min(lo + chunk, n)
            torch.index_select(self.w.table, 0, idx[lo:hi], out=w_out[lo:hi])
            if scales:
                torch.index_select(self.s.table, 0, idx[lo:hi], out=s_out[lo:hi])

        pool = _prefetch_pool(self.pf_threads)
        for lo in range(0, n, chunk):
            pool.submit(work, lo)
        return n

    def start(self, hashes: torch.Tensor, lazy: bool = False) -> list | None:
        """hashes: [T, n_hash_cols] int32 CPU (this layer). Starts the row gather; finish()
        completes it. Layers start before any finishes, so their reads overlap; lazy=True
        leaves the reads to begin() (or finish())."""
        emb = self.emb
        t = hashes.shape[0]
        n = t * self.local_heads
        if n == 0:
            return None
        local = hashes[:, emb.head_start: emb.head_start + self.local_heads].to(torch.int64)
        if (pad := self.local_heads - local.shape[1]) > 0:
            local = torch.nn.functional.pad(local, (0, pad), value=-1)
        rows = local.reshape(-1)
        owned = (rows >= emb.vocab_start_idx) & (rows < emb.vocab_end_idx)
        idx = torch.where(owned, rows, 0)   # file rows are global ids
        evict_loaded_shards(self.model_dir)
        # Host staging is reused every step: the previous step's copies must be done reading it.
        self._copied.synchronize()
        return [t, n, rows, owned, idx, None if lazy else self._submit(idx, n)]

    def begin(self, state: list | None) -> None:
        """Submit a lazy start()'s reads (no-op once submitted)."""
        if state is not None and state[5] is None:
            state[5] = self._submit(state[4], state[1])

    def finish(self, state: tuple | None, staged_rows: torch.Tensor) -> None:
        """Fills staged_rows[:T] on the current stream with exactly what the stock lookup writes."""
        from vllm.models.deepseek_v41.common.engram import _engram_lookup_int4_kernel
        from vllm.triton_utils import triton

        if state is None:
            return
        self.begin(state)
        t, n, rows, owned, _, futs = state
        for fut in futs:
            fut.result()
        # Compact ids: row i of the staged table is token i // LH, head i % LH; -1 = not owned,
        # which the kernel masks to the same zero row as the stock path.
        ids = self.ids_host[:n]
        torch.where(owned, torch.arange(n, dtype=torch.int32, device="cpu"), torch.full((), -1, dtype=torch.int32, device="cpu"), out=ids)
        self.w_dev[:n].copy_(self.w_host[:n], non_blocking=True)
        self.s_dev[:n].copy_(self.s_host[:n], non_blocking=True)
        self.ids_dev[:n].copy_(ids, non_blocking=True)
        self._copied.record()
        out = staged_rows[:t]
        assert out.is_contiguous() and out.shape[1:] == (self.local_heads, self.dim)
        grid = min(triton.cdiv(n, 16), self.emb._num_sms)
        _engram_lookup_int4_kernel[(grid,)](
            self.w_dev, self.s_dev.view(torch.float16), self.ids_dev[:n].view(t, self.local_heads), out,
            0, n, n, self.local_heads, 1,
            HEAD_START=0, LOCAL_HEADS=self.local_heads, TOTAL_HEADS=self.local_heads, DIM=self.dim,
            QUANT_BLOCK=self.block, BLOCK_R=16, GRID=grid,
        )
        if self.check:
            self._check(rows, owned, out)

    def prestage(self, hashes: torch.Tensor, staged_rows: torch.Tensor) -> None:
        self.finish(self.start(hashes), staged_rows)

    def _check(self, rows: torch.Tensor, owned: torch.Tensor, out: torch.Tensor, sample: int = 64) -> None:
        """Re-read a sample of rows with pread, dequantize on the CPU, compare bit for bit."""
        n = rows.numel()
        pick = torch.randperm(n, device="cpu")[:sample]
        got = out.reshape(n, self.dim)[pick.to(out.device)].cpu()
        file_rows = [int(r) for r in rows[pick]]
        safe = [r if o else 0 for r, o in zip(file_rows, owned[pick].tolist())]
        packed = self.w.pread_rows(safe).to(torch.int32)
        nib = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(len(safe), self.dim)
        scale = self.s.pread_rows(safe).view(torch.float16).to(torch.float32)
        ref = ((nib - 8).to(torch.float32) * scale.repeat_interleave(self.block, dim=1)).to(torch.bfloat16)
        ref[~owned[pick]] = 0
        if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
            bad = int((got.view(torch.int16) != ref.view(torch.int16)).any(dim=1).sum())
            raise RuntimeError(f"engram SSD check: {bad} of {len(safe)} sampled rows differ from the shard bytes")
