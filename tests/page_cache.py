#!/usr/bin/env python3
"""How much of each Engram tensor is in the page cache right now (mincore over a fresh read-only mapping; reads nothing).
Run on the host, no container or GPU needed:  python3 tests/page_cache.py $PACK
The cache is the Engram tables' only RAM in this recipe: if this stays low while the container's memory.stat shows a large
`file` figure, the cache is holding something else (see ENGRAM_SSD_EVICT in serve/serve-ssd.sh)."""
import ctypes, json, os, struct, sys
pack = sys.argv[1]
wm = json.load(open(os.path.join(pack, "model.safetensors.index.json")))["weight_map"]
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
PG, total, cached = 4096, 0, 0
for layer in (1, 14):
    for kind in ("weight", "scale"):
        name = f"layers.{layer}.engram.embed.{kind}"; path = os.path.join(pack, wm[name])
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; meta = json.loads(fh.read(n))[name]
            a, b = 8 + n + meta["data_offsets"][0], 8 + n + meta["data_offsets"][1]
            start = a - a % PG; length = b - start
            p = libc.mmap(None, length, 1, 1, fh.fileno(), start)   # PROT_READ, MAP_SHARED
            vec = (ctypes.c_ubyte * ((length + PG - 1) // PG))()
            if libc.mincore(p, length, vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore")
            libc.munmap(p, length)
        res = sum(v & 1 for v in bytes(vec)) * PG
        total += length; cached += res
        print(f"layer {layer:2d} {kind:6s}: {res / 2**30:5.1f} of {length / 2**30:5.1f} GiB cached ({100 * res / length:4.1f}%)")
print(f"Engram total: {cached / 2**30:.1f} of {total / 2**30:.1f} GiB cached")
