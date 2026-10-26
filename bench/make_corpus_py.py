"""Python sources from the host as fresh prefill text, in a seeded random file order so every slice prefill_ab.py takes
is a similar mix. Files of 2-64 KiB only (larger ones are mostly generated tables, whose rows are all hot), no vendored
copies, and each distinct file once (a second copy of a text is warm by construction)."""
import hashlib, os, random, sys
out_path, seen, files = sys.argv[1], set(), []
for root in ("/usr/lib/python3.12", "/usr/lib/python3/dist-packages"):
    for dp, _, fs in os.walk(root):
        if "_vendor" in dp.split(os.sep):
            continue
        for f in fs:
            p = os.path.join(dp, f)
            if f.endswith(".py") and 2048 < os.path.getsize(p) <= 65536:
                h = hashlib.sha1(open(p, "rb").read()).digest()
                if h not in seen:
                    seen.add(h); files.append(p)
files.sort(); random.Random(0).shuffle(files)
with open(out_path, "w") as out:
    for p in files:
        out.write(open(p, errors="replace").read())
print(len(files), "files,", os.path.getsize(out_path) // 2**20, "MiB")
