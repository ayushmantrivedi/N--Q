import os
import time

base = r"F:\NADI-Q data"
start_time = time.time()
print(f"Counting all files under {base}...")

ext_counts = {}
top_counts = {}
total_size = 0
total_files = 0

for root, dirs, files in os.walk(base):
    rel = os.path.relpath(root, base)
    top = rel.split(os.sep)[0] if rel != "." else "ROOT"
    top_counts[top] = top_counts.get(top, 0) + len(files)
    total_files += len(files)
    for f in files:
        ext = os.path.splitext(f)[1].lower() or "NO_EXT"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        fp = os.path.join(root, f)
        try:
            total_size += os.path.getsize(fp)
        except:
            pass

elapsed = time.time() - start_time
print(f"Total files: {total_files:,} across {len(top_counts)} top-level groupings in {elapsed:.2f}s")
print(f"Total size: {total_size / (1024**3):.2f} GB ({total_size:,} bytes)\n")

print("Files per top grouping:")
for k, v in sorted(top_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:30s} : {v:8,d} files")

print("\nFiles per extension:")
for k, v in sorted(ext_counts.items(), key=lambda x: -x[1])[:25]:
    print(f"  {k:15s} : {v:8,d} files")

