import os

base = r"F:\NADI-Q data"
for top in ["ADNI DATA", "PD_OPEN", "PPMI"]:
    top_path = os.path.join(base, top)
    if not os.path.exists(top_path):
        print(f"Directory {top_path} does not exist.")
        continue
    items = sorted(os.listdir(top_path))
    print(f"\n{'='*60}\n{top} (Total entries: {len(items)})\n{'='*60}")
    for it in items:
        p = os.path.join(top_path, it)
        is_dir = os.path.isdir(p)
        tag = "DIR " if is_dir else "FILE"
        size = "" if is_dir else f"({os.path.getsize(p):,} bytes)"
        print(f"  {tag} : {it} {size}")

