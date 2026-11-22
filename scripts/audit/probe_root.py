import os

base = r"F:\NADI-Q data"
if not os.path.exists(base):
    print(f"Error: {base} does not exist!")
else:
    items = os.listdir(base)
    print(f"Total items in {base}: {len(items)}")
    for item in sorted(items):
        p = os.path.join(base, item)
        is_dir = os.path.isdir(p)
        tag = "DIR " if is_dir else "FILE"
        size = "" if is_dir else f"({os.path.getsize(p):,} bytes)"
        print(f"  {tag} : {item} {size}")

