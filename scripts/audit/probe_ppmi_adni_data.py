import os

print("--- ADNI DATA\\ADNIMERGE2\\ADNIMERGE2\\data ---")
d_path = r"F:\NADI-Q data\ADNI DATA\ADNIMERGE2\ADNIMERGE2\data"
if os.path.exists(d_path):
    for f in os.listdir(d_path):
        fp = os.path.join(d_path, f)
        print(f"  {f} ({os.path.getsize(fp):,} bytes)")

print("\n--- PPMI subdirectories ---")
ppmi_path = r"F:\NADI-Q data\PPMI"
for sub in sorted(os.listdir(ppmi_path)):
    sp = os.path.join(ppmi_path, sub)
    if os.path.isdir(sp):
        files = [f for _, _, fl in os.walk(sp) for f in fl]
        exts = set(os.path.splitext(f)[1].lower() for f in files)
        print(f"  DIR: {sub:30s} | Total files: {len(files):6d} | Extensions: {sorted(list(exts))[:8]}")
    else:
        print(f"  FILE: {sub:29s} | Size: {os.path.getsize(sp):,} bytes")

