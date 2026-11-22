import os

p = r"F:\NADI-Q data\ADNI DATA\PD_OPEN\raw"
for root, dirs, files in os.walk(p):
    print("Folder:", root, f"({len(files)} files)")
    for f in files[:10]:
        fp = os.path.join(root, f)
        print(f"  {f} ({os.path.getsize(fp):,} bytes)")
    if len(files) > 10:
        print(f"  ... and {len(files)-10} more files")

