import os

man_dir = r"F:\NADI-Q data\ADNI DATA\ADNIMERGE2\ADNIMERGE2\man"
if os.path.exists(man_dir):
    rd_files = sorted(os.listdir(man_dir))
    print(f"Total .Rd files in man/: {len(rd_files)}")
    for f in rd_files[:10]:
        print(" ", f)
    # Print sample content of DXSUM.Rd or MMSE.Rd
    sample = os.path.join(man_dir, "DXSUM.Rd")
    if os.path.exists(sample):
        print(f"\n--- Sample from {sample} ---")
        with open(sample, "r", encoding="utf-8", errors="ignore") as fh:
            print(fh.read()[:600])

