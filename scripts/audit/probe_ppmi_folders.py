import os
import pandas as pd

for folder in ["Motor___MDS-UPDRS", "Biospecimen_Analysis", "MR_Analysis_Results", "MRI_Acquisition", "Study_Enrollment", "_Subject_Characteristics"]:
    p = os.path.join(r"F:\NADI-Q data\PPMI", folder)
    if not os.path.exists(p): continue
    print(f"\n{'='*70}\n{folder}\n{'='*70}")
    for f in os.listdir(p):
        fp = os.path.join(p, f)
        if os.path.isdir(fp):
            print(f"  DIR : {f}")
        else:
            sz = os.path.getsize(fp)
            if f.endswith(".csv"):
                try:
                    df = pd.read_csv(fp, nrows=2)
                    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                        cnt = sum(1 for _ in fh) - 1
                    cols = len(df.columns)
                    print(f"  FILE: {f:45s} | {cnt:6,d} rows | {cols:3d} cols | {sz:10,d} bytes")
                except Exception as e:
                    print(f"  FILE: {f:45s} | Error: {e}")
            else:
                print(f"  FILE: {f:45s} | {sz:10,d} bytes")

