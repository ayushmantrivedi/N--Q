import os
import pandas as pd

p = r"F:\NADI-Q data\PPMI\Roche_Smartphone_App"
for f in os.listdir(p):
    fp = os.path.join(p, f)
    print(f"File: {f} ({os.path.getsize(fp):,} bytes)")
    if f.endswith(".csv"):
        df = pd.read_csv(fp, nrows=5)
        print("  Columns:", df.columns.tolist()[:15])
        print("  Total rows approx...")
        with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
            row_count = sum(1 for _ in fh)
        print(f"  Exact row count: {row_count:,}")

