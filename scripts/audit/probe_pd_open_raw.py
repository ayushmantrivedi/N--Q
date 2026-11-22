import os

p = r"F:\NADI-Q data\ADNI DATA\PD_OPEN\raw"
if os.path.exists(p):
    for f in sorted(os.listdir(p)):
        fp = os.path.join(p, f)
        print(f"  {f:40s} | Size: {os.path.getsize(fp):12,d} bytes")

