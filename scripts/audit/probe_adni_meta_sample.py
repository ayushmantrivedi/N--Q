import os

p = r"F:\NADI-Q data\ADNI DATA\ADNI1_Complete_3Yr_3T_IDA_Metadata\ADNI\002_S_0413"
for root, dirs, files in os.walk(p):
    print("Folder:", root)
    for f in files:
        fp = os.path.join(root, f)
        print(f"  {f} ({os.path.getsize(fp):,} bytes)")

