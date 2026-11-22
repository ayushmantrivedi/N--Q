import os

print("--- ADNI DATA\\PD_OPEN tree ---")
pd_open = r"F:\NADI-Q data\ADNI DATA\PD_OPEN"
for sub in sorted(os.listdir(pd_open)):
    sp = os.path.join(pd_open, sub)
    if os.path.isdir(sp):
        files = [f for _, _, fl in os.walk(sp) for f in fl]
        exts = set(os.path.splitext(f)[1].lower() for f in files)
        print(f"  DIR: {sub:20s} | Total files: {len(files):5d} | Extensions: {sorted(list(exts))[:8]}")
    else:
        print(f"  FILE: {sub:19s} | Size: {os.path.getsize(sp):,} bytes")

print("\n--- ADNI DATA\\ADNI1_Complete_3Yr_3T_IDA_Metadata\\ADNI tree ---")
adni_ida = r"F:\NADI-Q data\ADNI DATA\ADNI1_Complete_3Yr_3T_IDA_Metadata\ADNI"
if os.path.exists(adni_ida):
    for sub in sorted(os.listdir(adni_ida))[:10]:
        sp = os.path.join(adni_ida, sub)
        if os.path.isdir(sp):
            files = [f for _, _, fl in os.walk(sp) for f in fl]
            print(f"  DIR: {sub:20s} | Files: {len(files)}")
        else:
            print(f"  FILE: {sub:19s}")

