import os

audit_out = r"F:\NADI-Q_AUDIT_OUTPUT"
os.makedirs(audit_out, exist_ok=True)
print(f"Audit output directory ready: {audit_out} (Exists: {os.path.exists(audit_out)})")

# Check ADNI DATA\PD_OPEN
p = r"F:\NADI-Q data\ADNI DATA\PD_OPEN"
if os.path.exists(p):
    items = os.listdir(p)
    print(f"\nADNI DATA\\PD_OPEN items ({len(items)}):")
    for it in items:
        sub_p = os.path.join(p, it)
        tag = "DIR " if os.path.isdir(sub_p) else "FILE"
        print(f"  {tag}: {it}")

# Check ADNIMERGE2
p2 = r"F:\NADI-Q data\ADNI DATA\ADNIMERGE2"
if os.path.exists(p2):
    items = os.listdir(p2)
    print(f"\nADNI DATA\\ADNIMERGE2 items ({len(items)}):")
    for it in items:
        sub_p = os.path.join(p2, it)
        tag = "DIR " if os.path.isdir(sub_p) else "FILE"
        print(f"  {tag}: {it}")

# Check ADNI1_Complete_3Yr_3T_IDA_Metadata
p3 = r"F:\NADI-Q data\ADNI DATA\ADNI1_Complete_3Yr_3T_IDA_Metadata"
if os.path.exists(p3):
    items = os.listdir(p3)
    print(f"\nADNI DATA\\ADNI1_Complete_3Yr_3T_IDA_Metadata items ({len(items)}):")
    for it in items[:15]:
        sub_p = os.path.join(p3, it)
        tag = "DIR " if os.path.isdir(sub_p) else "FILE"
        print(f"  {tag}: {it}")
    if len(items) > 15:
        print(f"  ... and {len(items)-15} more")

