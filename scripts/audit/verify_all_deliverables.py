import os

out_dir = r"F:\NADI-Q_AUDIT_OUTPUT"
expected = [
    "COMPLETE_DIRECTORY_INVENTORY.csv",
    "DATASET_SCHEMA_CATALOG.csv",
    "PARTICIPANT_COVERAGE_MATRIX.csv",
    "MODALITY_COMBINATION_RANKING.csv",
    "FEATURE_COVERAGE_RANKING.csv",
    "LONGITUDINAL_COVERAGE_REPORT.csv",
    "LABEL_AND_LEAKAGE_AUDIT.md",
    "OPEN_SOURCE_MODEL_COMPATIBILITY.md",
    "QUANTUM_FEATURE_FEASIBILITY.md",
    "NADIQ_COMPLETE_DATA_AUDIT.md"
]

print(f"Checking {len(expected)} deliverables in {out_dir}:")
all_ok = True
for f in expected:
    p = os.path.join(out_dir, f)
    exists = os.path.exists(p)
    sz = os.path.getsize(p) if exists else 0
    tag = "EXISTS" if exists else "MISSING"
    if not exists: all_ok = False
    print(f"  {tag:8s} : {f:38s} ({sz:10,d} bytes)")

print("\nAll deliverables present:", all_ok)

