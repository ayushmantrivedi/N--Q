import os

dirs_to_check = [
    r"C:\Users\ayush\OneDrive\Desktop",
    r"F:\NADI-Q data"
]

for d in dirs_to_check:
    print(f"\n--- Checking {d} ---")
    if not os.path.exists(d): continue
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if not x.startswith('.')]
        for f in files:
            low = f.lower()
            if any(k in low for k in ['sih', 'srs', 'abstract', 'coherence', 'si26139', 'sih26139']):
                print(os.path.join(root, f))

