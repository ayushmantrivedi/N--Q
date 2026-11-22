import os
import glob

DATA_DIR = r"F:\NADI-Q data"

saa_docs = []
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if 'saa' in f.lower() or 'alpha-synuclein' in f.lower() or 'codebook' in f.lower() or 'dictionary' in f.lower():
            saa_docs.append(os.path.join(root, f))

print("Found files:", len(saa_docs))
for f in saa_docs[:30]:
    print(f)

