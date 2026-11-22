import os
import time
import pydicom

base = r"F:\NADI-Q data\PPMI\MRI image download_dataset\PPMI"
t0 = time.time()
count = 0
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith(".dcm"):
            count += 1
t1 = time.time()
print(f"Traversed {count} DICOM paths in {t1 - t0:.2f}s")

