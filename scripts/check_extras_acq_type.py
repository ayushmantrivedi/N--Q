import os
import pydicom

base = r"F:\NADI-Q data\PPMI\MRI image download_dataset\PPMI"
extra_patnos = ["200627", "204745", "231122", "233881", "239437", "240559", "252794", "292996", "292998"]

for p in extra_patnos:
    p_dir = os.path.join(base, p)
    found = False
    for root, dirs, files in os.walk(p_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                ds = pydicom.dcmread(os.path.join(root, f), stop_before_pixels=True)
                acq = getattr(ds, "MRAcquisitionType", "N/A")
                desc = getattr(ds, "SeriesDescription", "N/A")
                thick = getattr(ds, "SliceThickness", "N/A")
                print(f"Extra {p}: MRAcqType={acq}, Desc={desc}, Thick={thick}")
                found = True
                break
        if found:
            break

