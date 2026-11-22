import os
import pydicom

base = r"F:\NADI-Q data\PPMI\MRI_raw\PPMI"
if os.path.exists(base):
    for p in sorted(os.listdir(base)):
        p_dir = os.path.join(base, p)
        if not os.path.isdir(p_dir):
            continue
        found = False
        for root, dirs, files in os.walk(p_dir):
            for f in files:
                if f.lower().endswith(".dcm"):
                    ds = pydicom.dcmread(os.path.join(root, f), stop_before_pixels=True)
                    acq = getattr(ds, "MRAcquisitionType", "N/A")
                    desc = getattr(ds, "SeriesDescription", "N/A")
                    thick = getattr(ds, "SliceThickness", "N/A")
                    print(f"Ref {p}: MRAcqType={acq}, Desc={desc}, Thick={thick}")
                    found = True
                    break
            if found:
                break

