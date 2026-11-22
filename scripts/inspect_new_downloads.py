import os
import glob
import numpy as np
import pandas as pd
import pydicom
from collections import defaultdict

def main():
    base_dir = r"F:\NADI-Q data\PPMI\MRI image download_dataset\PPMI"
    new_patnos = ["149006", "151050", "170472", "183263", "222893"]
    
    # Load source metadata
    xing_path = r"F:\NADI-Q data\PPMI\MRI_Acquisition\Xing_Core_Lab_-_MRI_acquisition_metadata_18Sep2026.csv"
    mri_path = r"F:\NADI-Q data\PPMI\MRI_Acquisition\Magnetic_Resonance_Imaging__MRI__18Sep2026.csv"
    df_xing = pd.read_csv(xing_path) if os.path.exists(xing_path) else pd.DataFrame()
    df_mri = pd.read_csv(mri_path) if os.path.exists(mri_path) else pd.DataFrame()
    
    print("=" * 80)
    print("INSPECTING NEW PARTICIPANT MRI DOWNLOADS")
    print("=" * 80)
    
    for patno in new_patnos:
        pat_dir = os.path.join(base_dir, patno)
        print(f"\n>>> PATNO: {patno}")
        if not os.path.exists(pat_dir):
            print(f"Directory {pat_dir} does not exist!")
            continue
            
        # Collect all files
        all_files = []
        for root, dirs, files in os.walk(pat_dir):
            for f in files:
                all_files.append(os.path.join(root, f))
        print(f"Total files on disk: {len(all_files)}")
        
        # Source metadata records
        p_int = int(patno)
        xing_sub = df_xing[df_xing["PATNO"] == p_int] if not df_xing.empty else pd.DataFrame()
        mri_sub = df_mri[df_mri["PATNO"] == p_int] if not df_mri.empty else pd.DataFrame()
        print("Source Xing records:")
        if not xing_sub.empty:
            for _, r in xing_sub.iterrows():
                print(f"  EVENT_ID: {r.get('EVENT_ID')}, MRI_DATE: {r.get('MRI_DATE')}, PROTOCOL: {r.get('PROTOCOL')}")
        else:
            print("  None found in Xing")
            
        print("Source MRI records:")
        if not mri_sub.empty:
            for _, r in mri_sub.iterrows():
                print(f"  EVENT_ID: {r.get('EVENT_ID')}, INFODT: {r.get('INFODT')}, MRICOMP: {r.get('MRICOMP')}")
        else:
            print("  None found in MRI table")
            
        # Group DICOM files by SeriesInstanceUID
        series_map = defaultdict(list)
        corrupt_files = []
        for fp in all_files:
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                suid = getattr(ds, "SeriesInstanceUID", "UNKNOWN")
                series_map[suid].append((fp, ds))
            except Exception as e:
                corrupt_files.append((fp, str(e)))
                
        if corrupt_files:
            print(f"Corrupt or unreadable files: {len(corrupt_files)}")
            for cf, err in corrupt_files[:5]:
                print(f"  {cf}: {err}")
                
        print(f"Distinct SeriesInstanceUIDs found: {len(series_map)}")
        for s_idx, (suid, items) in enumerate(series_map.items()):
            print(f"\n  --- Series #{s_idx + 1} (SUID: {suid}) ---")
            print(f"  Slice count: {len(items)}")
            sample = items[0][1]
            pid = getattr(sample, "PatientID", "N/A")
            study_date = getattr(sample, "StudyDate", "N/A")
            series_date = getattr(sample, "SeriesDate", "N/A")
            modality = getattr(sample, "Modality", "N/A")
            series_desc = getattr(sample, "SeriesDescription", "N/A")
            protocol = getattr(sample, "ProtocolName", "N/A")
            rows = getattr(sample, "Rows", "N/A")
            cols = getattr(sample, "Columns", "N/A")
            px_spacing = getattr(sample, "PixelSpacing", "N/A")
            slice_thick = getattr(sample, "SliceThickness", "N/A")
            
            print(f"  PatientID in DICOM: {pid}")
            print(f"  Modality: {modality}")
            print(f"  SeriesDescription: {series_desc}")
            print(f"  ProtocolName: {protocol}")
            print(f"  StudyDate: {study_date}, SeriesDate: {series_date}")
            print(f"  Rows x Cols: {rows} x {cols}")
            print(f"  PixelSpacing: {px_spacing}")
            print(f"  SliceThickness: {slice_thick}")
            
            # Subfolder of first slice
            rel_path = os.path.relpath(items[0][0], pat_dir)
            print(f"  Sample path relative to PATNO dir: {rel_path}")
            
            # Check full pixel reconstruction
            try:
                full_slices = [pydicom.dcmread(f[0]) for f in items]
                # Sorting check
                has_loc = all(hasattr(s, "SliceLocation") for s in full_slices)
                has_pos = all(hasattr(s, "ImagePositionPatient") for s in full_slices)
                has_inst = all(hasattr(s, "InstanceNumber") for s in full_slices)
                print(f"  Sorting attrs: SliceLocation={has_loc}, ImagePositionPatient={has_pos}, InstanceNumber={has_inst}")
                
                if has_pos:
                    full_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
                elif has_loc:
                    full_slices.sort(key=lambda x: float(x.SliceLocation))
                elif has_inst:
                    full_slices.sort(key=lambda x: int(x.InstanceNumber))
                    
                vol = np.stack([s.pixel_array for s in full_slices], axis=-1)
                is_finite = bool(np.isfinite(vol).all())
                var_val = float(vol.var())
                min_val = float(vol.min())
                max_val = float(vol.max())
                print(f"  3D Reconstruction: SUCCESS! Shape={vol.shape}")
                print(f"  Finite: {is_finite}, Min: {min_val}, Max: {max_val}, Variance: {var_val:.2f}")
            except Exception as e:
                print(f"  3D Reconstruction: FAILED! Reason: {e}")

if __name__ == "__main__":
    main()

