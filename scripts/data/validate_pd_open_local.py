import os
import sys
import glob
try:
    import nibabel as nib
    import pandas as pd
except ImportError:
    print("[CRITICAL] Missing required dependencies (nibabel, pandas) for deep validation.")
    sys.exit(1)

DATA_DIR = r"F:\NADI-Q data\PD_OPEN"
RAW_DIR = os.path.join(DATA_DIR, "raw")

def validate_mri():
    files = glob.glob(os.path.join(RAW_DIR, "mri", "**", "*.nii*"), recursive=True)
    if len(files) == 0:
        print("[ERROR] MRI Validation Failed: No NIfTI files found.")
        return False
    
    print(f"[OK] MRI Validation: {len(files)} NIfTI files found.")
    
    # Deep validation
    try:
        img = nib.load(files[0])
        shape = img.shape
        if len(shape) < 3:
            print(f"[ERROR] MRI Validation Failed: Volume is not 3D. Shape {shape}")
            return False
            
        print(f"[OK] MRI Volume Shape: {shape}")
        print(f"[OK] MRI Affine: \n{img.affine}")
        print(f"[OK] MRI Voxel Spacing: {img.header.get_zooms()}")
        print(f"[OK] MRI Datatype: {img.get_data_dtype()}")
        
        # Test loading a slice to verify finite values
        data_slice = img.slicer[..., shape[2]//2].get_fdata()
        if not (data_slice == data_slice).all(): # checks for NaN
            print("[ERROR] MRI Validation Failed: Non-finite values detected in volume.")
            return False
            
    except Exception as e:
        print(f"[ERROR] Could not deep validate NIfTI (corrupt or unreadable): {e}")
        return False
        
    return True

def validate_kinematics():
    # After extracting gaitpdb, the folders have format like 'gait-in-parkinsons-disease-1.0.0/demographics.txt'
    # and actual recordings in subdirectories.
    files = glob.glob(os.path.join(RAW_DIR, "kinematics", "**", "*.txt"), recursive=True)
    
    # Filter for data files vs demographic files
    data_files = [f for f in files if "demographics" not in f.lower() and os.path.getsize(f) > 1024]
    
    if len(data_files) == 0:
        print("[ERROR] Kinematics Validation Failed: No temporal recordings found.")
        return False
        
    print(f"[OK] Kinematics Validation: {len(data_files)} recordings found.")
    
    try:
        # Check first file for numeric time series
        with open(data_files[0], 'r') as f:
            lines = f.readlines()
            if len(lines) < 10:
                print(f"[ERROR] Kinematics Validation Failed: File {data_files[0]} has too few rows.")
                return False
            
            # Check number of columns in the first data row
            cols = len(lines[0].strip().split())
            print(f"[OK] Kinematics Temporal Columns per sample: {cols}")
    except Exception as e:
        print(f"[ERROR] Could not validate Kinematics temporal recording: {e}")
        return False
        
    return True

def validate_biomarkers():
    files = glob.glob(os.path.join(RAW_DIR, "biomarkers", "**", "*.csv"), recursive=True)
    
    # UCI datasets sometimes end in .data
    if not files:
        files = glob.glob(os.path.join(RAW_DIR, "biomarkers", "**", "*.data"), recursive=True)
        
    if len(files) == 0:
        print("[ERROR] Biomarker Validation Failed: No tabular files found.")
        return False
        
    print(f"[OK] Biomarker Validation: Tabular file found.")
    
    try:
        df = pd.read_csv(files[0])
        if df.shape[0] == 0:
            print("[ERROR] Biomarker CSV is empty.")
            return False
            
        print(f"[OK] Biomarker subjects/rows: {df.shape[0]}")
        print(f"[OK] Biomarker variables/columns: {df.shape[1]}")
        
        # Missingness report
        missing = df.isnull().sum().sum()
        print(f"[OK] Biomarker missing values across dataset: {missing}")
        
        # Numeric variables check
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 5:
            print(f"[ERROR] Biomarker CSV lacks sufficient quantitative numerical variables.")
            return False
            
        print(f"[OK] Biomarker quantitative variables count: {len(numeric_cols)}")
        
    except Exception as e:
        print(f"[ERROR] Could not deep validate Biomarker file: {e}")
        return False
        
    return True

def main():
    print("--- NADI-Q LOCAL DEEP VALIDATION GATE ---")
    
    print("\n--- Validating MRI ---")
    mri_ok = validate_mri()
    
    print("\n--- Validating Kinematics ---")
    kin_ok = validate_kinematics()
    
    print("\n--- Validating Biomarkers ---")
    bio_ok = validate_biomarkers()
    
    if not (mri_ok and kin_ok and bio_ok):
        print("\n[CRITICAL] One or more datasets failed deep validation.")
        print("[CRITICAL] Do NOT substitute synthetic data. Halting pipeline.")
        sys.exit(1)
        
    print("\n[SUCCESS] All datasets passed physical, structural, and content validation gates.")
    sys.exit(0)

if __name__ == "__main__":
    main()
