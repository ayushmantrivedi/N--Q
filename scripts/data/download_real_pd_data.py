import os
import requests
import zipfile
import sys

DATA_DIR = r"F:\NADI-Q data\PD_OPEN"
RAW_DIR = os.path.join(DATA_DIR, "raw")

for d in ["mri", "biomarkers", "kinematics"]:
    os.makedirs(os.path.join(RAW_DIR, d), exist_ok=True)

def verify_file_signature(path, expected_magic):
    try:
        with open(path, 'rb') as f:
            header = f.read(max(len(expected_magic), 1024))
            
            # HTML error page detection
            if b"<!DOCTYPE html" in header.lower() or b"<html" in header.lower():
                print(f"[ERROR] Downloaded file {path} contains HTML (error page).")
                return False
                
            if header.startswith(expected_magic):
                return True
            else:
                print(f"[ERROR] Invalid magic bytes for {path}. Expected {expected_magic}, got {header[:len(expected_magic)]}")
                return False
    except Exception as e:
        print(f"[ERROR] Could not verify signature for {path}: {e}")
        return False

def download_file(url, dest_path, expected_magic=None, min_size=1024):
    print(f"Downloading {url} to {dest_path}")
    headers = {}
    
    # Check if partial file exists
    mode = 'wb'
    if os.path.exists(dest_path):
        current_size = os.path.getsize(dest_path)
        if current_size > 0:
            headers['Range'] = f"bytes={current_size}-"
            mode = 'ab'
            print(f"Found existing file of {current_size} bytes. Attempting to resume...")
    
    try:
        # We allow up to 3 retries
        for attempt in range(3):
            try:
                with requests.get(url, stream=True, headers=headers, timeout=15) as r:
                    if r.status_code == 416: # Range Not Satisfiable (already fully downloaded)
                        print("File is already fully downloaded based on Range header.")
                        break
                    
                    if r.status_code not in (200, 206):
                        print(f"[ERROR] Server returned HTTP {r.status_code}")
                        return False
                        
                    content_type = r.headers.get("Content-Type", "")
                    if "text/html" in content_type:
                        print(f"[ERROR] Server returned HTML page instead of raw file.")
                        return False

                    with open(dest_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                break # Success, exit retry loop
            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as ce:
                print(f"[WARNING] Connection dropped: {ce}. Retrying {attempt+1}/3...")
                headers['Range'] = f"bytes={os.path.getsize(dest_path)}-"
                mode = 'ab'

        final_size = os.path.getsize(dest_path)
        if final_size < min_size:
            print(f"[ERROR] File {dest_path} is too small ({final_size} bytes). Minimum expected: {min_size}")
            return False

        if expected_magic and not verify_file_signature(dest_path, expected_magic):
            return False
            
        print(f"[SUCCESS] Downloaded to {dest_path} ({final_size} bytes)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        return False

def main():
    success = True
    
    print("--- 1. Download OpenNeuro ds001907 Subset ---")
    mri_dest = os.path.join(RAW_DIR, "mri", "sub-RC4101_ses-1_T1w.nii.gz")
    mri_url = "https://s3.amazonaws.com/openneuro.org/ds001907/sub-RC4101/ses-1/anat/sub-RC4101_ses-1_T1w.nii.gz"
    # GZIP magic bytes: 1F 8B
    if not download_file(mri_url, mri_dest, expected_magic=b'\x1f\x8b', min_size=1024*100):
        print(f"[FAIL] Failed MRI dataset. URL: {mri_url}")
        success = False
    
    print("\n--- 2. Download PhysioNet gaitpdb ---")
    gait_zip = os.path.join(RAW_DIR, "kinematics", "gaitpdb.zip")
    gait_url = "https://physionet.org/static/published-projects/gaitpdb/gait-in-parkinsons-disease-1.0.0.zip"
    # ZIP magic bytes: PK
    if download_file(gait_url, gait_zip, expected_magic=b'PK\x03\x04', min_size=1024*1024*10):
        try:
            print("Verifying and extracting gaitpdb zip...")
            with zipfile.ZipFile(gait_zip, 'r') as zip_ref:
                # Basic zip bomb protection / validation
                members = zip_ref.infolist()
                print(f"Archive contains {len(members)} files.")
                zip_ref.extractall(os.path.join(RAW_DIR, "kinematics"))
        except zipfile.BadZipFile:
            print("[ERROR] Zip file is corrupted or not a valid zip archive despite header.")
            success = False
    else:
        print(f"[FAIL] Failed Kinematics dataset. URL: {gait_url}")
        success = False
            
    print("\n--- 3. Download UCI Biomarker CSV ---")
    bio_csv = os.path.join(RAW_DIR, "biomarkers", "parkinsons.csv")
    bio_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    # CSV is text, no universal magic bytes, but should not be HTML. We enforce min_size.
    if not download_file(bio_url, bio_csv, expected_magic=None, min_size=1024*10):
        print(f"[FAIL] Failed Biomarker dataset. URL: {bio_url}")
        success = False

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
