import os
import sys
import time
import requests
import json
import zipfile
import shutil
import psutil

try:
    import nibabel as nib
    import pandas as pd
    from ucimlrepo import fetch_ucirepo
except ImportError as e:
    print(f"Failed to import required libraries: {e}")
    sys.exit(1)

DATA_DIR = r"F:\NADI-Q data\PD_OPEN"
RAW_DIR = os.path.join(DATA_DIR, "raw")
MANIFEST_DIR = os.path.join(DATA_DIR, "data_manifests")
REPORT_DIR = os.path.join(DATA_DIR, "reports")

for d in [os.path.join(RAW_DIR, "mri"), os.path.join(RAW_DIR, "kinematics"), os.path.join(RAW_DIR, "biomarkers"), MANIFEST_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------
# CORE DOWNLOADER
# ---------------------------------------------------------
def download_robust(url, dest_path, min_size=1024, max_retries=3):
    """Robust downloader with strict .part handling, progress reporting, and range resume."""
    part_path = dest_path + ".part"
    print(f"\nSOURCE: {url}")
    print(f"TARGET: {dest_path}")
    
    # Check space
    try:
        free_bytes = psutil.disk_usage(os.path.splitdrive(dest_path)[0]).free
        if free_bytes < min_size:
            print(f"[BLOCKED] Insufficient disk space on drive. Free: {free_bytes}")
            return False, "BLOCKED", 0
    except:
        pass
        
    if os.path.exists(dest_path):
        current_sz = os.path.getsize(dest_path)
        if current_sz >= min_size:
            print(f"STATUS: SUCCESS (Already exists {current_sz} bytes)")
            return True, "SUCCESS", current_sz
            
    for attempt in range(max_retries):
        print(f"RETRY COUNT: {attempt}")
        headers = {}
        mode = 'wb'
        local_size = 0
        if os.path.exists(part_path):
            local_size = os.path.getsize(part_path)
            if local_size > 0:
                headers['Range'] = f"bytes={local_size}-"
                mode = 'ab'
                
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(10, 30)) as r:
                if r.status_code == 416:
                    print("STATUS: Range not satisfiable (file likely complete). Moving part file.")
                    os.rename(part_path, dest_path)
                    return True, "SUCCESS", os.path.getsize(dest_path)
                    
                if r.status_code == 200 and local_size > 0:
                    print("STATUS: Server returned 200 instead of 206. Restarting from zero.")
                    mode = 'wb'
                    local_size = 0
                elif r.status_code not in (200, 206):
                    print(f"STATUS: FAILED (HTTP {r.status_code})")
                    return False, "FAILED", 0
                
                remote_size = r.headers.get('Content-Length')
                remote_size = int(remote_size) if remote_size else None
                print(f"REMOTE SIZE: {remote_size}")
                
                content_type = r.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    print(f"STATUS: FAILED (Server returned HTML instead of data)")
                    return False, "FAILED", 0

                start_t = time.time()
                last_print = start_t
                downloaded_this_session = 0
                
                with open(part_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=81920):
                        if chunk:
                            f.write(chunk)
                            local_size += len(chunk)
                            downloaded_this_session += len(chunk)
                            
                        now = time.time()
                        if now - last_print > 5.0:
                            elapsed = now - start_t
                            speed = downloaded_this_session / elapsed if elapsed > 0 else 0
                            print(f"LOCAL SIZE: {local_size} | PROGRESS: {local_size} bytes | SPEED: {speed/1024/1024:.2f} MB/s")
                            last_print = now
                            
            if local_size < min_size:
                print(f"STATUS: FAILED (File too small: {local_size} bytes)")
                return False, "FAILED", 0
                
            os.rename(part_path, dest_path)
            print(f"FINAL STATUS: SUCCESS ({local_size} bytes)")
            return True, "SUCCESS", local_size
            
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            time.sleep(2)
            
    print("FINAL STATUS: BLOCKED (Network retries exhausted)")
    return False, "BLOCKED", 0

# ---------------------------------------------------------
# NETWORK TEST
# ---------------------------------------------------------
def network_test():
    print("NETWORK TEST")
    print("------------")
    
    # 1. OpenNeuro API
    try:
        r = requests.get('https://api.github.com/repos/OpenNeuroDatasets/ds001907/git/trees/master', timeout=10)
        print(f"OpenNeuro API: {'PASS' if r.status_code==200 else 'FAIL'}")
    except: print("OpenNeuro API: FAIL")
    
    # 2. OpenNeuro metadata
    try:
        r = requests.head('https://s3.amazonaws.com/openneuro.org/ds001907/dataset_description.json', timeout=10)
        print(f"OpenNeuro file metadata: {'PASS' if r.status_code==200 else 'FAIL'}")
    except: print("OpenNeuro file metadata: FAIL")
        
    # 3. PhysioNet small
    try:
        r = requests.head('https://physionet.org/content/gaitpdb/1.0.0/', timeout=10)
        print(f"PhysioNet: {'PASS' if r.status_code==200 else 'FAIL'}")
    except: print("PhysioNet: FAIL")
        
    # 4. UCI
    try:
        r = requests.head('https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data', timeout=10)
        print(f"UCI: {'PASS' if r.status_code==200 else 'FAIL'}")
    except: print("UCI: FAIL")

# ---------------------------------------------------------
# ACQUISITION FUNCTIONS
# ---------------------------------------------------------
def acquire_mri():
    target_subjects = 10
    manifest = []
    status = "SUCCESS"
    
    # Discover via GitHub API (since S3 HTTP dir listing is disabled)
    try:
        r = requests.get('https://api.github.com/repos/OpenNeuroDatasets/ds001907/git/trees/master?recursive=1', timeout=15)
        if r.status_code != 200:
            return "BLOCKED", 0, 0, manifest
        tree = r.json().get('tree', [])
    except Exception as e:
        print(f"[MRI] Discovery failed: {e}")
        return "BLOCKED", 0, 0, manifest
        
    t1w_files = [t['path'] for t in tree if t['path'].endswith('T1w.nii.gz')]
    
    if len(t1w_files) < target_subjects:
        print(f"[MRI] FAILED: Insufficient subjects discovered ({len(t1w_files)}). Minimum: {target_subjects}")
        return "FAILED", 0, 0, manifest
        
    t1w_files = t1w_files[:target_subjects]
    validated = 0
    
    for relative_path in t1w_files:
        url = f"https://s3.amazonaws.com/openneuro.org/ds001907/{relative_path}"
        local_filename = os.path.basename(relative_path)
        parts = local_filename.split('_')
        subject = next((p for p in parts if p.startswith('sub-')), "unknown")
        session = next((p for p in parts if p.startswith('ses-')), "unknown")
        
        dest = os.path.join(RAW_DIR, "mri", local_filename)
        ok, res_status, size = download_robust(url, dest, min_size=10240)
        if not ok:
            status = res_status if res_status == "BLOCKED" else "PARTIAL"
            continue
            
        # VALIDATE
        try:
            img = nib.load(dest)
            if len(img.shape) >= 3 and (img.get_fdata() == img.get_fdata()).all():
                validated += 1
                manifest.append({
                    "subject": subject,
                    "session": session,
                    "relative_path": relative_path,
                    "filename": local_filename,
                    "size": size,
                    "annexed": "no",
                    "source_dataset": "ds001907",
                    "snapshot": "master"
                })
        except Exception as e:
            print(f"[MRI] VALIDATION FAILED for {dest}: {e}")
            os.remove(dest)
            
    if validated == 0:
        status = "FAILED"
    elif validated < target_subjects and status != "BLOCKED":
        status = "PARTIAL"
        
    if manifest:
        pd.DataFrame(manifest).to_csv(os.path.join(MANIFEST_DIR, "mri_manifest.csv"), index=False)
        
    return status, len(t1w_files), validated, manifest

def acquire_kinematics():
    # Use robust download for the zip
    url = "https://physionet.org/static/published-projects/gaitpdb/gait-in-parkinsons-disease-1.0.0.zip"
    dest = os.path.join(RAW_DIR, "kinematics", "gaitpdb.zip")
    ok, status, size = download_robust(url, dest, min_size=1024*1024*5)
    
    manifest = []
    validated = 0
    total_files = 0
    if not ok:
        return status, 0, 0, manifest
        
    try:
        with zipfile.ZipFile(dest, 'r') as z:
            members = [m for m in z.namelist() if m.endswith('.txt') and 'demographics' not in m.lower()]
            total_files = len(members)
            z.extractall(os.path.join(RAW_DIR, "kinematics"))
            
            for m in members:
                full_path = os.path.join(RAW_DIR, "kinematics", m)
                if os.path.exists(full_path) and os.path.getsize(full_path) > 100:
                    validated += 1
                    manifest.append({
                        "participant": os.path.basename(full_path).split('_')[0],
                        "recording": os.path.basename(full_path),
                        "path": m,
                        "size": os.path.getsize(full_path)
                    })
    except zipfile.BadZipFile:
        print("[KINEMATICS] VALIDATION FAILED: Corrupted Zip")
        return "FAILED", 0, 0, manifest
        
    if manifest:
        pd.DataFrame(manifest).to_csv(os.path.join(MANIFEST_DIR, "kinematics_manifest.csv"), index=False)
        
    return "SUCCESS" if validated > 0 else "FAILED", total_files, validated, manifest

def acquire_biomarkers():
    print("\nSOURCE: UCI Repository ID 174")
    dest = os.path.join(RAW_DIR, "biomarkers", "parkinsons.csv")
    manifest = []
    
    try:
        from ucimlrepo import fetch_ucirepo
        # fetch dataset 
        print("TARGET: UCI Python API")
        parkinsons = fetch_ucirepo(id=174) 
        
        # data (as pandas dataframes) 
        X = parkinsons.data.features 
        y = parkinsons.data.targets
        
        df = pd.concat([X, y], axis=1)
        
        if df.shape[0] != 195:
            print(f"[BIOMARKER] FAILED: Expected 195 rows, got {df.shape[0]}")
            return "FAILED", 0, 0, manifest
            
        df.to_csv(dest, index=False)
        print(f"FINAL STATUS: SUCCESS (Saved to {dest})")
        
        manifest.append({
            "dataset": "UCI Parkinson's Disease",
            "id": 174,
            "rows": df.shape[0],
            "features": X.shape[1],
            "path": dest
        })
        pd.DataFrame(manifest).to_csv(os.path.join(MANIFEST_DIR, "biomarker_manifest.csv"), index=False)
        return "SUCCESS", df.shape[0], df.shape[0], manifest
        
    except Exception as e:
        print(f"UCI fetch failed via API: {e}")
        # fallback to HTTP
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
        ok, status, sz = download_robust(url, dest, min_size=1024)
        if ok:
            df = pd.read_csv(dest)
            if df.shape[0] > 0:
                manifest.append({
                    "dataset": "UCI Parkinson's Disease",
                    "id": 174,
                    "rows": df.shape[0],
                    "features": df.shape[1],
                    "path": dest
                })
                pd.DataFrame(manifest).to_csv(os.path.join(MANIFEST_DIR, "biomarker_manifest.csv"), index=False)
                return "SUCCESS", df.shape[0], df.shape[0], manifest
        return status, 0, 0, manifest

def main():
    network_test()
    
    print("\nStarting Acquisition...")
    
    m_stat, m_files, m_val, m_man = acquire_mri()
    k_stat, k_files, k_val, k_man = acquire_kinematics()
    b_stat, b_rows, b_val, b_man = acquire_biomarkers()
    
    # Save provenance
    prov = f"""# ACQUISITION V2 PROVENANCE

## MRI
Source: OpenNeuro ds001907
Status: {m_stat}
Files Discovered: {m_files}
Validated: {m_val}

## Kinematics
Source: PhysioNet gaitpdb
Status: {k_stat}
Files Discovered: {k_files}
Validated: {k_val}

## Biomarkers
Source: UCI ID 174
Status: {b_stat}
Rows Discovered: {b_rows}
Validated: {b_val}
"""
    with open(os.path.join(REPORT_DIR, "acquisition_v2.md"), "w", encoding="utf-8") as f:
        f.write(prov)
        
    print("\n==================================================")
    print("NADI-Q DATA ACQUISITION V2")
    print("==================================================")
    
    print(f"\nMRI:\n    Acquisition: {m_stat}\n    Files: {m_files}\n    Subjects: {len(set([m['subject'] for m in m_man]))}\n    Validated: {m_val}")
    print(f"\nKinematics:\n    Acquisition: {k_stat}\n    Files: {k_files}\n    Subjects: {len(set([m['participant'] for m in k_man]))}\n    Validated: {k_val}")
    print(f"\nBiomarker:\n    Acquisition: {b_stat}\n    Rows: {b_rows}\n    Subjects: {b_rows}\n    Validated: {b_val}")
    print("\n==================================================")
    
    print(f"Local Paths:")
    print(f"  MRI: {os.path.join(RAW_DIR, 'mri')}")
    print(f"  Kinematics: {os.path.join(RAW_DIR, 'kinematics')}")
    print(f"  Biomarkers: {os.path.join(RAW_DIR, 'biomarkers')}")
    
    if m_stat == "SUCCESS" and k_stat == "SUCCESS" and b_stat == "SUCCESS":
        print("\nDATA ACQUISITION = PASS")
    else:
        print("\nDATA ACQUISITION = NOT READY")

if __name__ == "__main__":
    main()
