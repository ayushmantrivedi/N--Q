import os
import glob
import pandas as pd
import pydicom
import numpy as np

def reconstruct_volume(slices):
    """Attempt to reconstruct a 3D volume from a list of DICOM slices."""
    if not slices: return False, None
    try:
        # Sort by SliceLocation if available, else InstanceNumber
        if hasattr(slices[0], 'SliceLocation'):
            slices.sort(key=lambda x: float(x.SliceLocation))
        elif hasattr(slices[0], 'InstanceNumber'):
            slices.sort(key=lambda x: int(x.InstanceNumber))
        else:
            return False, None
            
        # Ensure consistent dimensions
        rows, cols = slices[0].Rows, slices[0].Columns
        if any(s.Rows != rows or s.Columns != cols for s in slices):
            return False, None
            
        vol = np.stack([s.pixel_array for s in slices], axis=-1)
        return True, vol
    except Exception as e:
        return False, None

def validate_mri_manifest_record(row):
    """
    Validates whether a manifest row is internally consistent and eligible for baseline acquisition.
    Returns: (is_valid: bool, error_reason: str)
    """
    visit = str(row.get('visit', '')).strip().upper()
    event_id = str(row.get('EVENT_ID', '')).strip().upper()
    
    # If manifest claims baseline visit, the source event must not be a follow-up
    if visit in ['BL', 'BASELINE']:
        if event_id and event_id not in ['BL', 'SC']:
            return False, f"DISCREPANT_VISIT_FOLLOWUP_EVENT_{event_id}_INCOMPATIBLE_WITH_BASELINE"
    return True, "VALID"

def validate_manifest_dataframe(df):
    """
    Audits an entire manifest DataFrame for visit consistency.
    Returns: (valid_mask: pd.Series, reasons: pd.Series)
    """
    valid_mask = []
    reasons = []
    for _, row in df.iterrows():
        is_valid, reason = validate_mri_manifest_record(row)
        valid_mask.append(is_valid)
        reasons.append(reason)
    return pd.Series(valid_mask, index=df.index), pd.Series(reasons, index=df.index)

def verify_mri_batch(manifest_path, raw_mri_dir, force_reaudit=False):
    """
    Verifies downloaded DICOMs against manifest using series-level consistency
    and physical 3D acquisition verification (MRAcquisitionType).
    """
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    updated = 0
    
    for idx, row in df.iterrows():
        if not force_reaudit and row['status'] in ['PREPROCESS_PENDING', 'VERIFIED'] and str(row.get('acquisition_type_verified', '')) == 'VERIFIED_3D_T1':
            continue
            
        patno = str(row['PATNO'])
        subj_dir = os.path.join(raw_mri_dir, patno)
        
        if not os.path.exists(subj_dir):
            continue
            
        dcm_files = []
        for root, dirs, files in os.walk(subj_dir):
            for f in files:
                if f.endswith('.dcm') or f.endswith('.DCM'):
                    dcm_files.append(os.path.join(root, f))
                    
        if not dcm_files:
            continue
            
        df.at[idx, 'status'] = 'DOWNLOADED'
        df.at[idx, 'download_status'] = 'SUCCESS'
        df.at[idx, 'dicom_file_count'] = len(dcm_files)
        
        # Enforce baseline manifest consistency: a manifest labeled baseline cannot acquire a follow-up scan
        is_valid_manifest, reason = validate_mri_manifest_record(row)
        if not is_valid_manifest:
            df.at[idx, 'status'] = 'FAILED'
            df.at[idx, 'verification_status'] = reason
            updated += 1
            continue
        
        # Group by SeriesInstanceUID
        series_dict = {}
        identity_failed = False
        
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                pid = str(getattr(ds, 'PatientID', '')).strip()
                if pid != patno:
                    identity_failed = True
                    break
                    
                suid = getattr(ds, 'SeriesInstanceUID', 'UNKNOWN')
                if suid not in series_dict:
                    series_dict[suid] = []
                series_dict[suid].append(f)
            except:
                pass
                
        if identity_failed:
            df.at[idx, 'status'] = 'FAILED'
            df.at[idx, 'verification_status'] = 'FAILED_IDENTITY'
            continue
            
        if not series_dict:
            df.at[idx, 'status'] = 'FAILED'
            df.at[idx, 'verification_status'] = 'NO_VALID_SERIES'
            continue
            
        eligible_series = []
        ambiguous_series = []
        failed_series = []
        
        for suid, files in series_dict.items():
            try:
                # Load full files for reconstruction
                slices = [pydicom.dcmread(f) for f in files]
                
                # 1. Physical 3D acquisition verification (MRAcquisitionType MUST be 3D)
                acq_types = {str(getattr(s, 'MRAcquisitionType', '')).strip().upper() for s in slices}
                if '2D' in acq_types:
                    failed_series.append((suid, 'INELIGIBLE_2D_ACQUISITION'))
                    continue
                if '3D' not in acq_types:
                    ambiguous_series.append((suid, 'AMBIGUOUS_NOT_3D_ACQUISITION'))
                    continue

                # 2. Check sequence description / protocol name
                desc = getattr(slices[0], 'SeriesDescription', '').lower() + getattr(slices[0], 'ProtocolName', '').lower()
                if any(x in desc for x in ['t2', 'flair', 'dti', 'rsfmri', 'fmri']):
                    failed_series.append((suid, 'INELIGIBLE_SEQUENCE'))
                    continue
                    
                if any(x in desc for x in [' 2d', '_2d', '2d ', '2-d']):
                    failed_series.append((suid, 'INELIGIBLE_2D_SEQUENCE_NAME'))
                    continue
                    
                if 't1' not in desc and 'mprage' not in desc and 'spgr' not in desc and 'tfl' not in desc and 'bravo' not in desc:
                    ambiguous_series.append((suid, 'AMBIGUOUS_SEQUENCE'))
                    continue
                    
                # 3. Check partition count and slice thickness for 3D volumetric coverage
                slice_thick = float(getattr(slices[0], 'SliceThickness', 1.0) or 1.0)
                if slice_thick > 2.0:
                    failed_series.append((suid, f"EXCESSIVE_SLICE_THICKNESS_{slice_thick}mm"))
                    continue
                    
                if len(slices) < 100:
                    failed_series.append((suid, f"INSUFFICIENT_PARTITIONS_{len(slices)}"))
                    continue

                # 4. Reconstruct volume
                success, vol = reconstruct_volume(slices)
                if not success:
                    failed_series.append((suid, 'RECONSTRUCTION_FAILED'))
                    continue
                    
                # 3D check
                if len(vol.shape) != 3 or vol.shape[2] < 2:
                    failed_series.append((suid, 'NOT_3D_VOLUME'))
                    continue
                    
                # Variance check
                if vol.var() == 0:
                    failed_series.append((suid, 'ZERO_VARIANCE'))
                    continue
                    
                # NaN/Inf check
                if not np.isfinite(vol).all():
                    failed_series.append((suid, 'NAN_INF_VALUES'))
                    continue
                    
                # All checks passed
                eligible_series.append((suid, vol.shape, slices[0]))
                
            except Exception as e:
                failed_series.append((suid, f"ERROR_{str(e)}"))
                
        # Record series metadata
        df.at[idx, 'total_series_found'] = len(series_dict)
        df.at[idx, 'eligible_series_count'] = len(eligible_series)
        
        if len(eligible_series) == 0:
            df.at[idx, 'prior_status'] = str(row.get('status', ''))
            df.at[idx, 'status'] = 'REVIEW' if len(ambiguous_series) > 0 else 'FAILED'
            if any('INELIGIBLE_2D' in f[1] for f in failed_series):
                df.at[idx, 'verification_status'] = 'VERIFIED_2D_NOT_ELIGIBLE'
                df.at[idx, 'acquisition_type_verified'] = 'VERIFIED_2D_NOT_ELIGIBLE'
            elif len(ambiguous_series) > 0:
                df.at[idx, 'verification_status'] = 'AMBIGUOUS_NEEDS_MANUAL_REVIEW'
                df.at[idx, 'acquisition_type_verified'] = 'AMBIGUOUS_NEEDS_MANUAL_REVIEW'
            else:
                df.at[idx, 'verification_status'] = 'NO_ELIGIBLE_T1_SERIES'
                df.at[idx, 'acquisition_type_verified'] = 'INELIGIBLE'
            updated += 1
            continue
            
        # Select best series (just take the first eligible one, but explicitly log it)
        selected_suid, shape, ds_ref = eligible_series[0]
        
        df.at[idx, 'prior_status'] = str(row.get('status', ''))
        df.at[idx, 'status'] = 'PREPROCESS_PENDING'
        df.at[idx, 'verification_status'] = 'VERIFIED_3D_T1'
        df.at[idx, 'acquisition_type_verified'] = 'VERIFIED_3D_T1'
        df.at[idx, 'mr_acquisition_type'] = getattr(ds_ref, 'MRAcquisitionType', '3D')
        df.at[idx, 'image_eligibility'] = 'ELIGIBLE'
        df.at[idx, 'selected_series_uid'] = selected_suid
        df.at[idx, 'reconstructed_shape'] = f"{shape[0]}x{shape[1]}x{shape[2]}"
        df.at[idx, 'voxel_spacing'] = str(getattr(ds_ref, 'PixelSpacing', [0,0]))
        df.at[idx, 'slice_thickness'] = str(getattr(ds_ref, 'SliceThickness', ''))
        
        if len(eligible_series) > 1:
            df.at[idx, 'multiple_t1_note'] = f"Selected {selected_suid} from {len(eligible_series)} eligible."
            
        updated += 1
            
    if updated > 0 or 'dicom_file_count' in df.columns:
        df.to_csv(manifest_path, index=False)
        print(f"Verified and updated manifest records.")
    else:
        print("No new valid acquisitions verified.")

if __name__ == "__main__":
    import sys
    manifest = sys.argv[1] if len(sys.argv) > 1 else r'F:\NADI-Q data\PPMI\manifests\ppmi_mri_cohort_expansion_v1.csv'
    raw_dir = sys.argv[2] if len(sys.argv) > 2 else r'F:\NADI-Q data\PPMI\MRI_raw\PPMI'
    force = ('--force' in sys.argv) or ('-f' in sys.argv)
    verify_mri_batch(manifest, raw_dir, force_reaudit=force)
