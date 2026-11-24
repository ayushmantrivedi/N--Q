# PUBLIC DATASET SELECTION V2 - PARKINSON'S DISEASE

## Kinematics - Gait in Parkinson's Disease
- **Identifier**: gaitpdb
- **Source URL**: https://physionet.org/content/gaitpdb/1.0.0/
- **License**: ODC-BY 1.0
- **Login Required**: No
- **Access Method**: direct_http
- **Download URL**: https://physionet.org/static/published-projects/gaitpdb/gait-in-parkinsons-disease-1.0.0.zip
- **File Format**: zip (containing txt)
- **Expected Size**: 302,436,928 bytes (~302 MB)
- **Subjects**: 93 PD + 73 Control
- **Longitudinal?**: False
- **Target Variables**: VGRF, gait_force, PD_Status
- **Reason for Selection**: Matches CTCLM requirement. Real human gait force sensor data for PD vs Control. Public, no login.
- **Status**: **PASS** (Headers and payload existence verified)

## MRI - OpenNeuro ds001907
- **Identifier**: ds001907
- **Source URL**: https://openneuro.org/datasets/ds001907
- **License**: CC0
- **Login Required**: No
- **Access Method**: openneuro AWS S3
- **Download URL**: https://s3.amazonaws.com/openneuro.org/ds001907/
- **Verified Sub-path**: sub-RC4101/ses-1/anat/sub-RC4101_ses-1_T1w.nii.gz
- **File Format**: NIfTI (.nii.gz)
- **Size**: ~2-5 GB Total
- **Subjects**: Multiple (e.g. sub-RC4101)
- **Longitudinal?**: Yes (e.g. ses-1, ses-2)
- **Target Variables**: T1w, PD_vs_Control
- **Reason for Selection**: Public real human T1w MRI of PD patients and controls. No login required. Verified exact S3 path.
- **Status**: **PASS** (Actual payload and S3 bucket routing verified)

## Biomarkers - Oxford Parkinson's Disease Acoustic Biomarker Dataset (UCI)
- **Identifier**: UCI ML Parkinsons
- **Source URL**: https://archive.ics.uci.edu/ml/datasets/parkinsons
- **License**: Open/Public Domain Citation
- **Login Required**: No
- **Access Method**: direct_http
- **Download URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data
- **File Format**: CSV (.data)
- **Expected Size**: ~40 KB
- **Subjects**: 31 (23 PD)
- **Record Count**: 195
- **Longitudinal?**: No (Multiple recordings per subject, treated cross-sectionally)
- **Target Variables**: MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Jitter(%), status (Diagnosis)
- **Reason for Selection**: Replaces the mocked placeholder. This is a legitimate, widely used, quantitative participant-level biomarker dataset (acoustic/digital biomarker) for PD that does not require a login.
- **Status**: **PASS** (URL returns 200 OK, payload is real data)
