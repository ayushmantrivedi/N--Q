import os
import sys
import time
import zipfile
import json
import csv
import re
import numpy as np
import pandas as pd
import pydicom
import nibabel as nib

base_dir = r"F:\NADI-Q data"
out_dir = r"F:\NADI-Q_AUDIT_OUTPUT"
os.makedirs(out_dir, exist_ok=True)

# Helper to identify dataset identity
def get_dataset_identity(rel_path):
    parts = rel_path.split(os.sep)
    top = parts[0] if parts else ""
    if "PPMI" in top.upper() or "PPMI" in rel_path.upper():
        if "PD_OPEN" in rel_path.upper():
            return "PD_OPEN"
        return "PPMI"
    elif "ADNI" in top.upper():
        if "PD_OPEN" in rel_path.upper():
            return "PD_OPEN"
        return "ADNI"
    elif "PD_OPEN" in top.upper():
        return "PD_OPEN"
    return "UNKNOWN"

# Candidate fields regexes
ID_PATTERNS = re.compile(r"^(PATNO|PTID|RID|SUBJECT|SUBJID|PARTICIPANT_ID|SUBJECTIDENTIFIER|STUDYID|ID)$", re.I)
VISIT_PATTERNS = re.compile(r"^(EVENT_ID|VISCODE|VISIT|VISITNO|VISIT_TYPE|EVENT|PAG_NAME|SES)$", re.I)
DATE_PATTERNS = re.compile(r"(DATE|INFODT|EXAMDATE|SCAN_DATE|ORIG_ENTRY|LAST_UPDATE|VISIT_DATE|YEAR)", re.I)
LABEL_PATTERNS = re.compile(r"(COHORT|DIAGNOSIS|DX|DXSUM|RESEARCH_GROUP|APPRDX|PRIMDIAG|STATUS|STAGE|GROUP)", re.I)

def scan_file_metadata(full_path, rel_path, rd_cache):
    ext = os.path.splitext(full_path)[1].lower()
    fname = os.path.basename(full_path)
    sz = os.path.getsize(full_path)
    dataset_id = get_dataset_identity(rel_path)
    
    file_type = ext.replace(".", "").upper() or "BINARY"
    row_count = None
    col_count = None
    schema_str = ""
    id_field = ""
    visit_field = ""
    date_field = ""
    label_field = ""
    status = "INSPECTED"
    notes = ""

    try:
        if ext == ".csv":
            file_type = "CSV"
            # Read first few lines for header
            with open(full_path, "r", encoding="utf-8", errors="ignore") as fh:
                first_line = fh.readline()
                dialect = csv.Sniffer().sniff(first_line[:1024]) if ("," in first_line or "\t" in first_line) else None
                delimiter = dialect.delimiter if dialect else ("," if "," in first_line else "\t")
            
            # Use pandas for first row schema
            df_sample = pd.read_csv(full_path, nrows=1, sep=delimiter, low_memory=False, encoding="utf-8", on_bad_lines="skip")
            cols = list(df_sample.columns)
            col_count = len(cols)
            schema_str = "; ".join(str(c) for c in cols[:20]) + (f"; ... (+{len(cols)-20} more)" if len(cols) > 20 else "")
            
            # Count lines
            with open(full_path, "rb") as fh:
                num_lines = sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1024*1024), b""))
            row_count = max(0, num_lines - 1)
            
            # Field identification
            id_field = "; ".join([c for c in cols if ID_PATTERNS.match(str(c).strip())])
            visit_field = "; ".join([c for c in cols if VISIT_PATTERNS.match(str(c).strip())])
            date_field = "; ".join([c for c in cols if DATE_PATTERNS.search(str(c).strip())][:5])
            label_field = "; ".join([c for c in cols if LABEL_PATTERNS.search(str(c).strip())][:5])

        elif ext == ".xlsx":
            file_type = "XLSX"
            import openpyxl
            wb = openpyxl.load_workbook(full_path, read_only=True)
            sheets_info = []
            tot_rows = 0
            for sname in wb.sheetnames:
                sh = wb[sname]
                tot_rows += (sh.max_row or 0)
                sheets_info.append(f"{sname} ({sh.max_row}x{sh.max_column})")
            row_count = tot_rows
            schema_str = "Sheets: " + ", ".join(sheets_info)
            notes = "Excel Workbook"

        elif ext == ".dcm":
            file_type = "DICOM"
            ds = pydicom.dcmread(full_path, stop_before_pixels=True, force=True)
            pid = str(getattr(ds, "PatientID", ""))
            s_uid = str(getattr(ds, "SeriesInstanceUID", ""))
            s_desc = str(getattr(ds, "SeriesDescription", ""))
            acq = str(getattr(ds, "MRAcquisitionType", ""))
            s_date = str(getattr(ds, "StudyDate", ""))
            schema_str = f"PID={pid}, SUID={s_uid[:15]}..., Desc={s_desc}, MRAcq={acq}"
            id_field = "PatientID" if pid else ""
            date_field = "StudyDate" if s_date else ""
            notes = f"DICOM slice: {s_desc} [{acq}]"

        elif ext in [".gz", ".nii"]:
            file_type = "NIFTI"
            try:
                img = nib.load(full_path, mmap=False)
                hdr = img.header
                shape = img.shape
                schema_str = f"Shape={shape}, Dtype={hdr.get_data_dtype()}, Zooms={hdr.get_zooms()}"
                row_count = shape[-1] if len(shape) >= 3 else 1
                notes = f"NIfTI 3D/4D Volume"
            except Exception as e:
                status = "PARSE_ERROR"
                notes = f"NIfTI error: {e}"

        elif ext == ".npy":
            file_type = "NUMPY"
            try:
                arr = np.load(full_path, mmap_mode="r")
                schema_str = f"Shape={arr.shape}, Dtype={arr.dtype}"
                row_count = arr.shape[0] if len(arr.shape) > 0 else 1
                col_count = arr.shape[1] if len(arr.shape) > 1 else None
            except Exception as e:
                notes = f"Numpy load: {e}"

        elif ext == ".rda":
            file_type = "RDA"
            stem = os.path.splitext(fname)[0]
            if stem in rd_cache:
                info = rd_cache[stem]
                row_count = info.get("obs")
                col_count = info.get("vars")
                schema_str = info.get("desc", "")
                id_field = info.get("id_field", "")
                visit_field = info.get("visit_field", "")
                date_field = info.get("date_field", "")
                label_field = info.get("label_field", "")
                notes = "Schema documented via .Rd roxygen metadata"
            else:
                notes = "R Data file (no direct .Rd found)"

        elif ext == ".rd":
            file_type = "RD_DOC"
            notes = "R Documentation source"

        elif ext == ".xml":
            file_type = "XML"
            with open(full_path, "r", encoding="utf-8", errors="ignore") as fh:
                txt = fh.read(512)
            schema_str = txt[:150].replace("\n", " ")
            notes = "XML Metadata stub"

        elif ext in [".json", ".jsonl"]:
            file_type = "JSON" if ext == ".json" else "JSONL"
            with open(full_path, "r", encoding="utf-8", errors="ignore") as fh:
                try:
                    data = json.load(fh)
                    if isinstance(data, list):
                        row_count = len(data)
                        schema_str = f"JSON Array (length {len(data)})"
                    elif isinstance(data, dict):
                        col_count = len(data.keys())
                        schema_str = f"Keys: {', '.join(list(data.keys())[:10])}"
                except:
                    notes = "Large or multi-line JSON"

        elif ext == ".zip":
            file_type = "ZIP_ARCHIVE"
            try:
                with zipfile.ZipFile(full_path, "r") as zf:
                    nl = zf.namelist()
                    row_count = len(nl)
                    schema_str = f"Contains {len(nl)} archived files: {', '.join(nl[:5])}"
                    notes = "Unextracted zip archive"
            except Exception as e:
                notes = f"Zip read error: {e}"

        elif ext in [".md", ".txt"]:
            file_type = "DOCUMENT_TEXT"
            with open(full_path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = sum(1 for _ in fh)
            row_count = lines
            notes = "Markdown / Plaintext documentation"

        elif ext in [".pt", ".pth"]:
            file_type = "PYTORCH_WEIGHTS"
            notes = "PyTorch Model checkpoint / tensor"

        elif ext == ".pdf":
            file_type = "PDF_DOCUMENT"
            notes = "Scientific protocol / documentation PDF"

        elif ext == ".fdt":
            file_type = "EEG_BINARY"
            notes = "EEGLAB Binary Signal Data (OpenNeuro)"

    except Exception as e:
        status = "INSPECTION_WARNING"
        notes = f"Warning during inspect: {str(e)}"

    return {
        "relative_path": rel_path,
        "file_name": fname,
        "extension": ext,
        "file_size_bytes": sz,
        "file_type": file_type,
        "record_or_row_count": row_count,
        "column_count": col_count,
        "available_schema": schema_str,
        "dataset_identity": dataset_id,
        "participant_id_field": id_field,
        "visit_event_id_field": visit_field,
        "date_field": date_field,
        "label_diagnosis_field": label_field,
        "inspection_status": status,
        "notes": notes
    }

print("Scan helper compiled successfully.")

