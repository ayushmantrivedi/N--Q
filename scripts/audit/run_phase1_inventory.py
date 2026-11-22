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
import openpyxl

base_dir = r"F:\NADI-Q data"
out_dir = r"F:\NADI-Q_AUDIT_OUTPUT"
os.makedirs(out_dir, exist_ok=True)
csv_out_path = os.path.join(out_dir, "COMPLETE_DIRECTORY_INVENTORY.csv")

# 1. Build .Rd cache for ADNIMERGE2
man_dir = r"F:\NADI-Q data\ADNI DATA\ADNIMERGE2\ADNIMERGE2\man"
obs_re = re.compile(r"A data frame with (\d+) observations and (\d+) variables", re.I)
rd_cache = {}
if os.path.exists(man_dir):
    for fname in os.listdir(man_dir):
        if not fname.endswith(".Rd"): continue
        stem = fname[:-3]
        fp = os.path.join(man_dir, fname)
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                txt = fh.read()
            m = obs_re.search(txt)
            obs = int(m.group(1)) if m else None
            vars_ = int(m.group(2)) if m else None
            items = re.findall(r"\\item\s+\\strong\{([^}]+)\}", txt)
            id_fields = [it for it in items if it.upper() in ["RID", "PTID", "ID", "SUBJID"]]
            visit_fields = [it for it in items if it.upper() in ["VISCODE", "VISCODE2", "EVENT_ID", "VISIT"]]
            date_fields = [it for it in items if any(k in it.upper() for k in ["DATE", "EXAMDATE", "YEAR"])]
            label_fields = [it for it in items if any(k in it.upper() for k in ["DX", "DIAG", "STATUS", "COHORT"])]
            
            rd_cache[stem] = {
                "obs": obs,
                "vars": vars_,
                "cols": items,
                "desc": f"{len(items)} cols: {', '.join(items[:15])}" if items else "",
                "id_field": "; ".join(id_fields),
                "visit_field": "; ".join(visit_fields),
                "date_field": "; ".join(date_fields[:5]),
                "label_field": "; ".join(label_fields[:5])
            }
        except:
            pass

print(f"Loaded {len(rd_cache)} dataset definitions from ADNIMERGE2 documentation.")

# Regex patterns
ID_PATTERNS = re.compile(r"^(PATNO|PTID|RID|SUBJECT|SUBJID|PARTICIPANT_ID|SUBJECTIDENTIFIER|STUDYID|ID)$", re.I)
VISIT_PATTERNS = re.compile(r"^(EVENT_ID|VISCODE|VISCODE2|VISIT|VISITNO|VISIT_TYPE|EVENT|PAG_NAME|SES)$", re.I)
DATE_PATTERNS = re.compile(r"(DATE|INFODT|EXAMDATE|SCAN_DATE|ORIG_ENTRY|LAST_UPDATE|VISIT_DATE|YEAR)", re.I)
LABEL_PATTERNS = re.compile(r"(COHORT|DIAGNOSIS|DX|DXSUM|RESEARCH_GROUP|APPRDX|PRIMDIAG|STATUS|STAGE|GROUP)", re.I)

def get_dataset_identity(rel_path):
    parts = rel_path.split(os.sep)
    top = parts[0] if parts else ""
    rel_upper = rel_path.upper()
    if "PD_OPEN" in rel_upper:
        return "PD_OPEN"
    if "PPMI" in rel_upper:
        return "PPMI"
    if "ADNI" in rel_upper:
        return "ADNI"
    return "UNKNOWN"

# Cache for DICOM series folders to avoid redundant header parsing across thousands of slices
dcm_folder_cache = {}

print("Scanning all files recursively across F:\\NADI-Q data...")
t0 = time.time()
inventory = []
file_count = 0

for root, dirs, files in os.walk(base_dir):
    rel_root = os.path.relpath(root, base_dir)
    for fname in files:
        file_count += 1
        full_path = os.path.join(root, fname)
        rel_path = os.path.relpath(full_path, base_dir)
        ext = os.path.splitext(fname)[1].lower()
        sz = os.path.getsize(full_path)
        ds_id = get_dataset_identity(rel_path)
        
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
                # Peek top row for delimiter & columns
                with open(full_path, "r", encoding="utf-8", errors="ignore") as fh:
                    first_line = fh.readline()
                delimiter = "\t" if ("\t" in first_line and first_line.count("\t") > first_line.count(",")) else ","
                
                df_sample = pd.read_csv(full_path, nrows=1, sep=delimiter, low_memory=False, encoding="utf-8", on_bad_lines="skip")
                cols = list(df_sample.columns)
                col_count = len(cols)
                schema_str = "; ".join(str(c) for c in cols[:15]) + (f"; ... (+{len(cols)-15} more)" if len(cols) > 15 else "")
                
                with open(full_path, "rb") as fh:
                    num_lines = sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1024*1024), b""))
                row_count = max(0, num_lines - 1)
                
                id_field = "; ".join([c for c in cols if ID_PATTERNS.match(str(c).strip())])
                visit_field = "; ".join([c for c in cols if VISIT_PATTERNS.match(str(c).strip())])
                date_field = "; ".join([c for c in cols if DATE_PATTERNS.search(str(c).strip())][:5])
                label_field = "; ".join([c for c in cols if LABEL_PATTERNS.search(str(c).strip())][:5])

            elif ext == ".xlsx":
                file_type = "XLSX"
                wb = openpyxl.load_workbook(full_path, read_only=True)
                sheets_info = []
                tot_rows = 0
                for sname in wb.sheetnames:
                    sh = wb[sname]
                    tot_rows += (sh.max_row or 0)
                    sheets_info.append(f"{sname} ({sh.max_row}x{sh.max_column})")
                row_count = tot_rows
                schema_str = "Sheets: " + ", ".join(sheets_info)
                notes = "Excel Curated Cut Workbook"

            elif ext == ".dcm":
                file_type = "DICOM"
                # Use directory-level cache for series metadata
                if root not in dcm_folder_cache:
                    try:
                        ds = pydicom.dcmread(full_path, stop_before_pixels=True, force=True)
                        pid = str(getattr(ds, "PatientID", ""))
                        suid = str(getattr(ds, "SeriesInstanceUID", ""))
                        sdesc = str(getattr(ds, "SeriesDescription", ""))
                        acq = str(getattr(ds, "MRAcquisitionType", ""))
                        sdate = str(getattr(ds, "StudyDate", ""))
                        dcm_folder_cache[root] = {
                            "pid": pid, "suid": suid, "sdesc": sdesc, "acq": acq, "sdate": sdate
                        }
                    except Exception as ex:
                        dcm_folder_cache[root] = {"pid": "", "suid": "", "sdesc": "", "acq": "", "sdate": "", "err": str(ex)}
                
                cached = dcm_folder_cache[root]
                pid = cached["pid"]
                sdesc = cached["sdesc"]
                acq = cached["acq"]
                schema_str = f"PID={pid}, SUID={cached['suid'][:15]}..., Desc={sdesc}, MRAcq={acq}"
                id_field = "PatientID" if pid else ""
                date_field = "StudyDate" if cached["sdate"] else ""
                notes = f"DICOM slice: {sdesc} [{acq}]"

            elif ext in [".gz", ".nii"]:
                file_type = "NIFTI"
                try:
                    img = nib.load(full_path, mmap=False)
                    hdr = img.header
                    shape = img.shape
                    schema_str = f"Shape={shape}, Dtype={hdr.get_data_dtype()}, Zooms={hdr.get_zooms()}"
                    row_count = shape[-1] if len(shape) >= 3 else 1
                    notes = "NIfTI 3D/4D Volumetric Image"
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
                    notes = "Preprocessed Numpy Tensor"
                except Exception as e:
                    notes = f"Numpy load error: {e}"

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
                notes = "R Documentation source file"

            elif ext == ".xml":
                file_type = "XML"
                with open(full_path, "r", encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read(512)
                schema_str = txt[:120].replace("\n", " ").strip()
                notes = "LONI XML Metadata stub"

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
                        notes = "Multi-line or unstructured JSON"

            elif ext == ".zip":
                file_type = "ZIP_ARCHIVE"
                try:
                    with zipfile.ZipFile(full_path, "r") as zf:
                        nl = zf.namelist()
                        row_count = len(nl)
                        schema_str = f"Contains {len(nl)} files: {', '.join(nl[:4])}"
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
            notes = f"Error during inspect: {str(e)}"

        inventory.append({
            "relative_path": rel_path,
            "file_name": fname,
            "extension": ext,
            "file_size_bytes": sz,
            "file_type": file_type,
            "record_or_row_count": row_count,
            "column_count": col_count,
            "available_schema": schema_str,
            "dataset_identity": ds_id,
            "participant_id_field": id_field,
            "visit_event_id_field": visit_field,
            "date_field": date_field,
            "label_diagnosis_field": label_field,
            "inspection_status": status,
            "notes": notes
        })

t1 = time.time()
print(f"Scanned {len(inventory)} files in {t1 - t0:.2f} seconds.")

df_inv = pd.DataFrame(inventory)
df_inv.to_csv(csv_out_path, index=False, encoding="utf-8")
print(f"COMPLETE_DIRECTORY_INVENTORY.csv successfully written to: {csv_out_path}")
print(f"File size: {os.path.getsize(csv_out_path):,} bytes")
print("\nFile types distribution:")
print(df_inv["file_type"].value_counts())
print("\nDataset identities distribution:")
print(df_inv["dataset_identity"].value_counts())

