import os
import re

man_dir = r"F:\NADI-Q data\ADNI DATA\ADNIMERGE2\ADNIMERGE2\man"
obs_re = re.compile(r"A data frame with (\d+) observations and (\d+) variables", re.I)
item_re = re.compile(r"\\item\s+\\strong\{([^}]+)\}\s+\\emph\{([^}]+)\}\s*(.*?)(?=(\\item|\\itemize|\\description|\\format|\\details|\\source|\\alias|\\name|\\usage|\\keyword|\\docType|\}))", re.DOTALL)

rd_cache = {}
for fname in os.listdir(man_dir):
    if not fname.endswith(".Rd"): continue
    stem = fname[:-3]
    fp = os.path.join(man_dir, fname)
    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
        txt = fh.read()
    
    m = obs_re.search(txt)
    obs = int(m.group(1)) if m else None
    vars_ = int(m.group(2)) if m else None
    
    # Extract item strongs (columns)
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

print(f"Cached metadata for {len(rd_cache)} datasets from ADNIMERGE2 man/.")
# Print sample
for k in ["DXSUM", "MMSE", "UCSFFSX7", "APOERES"]:
    print(f"{k}: obs={rd_cache[k]['obs']}, vars={rd_cache[k]['vars']}, cols={rd_cache[k]['cols'][:6]}")

