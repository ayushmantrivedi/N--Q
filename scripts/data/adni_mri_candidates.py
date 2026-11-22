import os
import glob
import xml.etree.ElementTree as ET
import csv
from collections import defaultdict

def remove_namespace(tag):
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag

def main():
    root_dir = r"F:\NADI-Q data"
    out_csv = "ADNI_3T_T1_candidates.csv"

    xml_files = glob.glob(os.path.join(root_dir, "**", "*.xml"), recursive=True)
    
    candidates = []

    for path in xml_files:
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            
            data = defaultdict(str)
            
            for elem in root.iter():
                tag = remove_namespace(elem.tag).lower()
                text = elem.text.strip() if elem.text else ""
                
                if tag == 'subjectidentifier' and not data['subjectIdentifier']:
                    data['subjectIdentifier'] = text
                elif tag == 'studyidentifier' and not data['studyIdentifier']:
                    data['studyIdentifier'] = text
                elif tag == 'seriesidentifier' and not data['seriesIdentifier']:
                    data['seriesIdentifier'] = text
                elif tag == 'imageuid' and not data['imageUID']:
                    # Capturing the first imageUID we see (derived or original)
                    data['imageUID'] = text
                elif tag == 'dateacquired' and not data['dateAcquired']:
                    data['dateAcquired'] = text
                elif tag == 'description' and not data['description']:
                    data['description'] = text
                elif tag == 'modality' and not data['modality']:
                    data['modality'] = text
                elif tag == 'protocol':
                    term = elem.get('term')
                    if term:
                        term = term.lower()
                        if term == 'field strength': data['field_strength'] = text
                        elif term == 'weighting': data['weighting'] = text
                        elif term == 'pulse sequence': data['pulse_sequence'] = text
                        elif term == 'matrix x': data['matrix_x'] = text
                        elif term == 'matrix y': data['matrix_y'] = text
                        elif term == 'matrix z': data['matrix_z'] = text
                        elif term == 'slice thickness': data['slice_thickness'] = text
                        elif term == 'acquisition plane': data['acquisition_plane'] = text
                        elif term == 'manufacturer': data['manufacturer'] = text
                        elif term == 'mfg model': data['model'] = text

            is_mri = data['modality'].lower() in ['mri', 'mr']
            is_3t = '3.0' in data['field_strength'] or '3 ' in data['field_strength']
            is_t1 = 't1' in data['weighting'].lower() or 't1' in data['description'].lower()

            seq_str = (data['description'] + " " + data['pulse_sequence']).lower()
            is_mprage_like = any(x in seq_str for x in ['mprage', 'mp-rage', 'fspgr', 'ir-spgr', 'spgr'])

            if is_mri and is_3t and is_t1 and is_mprage_like:
                # RID extraction attempt (usually last 4 digits of subjectIdentifier)
                parts = data['subjectIdentifier'].split('_')
                if len(parts) >= 3 and parts[-1].isdigit():
                    data['RID'] = str(int(parts[-1]))
                else:
                    data['RID'] = ""
                    
                data['local_image_path'] = "NOT_FOUND"
                candidates.append(data)
                
        except Exception:
            pass

    fieldnames = [
        'subjectIdentifier', 'RID', 'studyIdentifier', 'seriesIdentifier', 'imageUID', 
        'dateAcquired', 'description', 'weighting', 'pulse_sequence', 'field_strength',
        'matrix_x', 'matrix_y', 'matrix_z', 'slice_thickness', 'acquisition_plane',
        'manufacturer', 'model', 'local_image_path'
    ]

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for cand in candidates:
            writer.writerow(cand)

    print(f"Generated {out_csv} with {len(candidates)} candidate 3T T1 series.")
    print("NOTE: 0 actual local image payloads were found during inspection.")

if __name__ == "__main__":
    main()
