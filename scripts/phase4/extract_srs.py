import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os

def read_docx(path):
    import zipfile
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(path) as z:
        xml_content = z.read('word/document.xml')
    tree = ET.fromstring(xml_content)
    text = []
    for elem in tree.iter():
        if elem.tag.endswith('t'):
            if elem.text:
                text.append(elem.text)
        elif elem.tag.endswith('p'):
            text.append('\n')
    return ''.join(text)

docx_path = r"C:\Users\ayush\OneDrive\Desktop\NADI-Q_Abstract_SRS_.docx"
if os.path.exists(docx_path):
    content = read_docx(docx_path)
    with open("docs/phase4/extracted_abstract_srs.txt", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved full text to docs/phase4/extracted_abstract_srs.txt ({len(content)} chars)")
    print("\n--- FIRST 2000 CHARS ---")
    print(content[:2000])
