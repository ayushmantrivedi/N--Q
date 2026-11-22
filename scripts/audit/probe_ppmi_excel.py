import openpyxl

xlsx_path = r"F:\NADI-Q data\PPMI\PPMI_Curated_Data_Cut_Public_20260810.xlsx"
wb = openpyxl.load_workbook(xlsx_path, read_only=True)
print(f"Total sheets in PPMI Curated Data Cut: {len(wb.sheetnames)}")
for sname in wb.sheetnames:
    sheet = wb[sname]
    print(f"  Sheet: {sname:35s} (max_row={sheet.max_row}, max_column={sheet.max_column})")

