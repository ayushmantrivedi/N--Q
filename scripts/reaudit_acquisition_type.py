import os
import pydicom
import pandas as pd
from collections import defaultdict

def main():
    base_dir = r"F:\NADI-Q data\PPMI\MRI image download_dataset\PPMI"
    target_patnos = [
        "220494", "229409", "278950", "311004", "315185",
        "149006", "151050", "170472", "183263", "222893"
    ]
    
    # Also load Xing table to compare
    xing_path = r"F:\NADI-Q data\PPMI\MRI_Acquisition\Xing_Core_Lab_-_MRI_acquisition_metadata_18Sep2026.csv"
    df_xing = pd.read_csv(xing_path) if os.path.exists(xing_path) else pd.DataFrame()

    print("=" * 100)
    print("TARGETED RE-AUDIT OF MRI ACQUISITION TYPE (MRAcquisitionType, ScanningSequence, SequenceVariant)")
    print("=" * 100)

    results = []

    for patno in target_patnos:
        pat_dir = os.path.join(base_dir, patno)
        print(f"\n=================== PATNO: {patno} ===================")
        if not os.path.exists(pat_dir):
            print(f"Directory {pat_dir} DOES NOT EXIST!")
            continue

        dcm_files = []
        for root, dirs, files in os.walk(pat_dir):
            for f in files:
                if f.lower().endswith(".dcm"):
                    dcm_files.append(os.path.join(root, f))

        print(f"Total DICOM files: {len(dcm_files)}")
        if not dcm_files:
            continue

        # Group by SUID
        by_suid = defaultdict(list)
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
                suid = getattr(ds, "SeriesInstanceUID", "UNKNOWN")
                by_suid[suid].append((f, ds))
            except Exception as e:
                print(f"Error reading {f}: {e}")

        # Check Xing metadata for this PATNO
        p_int = int(patno)
        xing_rows = df_xing[df_xing["PATNO"] == p_int] if not df_xing.empty else pd.DataFrame()
        print(f"Xing Core Lab records ({len(xing_rows)} rows):")
        for _, xr in xing_rows.iterrows():
            print(f"  EVENT_ID={xr.get('EVENT_ID')}, MRI_DATE={xr.get('MRI_DATE')}, PROTOCOL={xr.get('PROTOCOL')}, "
                  f"ACQ_DATE={xr.get('ACQ_DATE')}, SEQ_TYPE={xr.get('SEQ_TYPE')}, SER_DESC={xr.get('SER_DESC')}")

        for s_idx, (suid, items) in enumerate(by_suid.items()):
            ds0 = items[0][1]
            mr_acq_type = getattr(ds0, "MRAcquisitionType", "NOT_SPECIFIED")
            scanning_seq = getattr(ds0, "ScanningSequence", "NOT_SPECIFIED")
            seq_variant = getattr(ds0, "SequenceVariant", "NOT_SPECIFIED")
            scan_options = getattr(ds0, "ScanOptions", "NOT_SPECIFIED")
            seq_name = getattr(ds0, "SequenceName", "NOT_SPECIFIED")
            ser_desc = getattr(ds0, "SeriesDescription", "NOT_SPECIFIED")
            protocol = getattr(ds0, "ProtocolName", "NOT_SPECIFIED")
            image_type = getattr(ds0, "ImageType", "NOT_SPECIFIED")
            slice_thick = getattr(ds0, "SliceThickness", "NOT_SPECIFIED")
            spacing_between = getattr(ds0, "SpacingBetweenSlices", "NOT_SPECIFIED")
            px_spacing = getattr(ds0, "PixelSpacing", "NOT_SPECIFIED")
            tr = getattr(ds0, "RepetitionTime", "NOT_SPECIFIED")
            te = getattr(ds0, "EchoTime", "NOT_SPECIFIED")
            ti = getattr(ds0, "InversionTime", "NOT_SPECIFIED")
            flip_angle = getattr(ds0, "FlipAngle", "NOT_SPECIFIED")
            manufacturer = getattr(ds0, "Manufacturer", "NOT_SPECIFIED")
            model = getattr(ds0, "ManufacturerModelName", "NOT_SPECIFIED")
            field_strength = getattr(ds0, "MagneticFieldStrength", "NOT_SPECIFIED")
            rows = getattr(ds0, "Rows", "NOT_SPECIFIED")
            cols = getattr(ds0, "Columns", "NOT_SPECIFIED")
            study_date = getattr(ds0, "StudyDate", "NOT_SPECIFIED")
            series_date = getattr(ds0, "SeriesDate", "NOT_SPECIFIED")

            print(f"\n  Series #{s_idx+1} [SUID: {suid}] (Slices: {len(items)})")
            print(f"    MRAcquisitionType:       {mr_acq_type}")
            print(f"    ScanningSequence:        {scanning_seq}")
            print(f"    SequenceVariant:         {seq_variant}")
            print(f"    ScanOptions:             {scan_options}")
            print(f"    SequenceName:            {seq_name}")
            print(f"    SeriesDescription:       {ser_desc}")
            print(f"    ProtocolName:            {protocol}")
            print(f"    ImageType:               {image_type}")
            print(f"    SliceThickness:          {slice_thick} mm")
            print(f"    SpacingBetweenSlices:    {spacing_between} mm")
            print(f"    PixelSpacing:            {px_spacing} mm")
            print(f"    Dimensions:              {rows} x {cols} x {len(items)}")
            print(f"    TR: {tr} ms, TE: {te} ms, TI: {ti} ms, FA: {flip_angle} deg")
            print(f"    Scanner:                 {manufacturer} {model} ({field_strength}T)")
            print(f"    StudyDate:               {study_date}")

            # Inspect consistency across slices for MRAcquisitionType
            acq_types_in_series = set(getattr(x[1], "MRAcquisitionType", "N/A") for x in items)
            print(f"    MRAcquisitionType across all {len(items)} slices: {acq_types_in_series}")

            results.append({
                "PATNO": patno,
                "SUID": suid,
                "slice_count": len(items),
                "mr_acq_type": str(mr_acq_type),
                "scanning_seq": str(scanning_seq),
                "seq_variant": str(seq_variant),
                "seq_name": str(seq_name),
                "ser_desc": str(ser_desc),
                "image_type": str(image_type),
                "slice_thick": str(slice_thick),
                "spacing_between": str(spacing_between),
                "pixel_spacing": str(px_spacing),
                "tr": str(tr),
                "te": str(te),
                "ti": str(ti),
                "flip_angle": str(flip_angle),
                "scanner": f"{manufacturer} {model} {field_strength}T",
                "study_date": str(study_date)
            })

    df_res = pd.DataFrame(results)
    df_res.to_csv(r"F:\NADI-Q data\PPMI\reports\mri_acquisition_type_reaudit_table.csv", index=False)
    print("\n" + "=" * 100)
    print("SUMMARY TABLE OF ACQUISITION TYPES:")
    print("=" * 100)
    print(df_res[["PATNO", "mr_acq_type", "ser_desc", "slice_thick", "spacing_between", "slice_count"]])

if __name__ == "__main__":
    main()

