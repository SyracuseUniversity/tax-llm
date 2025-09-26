import pandas as pd
import re

# === Config ===
OCR_CSV = r'new-work/output/cleaned_classified_words.csv'
FINAL_CSV = r'new-work/output/final-table.csv'

# === Patterns ===
commodity_pattern = re.compile(r'^\d{4}\s?\d{3}$')
tariff_pattern = re.compile(r'^\d{3}$')


def extract_tariff_ranges(df):
    """Identify commodity positions and map tariff paragraph ranges."""
    # Use separated columns for commodity and tariff paragraph
    commodity_rows = []
    for idx, row in df.iterrows():
        word = str(row['Commodity Number']).strip()
        if word and word != 'nan':
            commodity_rows.append((idx, row['TopLeft_Y']))

    tariff_rows = []
    for idx, row in df.iterrows():
        para = str(row['Tariff Paragraph']).strip()
        if para and para != 'nan':
            tariff_rows.append((idx, row['TopLeft_Y'], para))

    commodity_to_tariff = {}
    commodity_idx = 0
    last_endpoint = None

    for t_idx, mid_y, para_no in tariff_rows:
        # Find the next commodity number after the last endpoint
        if last_endpoint is not None:
            while commodity_idx < len(commodity_rows) and commodity_rows[commodity_idx][1] <= last_endpoint:
                commodity_idx += 1
        if commodity_idx >= len(commodity_rows):
            break
        start_idx, start_y = commodity_rows[commodity_idx]
        diff = mid_y - start_y
        end_y = mid_y + diff
        # Assign para_no to all commodity numbers in this range
        while commodity_idx < len(commodity_rows):
            idx, y = commodity_rows[commodity_idx]
            if start_y <= y <= end_y:
                commodity_to_tariff[idx] = para_no
                commodity_idx += 1
            else:
                break
        last_endpoint = end_y
    return commodity_to_tariff


def apply_tariff_to_final_table(commodity_to_tariff):
    """Append tariff paragraph numbers to the final-table.csv under correct rows."""
    final_df = pd.read_csv(FINAL_CSV)

    # Ensure all expected columns exist with proper headers for new 6-column structure
    expected_columns = [
        'SCHEDULE A COMMODITY NUMBER',
        'COMMODITY DESCRIPTION AND ECONOMIC CLASS', 
        'UNIT OF QUANTITY',
        'RATE OF DUTY 1930',
        'RATE OF DUTY TRADE AGREEMENT',
        'TARIFF PARAGRAPH'
    ]
    
    # Add missing columns if they don't exist
    for col in expected_columns:
        if col not in final_df.columns:
            final_df[col] = ''

    # Build a mapping of normalized commodity number to TopLeft_Y from clean CSV
    ocr_df = pd.read_csv(OCR_CSV)
    def normalize_commodity(num):
        num = str(num).strip()
        if num and num != 'nan':
            # Handle decimal format like "10600.0" -> "0106 000"
            if '.' in num:
                base = num.split('.')[0]
                if len(base) <= 5:
                    # Pad with leading zero and format: "10600" -> "0106 000"
                    padded = base.zfill(5)  # "01060"
                    if len(padded) >= 4:
                        return f"0{padded[:3]} {padded[3:].ljust(3, '0')}"
            # If 7 digits, convert to 'XXXX XXX'
            elif re.fullmatch(r'\d{7}', num):
                return f"{num[:4]} {num[4:]}"
            # If 6 digits, convert to 'XXXX XX'  
            elif re.fullmatch(r'\d{6}', num):
                return f"{num[:4]} {num[4:]}"
            # If already spaced, return as is
            return num
        return ''
    commodity_y_map = {}
    for _, row in ocr_df.iterrows():
        raw_num = str(row['Commodity Number']).strip()
        norm_num = normalize_commodity(raw_num)
        if norm_num:
            commodity_y_map[norm_num] = row['TopLeft_Y']

    # Build a list of ranges: (start_y, end_y, para_no)
    ranges = []
    commodity_rows = [(str(row['Commodity Number']).strip(), row['TopLeft_Y']) for _, row in ocr_df.iterrows() if str(row['Commodity Number']).strip() and str(row['Commodity Number']).strip() != 'nan']
    tariff_rows = [(row['TopLeft_Y'], str(row['Tariff Paragraph']).strip()) for _, row in ocr_df.iterrows() if str(row['Tariff Paragraph']).strip() and str(row['Tariff Paragraph']).strip() != 'nan']
    commodity_idx = 0
    last_endpoint = None
    for mid_y, para_no in tariff_rows:
        # Find the next commodity number after the last endpoint
        if last_endpoint is not None:
            while commodity_idx < len(commodity_rows) and commodity_rows[commodity_idx][1] <= last_endpoint:
                commodity_idx += 1
        if commodity_idx >= len(commodity_rows):
            break
        _, start_y = commodity_rows[commodity_idx]
        diff = mid_y - start_y
        end_y = mid_y + diff
        ranges.append((start_y, end_y, para_no))
        last_endpoint = end_y

    # For each row in final_df, assign tariff paragraph if its normalized commodity number's Y falls in a range
    for i, row in final_df.iterrows():
        num = str(row['SCHEDULE A COMMODITY NUMBER']).strip()
        norm_num = normalize_commodity(num)
        y = commodity_y_map.get(norm_num, None)
        assigned = False
        if y is not None:
            for start_y, end_y, para_no in ranges:
                if start_y <= y <= end_y:
                    final_df.at[i, 'TARIFF PARAGRAPH'] = para_no
                    assigned = True
                    break
        if not assigned:
            final_df.at[i, 'TARIFF PARAGRAPH'] = ''

    # Reorder columns to match new 6-column structure
    final_df = final_df[expected_columns]
    
    final_df.to_csv(FINAL_CSV, index=False)
    print(f"Updated 'TARIFF PARAGRAPH' column in {FINAL_CSV}")
    print(f"File now uses new 6-column structure: {expected_columns}")


def main():
    ocr_df = pd.read_csv(OCR_CSV)
    commodity_to_tariff = extract_tariff_ranges(ocr_df)
    apply_tariff_to_final_table(commodity_to_tariff)


if __name__ == "__main__":
    main()
