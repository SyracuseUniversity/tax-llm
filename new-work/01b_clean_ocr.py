import pandas as pd
import numpy as np
import re
import logging
import os

# Configuration
INPUT_CSV = r'new-work/output/ocr_word_coords.csv'
OUTPUT_CSV = r'new-work/output/cleaned_classified_words.csv'

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------------------------------------------------------
# OCR corrections (character-level) + special-case map (whole-string fixes)
# ----------------------------------------------------------------------------
ocr_corrections = {
    'O': '0', 'o': '0',
    'l': '1', 'I': '1', 'i': '1',
    'S': '5', 's': '5',
    'G': '6', 'B': '8',
    'D': '0', 'd': '0'
}

# For very specific OCR misreads (use before generic cleanup)
special_cases = {
    '0023800d1': '0023 800',
    '265005': '0026 500',
    '269005': '0026 900',
}

# ----------------------------------------------------------------------------
# 1️⃣ Load & preprocess
# ----------------------------------------------------------------------------
def load_and_preprocess(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(df)} rows from {input_csv}")

    # Remove header/footer
    min_y, max_y = df['TopLeft_Y'].min(), df['TopLeft_Y'].max()
    header_limit, footer_limit = min_y + 200, max_y - 100
    df = df[(df['TopLeft_Y'] > header_limit) & (df['TopLeft_Y'] < footer_limit)]
    logging.info(f"Trimmed rows to {len(df)} (removed headers/footers)")
    return df

# ----------------------------------------------------------------------------
# 2️⃣ Analyze coordinate zones
# ----------------------------------------------------------------------------
def analyze_and_refine_zones(df):
    zones_28 = {
        # 'commodity': {'min_x': 220, 'max_x': 320},
        # 'description': {'min_x': 450, 'max_x': 1250},
        # 'unit': {'min_x': 1300, 'max_x': 1320},
        # 'rate_1930': {'min_x': 1450, 'max_x': 1690},
        # 'rate_trade': {'min_x': 1691, 'max_x': 1730},
        # 'tariff': {'min_x': 2140, 'max_x': 2190}
    },
    zones_29 = {
        'commodity': {'min_x': 208, 'max_x': 213},
        'description': {'min_x': 448, 'max_x': 1205},
        'unit': {'min_x': 1273, 'max_x': 1282},
        'rate_1930': {'min_x': 1423, 'max_x': 1607},
        'rate_trade': {'min_x': 1661, 'max_x': 1763},
        'tariff': {'min_x': 2125, 'max_x': 2151}
    }
    return zones_29, zones_28

# ----------------------------------------------------------------------------
# 3️⃣ Classification function
# ----------------------------------------------------------------------------
def classify_word_by_coordinates(word: str, x_coord: float, zones: dict):
    word = str(word).strip()

    commodity_num = description = unit = rate_1930 = rate_trade = tariff_para = None

    # Strong pattern detection
    if re.match(r'^\d{7}$', word) or re.match(r'^00\d{2}\s?\d{3}$', word):
        commodity_num = word
    elif re.match(r'^(70[0-9]|71[0-9]|1558|706|702|703|704|711|712)$', word):
        tariff_para = word
    elif re.match(r'^(No|Lb|each|ea|N0|1b|Ib)\.?$', word, re.IGNORECASE):
        unit = word
    else:
        if zones['commodity']['min_x'] <= x_coord <= zones['commodity']['max_x']:
            description = word
        elif zones['description']['min_x'] <= x_coord <= zones['description']['max_x']:
            description = word
        elif zones['unit']['min_x'] <= x_coord <= zones['unit']['max_x']:
            unit = word
        elif zones['rate_1930']['min_x'] <= x_coord <= zones['rate_1930']['max_x']:
            rate_1930 = word
        elif zones['rate_trade']['min_x'] <= x_coord <= zones['rate_trade']['max_x']:
            rate_trade = word
        elif zones['tariff']['min_x'] <= x_coord <= zones['tariff']['max_x']:
            tariff_para = word
        else:
            description = word

    return commodity_num, description, unit, rate_1930, rate_trade, tariff_para

# ----------------------------------------------------------------------------
# 4️⃣ Clean commodity numbers
# ----------------------------------------------------------------------------
def clean_commodity_number(raw_number):
    """Clean and standardize commodity numbers to 'XXXX XXX' format."""
    if pd.isna(raw_number):
        return None

    text = str(raw_number).strip()

    # Handle full-string special cases before regex
    if text in special_cases:
        return special_cases[text]

    # Apply OCR corrections
    for wrong, correct in ocr_corrections.items():
        text = re.sub(wrong, correct, text, flags=re.IGNORECASE)

    # Extract digits only
    digits_only = re.sub(r'\D', '', text)

    if len(digits_only) < 4:
        return None

    # Normalize length
    digits_only = digits_only[-7:].rjust(7, '0')
    return f"{digits_only[:4]} {digits_only[4:]}"


# ----------------------------------------------------------------------------
# 6️⃣ Main
# ----------------------------------------------------------------------------
def main():
    df = load_and_preprocess(INPUT_CSV)
    zones = analyze_and_refine_zones(df)

    df[['Commodity Number', 'Commodity Description', 'Unit of Quantity',
        'Rate of Duty 1930 Tariff Act', 'Rate of Duty Trade Agreement', 'Tariff Paragraph']] = df.apply(
        lambda row: pd.Series(classify_word_by_coordinates(row['Word'], row['TopLeft_X'], zones)), axis=1
    )

    # Clean commodity numbers
    df['Commodity Number'] = df['Commodity Number'].apply(clean_commodity_number)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Saved classified words → {OUTPUT_CSV}")


    logging.info("\n✅ Processing complete!")
    logging.info(f"Valid commodity numbers found: {df['Commodity Number'].notna().sum()}")
    logging.info(f"Sample: {sorted(df['Commodity Number'].dropna().unique())[:20]}")


if __name__ == "__main__":
    word_df = main()