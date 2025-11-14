import pandas as pd
import re
import logging
import os

# =========================
# Configuration
# =========================
INPUT_CSV = 'old-work/output/ocr_word_coords.csv'
OUTPUT_CSV = 'old-work/output/cleaned_classified_words.csv'
DROP_ROWS_BEFORE = 14  # Number of initial rows to drop (start at first commodity)
X_MIN, X_MAX = 1305, 2150  # X coordinate range to filter out unwanted columns

# =========================
# Logging Setup
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =========================
# Helper Functions
# =========================

def load_and_preprocess(input_csv: str) -> pd.DataFrame:
    """
    Load the OCR CSV and perform initial cleaning:
    - Drop first N rows (headers/irrelevant)
    - Remove rows with X coordinate in unwanted range
    """
    df = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(df)} rows from {input_csv}")
    df = df.iloc[DROP_ROWS_BEFORE:].reset_index(drop=True)
    df = df[~((df['TopLeft_X'] > X_MIN) & (df['TopLeft_X'] < X_MAX))].reset_index(drop=True)
    logging.info(f"After dropping rows and filtering X, {len(df)} rows remain.")
    return df

def classify_word(word: str):
    """
    Classify a word as Commodity Number, Description, or Tariff Paragraph.
    Returns a tuple: (Commodity Number, Description, Tariff Paragraph)
    """
    word = str(word).strip()
    # Commodity Number: e.g., "0010 600"
    if re.match(r'^(00\d{2} \d{3})', word):
        return word, None, None
    # Tariff Paragraph: 3+ digits, not a commodity number
    elif re.fullmatch(r'.\d{3}.*', word):
        return None, None, word
    # Otherwise, it's a description
    else:
        return None, word, None

def move_to_tariff_paragraph(desc):
    """
    If a description is actually a short tariff code (e.g., '701A'), move it to Tariff Paragraph.
    """
    if pd.isna(desc):
        return None, None
    cleaned = re.sub(r'[^\w\s]', '', desc)  # Remove special characters
    match = re.fullmatch(r'(\d{3})([A-Za-z]{1,3})?', cleaned)
    if match:
        return None, match.group(0)
    else:
        return desc, None

def is_digits_or_negative(val) -> bool:
    """
    Returns True if the value is only digits (positive or negative), else False.
    """
    if pd.isna(val):
        return False
    return re.fullmatch(r'-?\d+', str(val)) is not None

def summarize(df: pd.DataFrame):
    """
    Print summary statistics for classified columns.
    """
    logging.info(f"Total rows: {len(df)}")
    logging.info(f"Rows with Commodity Number: {df['Commodity Number'].notna().sum()}")
    logging.info(f"Rows with Description: {df['Commodity Description'].notna().sum()}")
    logging.info(f"Rows with Tariff Paragraph: {df['Tariff Paragraph'].notna().sum()}")

# =========================
# Main Processing Function
# =========================

def main():
    # Load and preprocess
    df = load_and_preprocess(INPUT_CSV)

    # Classify each word
    df[['Commodity Number', 'Commodity Description', 'Tariff Paragraph']] = df['Word'].apply(
        lambda w: pd.Series(classify_word(w))
    )
    df = df.drop(['Word'], axis=1)

    # Move short codes from Description to Tariff Paragraph
    df[['Commodity Description', 'Moved_To_Tariff']] = df['Commodity Description'].apply(
        lambda x: pd.Series(move_to_tariff_paragraph(x))
    )
    df['Tariff Paragraph'] = df['Tariff Paragraph'].combine_first(df['Moved_To_Tariff'])
    df.drop(columns=['Moved_To_Tariff'], inplace=True)

    # Remove rows where Description is only digits or negative numbers
    df = df[~df['Commodity Description'].apply(is_digits_or_negative)].reset_index(drop=True)

    # Save cleaned output
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Saved cleaned and classified data to {OUTPUT_CSV}")

    # Show summary and preview
    summarize(df)
    logging.info("\nPreview:\n" + str(df[['Commodity Number', 'Commodity Description', 'Tariff Paragraph']].head()))

# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    main()