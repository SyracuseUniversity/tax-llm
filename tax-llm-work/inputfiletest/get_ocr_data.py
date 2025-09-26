#python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
#python -m pip install paddleocr --quiet
#python -m pip install pymupdf --quiet

import os
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF for PDF processing
from paddleocr import PaddleOCR
import numpy as np
import re
import logging
from typing import List

###############################################
# Configuration
###############################################
pdf_path = r'1950 Schedule A (scanned by ILL, Reed College).pdf'
output_word_coords = r'new-work/output/ocr_word_coords.csv'
output_cleaned_csv = r'new-work/output/cleaned_classified_words.csv'
start_page = 28
end_page = 28  # Pages with the table

###############################################
# Logging Setup
###############################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###############################################
# OCR Extraction: Extract Words with Coordinates (Original Style)
###############################################
def extract_ocr_words_with_coords(pdf_path: str, start_page: int, end_page: int, ocr: PaddleOCR, output_csv: str = output_word_coords) -> None:
    """
    Extract words and their coordinates from PDF using OCR and save to CSV.
    """
    logging.info(f"Extracting words with coordinates from PDF pages {start_page}-{end_page}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Failed to open PDF: {e}")
        return

    extracted_data = []

    for page_number in range(start_page - 1, end_page):
        if page_number >= len(doc):
            logging.warning(f"Page {page_number + 1} does not exist. Skipping.")
            continue

        logging.info(f"Processing Page {page_number + 1}...")
        page = doc[page_number]
        pix = page.get_pixmap(dpi=300)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_np = np.array(image)

        # Updated: Use structure like rec_texts, rec_polys, rec_scores
        results = ocr.predict(image_np)[0]  # Assuming single image result

        rec_texts = results.get("rec_texts", [])
        rec_polys = results.get("rec_polys", [])
        rec_scores = results.get("rec_scores", [])

        for i, word in enumerate(rec_texts):
            poly = rec_polys[i] if i < len(rec_polys) else [[None, None]] * 4
            score = rec_scores[i] if i < len(rec_scores) else None

            row = {
                "Word": word,
                "Confidence": score,
                "TopLeft_X": poly[0][0],
                "TopLeft_Y": poly[0][1],
                "TopRight_X": poly[1][0],
                "TopRight_Y": poly[1][1],
                "BottomRight_X": poly[2][0],
                "BottomRight_Y": poly[2][1],
                "BottomLeft_X": poly[3][0],
                "BottomLeft_Y": poly[3][1],
                "Page": page_number + 1
            }
            extracted_data.append(row)

    headers = [
        "Word", "Confidence",
        "TopLeft_X", "TopLeft_Y",
        "TopRight_X", "TopRight_Y",
        "BottomRight_X", "BottomRight_Y",
        "BottomLeft_X", "BottomLeft_Y",
        "Page"
    ]

    df = pd.DataFrame(extracted_data, columns=headers)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logging.info(f"Word-coordinate CSV saved to: {output_csv}")

###############################################
# Pattern-based Classification Functions
###############################################

def classify_by_pattern(word: str) -> str:
    """
    Classify word into 3 categories: commodity_number, tariff_paragraph, or description.
    Returns: 'commodity_number', 'tariff_paragraph', 'description'
    """
    if pd.isna(word) or not word:
        return 'description'
    
    word = str(word).strip()
    
    # Commodity Number patterns: 0291 000, 0293 100, etc.
    if re.match(r'^\d{4}\s*\d{3}$', word):
        return 'commodity_number'
    
    # Tariff Paragraph patterns: 1530(b), 1765, etc.
    if re.match(r'^\d{3,4}(\([a-zA-Z]\))?$', word):
        return 'tariff_paragraph'
    
    # Everything else is description (includes units, rates, trade agreements, etc.)
    return 'description'

def group_words_by_rows(df: pd.DataFrame, y_threshold: float = 25) -> List[List[int]]:
    """
    Group words into rows based on Y-coordinates proximity.
    Returns list of lists containing indices for each row.
    """
    df_sorted = df.sort_values('TopLeft_Y').reset_index(drop=True)
    rows = []
    current_row = [0]
    current_y = df_sorted.iloc[0]['TopLeft_Y']
    
    for i in range(1, len(df_sorted)):
        y_coord = df_sorted.iloc[i]['TopLeft_Y']
        if abs(y_coord - current_y) <= y_threshold:
            current_row.append(i)
        else:
            rows.append(current_row)
            current_row = [i]
            current_y = y_coord
    
    if current_row:
        rows.append(current_row)
    
    # Convert back to original indices
    original_rows = []
    for row in rows:
        original_indices = [df_sorted.index[i] for i in row]
        original_rows.append(original_indices)
    
    return original_rows

def classify_row_context(row_words: List[str], row_patterns: List[str]) -> dict:
    """
    Refine classification based on row context.
    If we find a commodity number in a row, nearby words are likely descriptions.
    """
    refined_classification = {}
    
    # Check if row has commodity number
    has_commodity = any(pattern == 'commodity_number' for pattern in row_patterns)
    
    for i, (word, pattern) in enumerate(zip(row_words, row_patterns)):
        if pattern == 'description' and has_commodity:
            # Keep as description if in same row as commodity number
            refined_classification[word] = 'description'
        else:
            refined_classification[word] = pattern
    
    return refined_classification

###############################################
# OCR Cleaning: Clean and Classify Words
###############################################

def clean_ocr_words_with_coords(input_csv: str, output_csv: str) -> None:
    """
    Clean and classify OCR words using pattern-based classification.
    Keeps all coordinates and confidence information.
    Skips all rows before the first commodity number.
    """
    logging.info(f"Loading OCR data from {input_csv}...")
    
    try:
        df = pd.read_csv(input_csv)
        logging.info(f"Loaded {len(df)} words from OCR data")
    except Exception as e:
        logging.error(f"Failed to load CSV: {e}")
        return
    
    # Find first commodity number and skip rows before it
    first_commodity_idx = None
    for idx, word in enumerate(df['Word']):
        if pd.notna(word) and re.match(r'^\d{4}\s*\d{3}$', str(word).strip()):
            first_commodity_idx = idx
            break
    
    if first_commodity_idx is not None:
        df = df.iloc[first_commodity_idx:].reset_index(drop=True)
        logging.info(f"Found first commodity number at index {first_commodity_idx}")
        logging.info(f"After skipping header rows: {len(df)} words remain")
    else:
        logging.warning("No commodity numbers found in the data!")
    
    # Initialize classification columns (only 3 categories)
    df['Commodity_Number'] = None
    df['Description'] = None
    df['Tariff_Paragraph'] = None
    df['Pattern_Classification'] = None
    
    # First pass: Pattern-based classification
    logging.info("Performing pattern-based classification...")
    df['Pattern_Classification'] = df['Word'].apply(classify_by_pattern)
    
    # Second pass: Row-based context refinement
    logging.info("Refining classification with row context...")
    word_rows = group_words_by_rows(df)
    
    for row_indices in word_rows:
        if len(row_indices) <= 1:
            continue
            
        row_words = [df.iloc[i]['Word'] for i in row_indices]
        row_patterns = [df.iloc[i]['Pattern_Classification'] for i in row_indices]
        
        refined = classify_row_context(row_words, row_patterns)
        
        # Update refined classifications
        for i in row_indices:
            word = df.iloc[i]['Word']
            if word in refined:
                df.at[i, 'Pattern_Classification'] = refined[word]
    
    # Third pass: Assign to specific columns (only 3 categories)
    logging.info("Assigning words to specific columns...")
    for idx, row in df.iterrows():
        word = row['Word']
        classification = row['Pattern_Classification']
        
        if classification == 'commodity_number':
            df.at[idx, 'Commodity_Number'] = word
        elif classification == 'tariff_paragraph':
            df.at[idx, 'Tariff_Paragraph'] = word
        else:  # Everything else goes to description
            df.at[idx, 'Description'] = word
    
    # Keep only new classified columns first, then coordinates (no Word or Pattern_Classification)
    output_columns = [
        'Commodity_Number', 'Description', 'Tariff_Paragraph',
        'Confidence', 
        'TopLeft_X', 'TopLeft_Y', 'TopRight_X', 'TopRight_Y',
        'BottomRight_X', 'BottomRight_Y', 'BottomLeft_X', 'BottomLeft_Y',
        'Page'
    ]
    
    df_output = df[output_columns].copy()
    
    # Save the classified data
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_output.to_csv(output_csv, index=False)
    logging.info(f"Cleaned and classified data saved to: {output_csv}")
    
    # Print summary statistics (simplified to 3 categories)
    logging.info("\n=== Classification Summary ===")
    for col in ['Commodity_Number', 'Description', 'Tariff_Paragraph']:
        count = df_output[col].notna().sum()
        logging.info(f"{col}: {count} words")
    
    # Show sample of classified data
    logging.info("\n=== Sample Classifications ===")
    sample_data = df_output[df_output['Commodity_Number'].notna()].head(3)
    if len(sample_data) > 0:
        for idx, row in sample_data.iterrows():
            logging.info(f"Row {idx}: Commodity={row['Commodity_Number']}")

def main():
    # ocr = PaddleOCR(use_angle_cls=True, lang="en")
    ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False) #new ocr model

    # Step 1: Extract OCR words with coordinates
    print("Step 1: Extracting OCR words with coordinates...")
    extract_ocr_words_with_coords(pdf_path, start_page, end_page, ocr)
    print(f"Raw OCR data saved to: {output_word_coords}")
    
    # Step 2: Clean and classify the OCR words
    print("\nStep 2: Cleaning and classifying OCR words...")
    clean_ocr_words_with_coords(output_word_coords, output_cleaned_csv)
    print(f"Cleaned classified data saved to: {output_cleaned_csv}")
    
    print("\nProcessing completed!")
    print(f"Final output: {output_cleaned_csv}")

if __name__ == "__main__":
    main()