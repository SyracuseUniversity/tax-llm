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
pdf_path = r'D:/Users/dhair/Downloads/Compressed/tarrf-ocr/Input/1950 Schedule A (scanned by ILL, Reed College).pdf'
output_csv_results = r'old-work/output/inference_results.csv'
output_word_coords = r'old-work/output/ocr_word_coords.csv'
start_page = 38
end_page = 38  # Pages with the table

###############################################
# Logging Setup
###############################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###############################################
# Regex Patterns
###############################################
commodity_regex = re.compile(r'^\d{4}\s*\d{3,4}(?:\s*\w*)?$')  # e.g. "0010 600", "0010 700 a"
tariff_regex = re.compile(r'^\(?\d{1,4}\)?$')                  # e.g. "701", "(2)"

###############################################
# Unwanted Keywords (case-insensitive)
###############################################
skip_keywords = [
    "GROUP 00", "ANIMALS AND ANIMAL PRODUCTS", "RATE OR DUTY",
    "SCHEDULE A", "UNIT OR", "TARIPE", "ECONOMIC CLASS", "ILIIOR",
    "COMMODIT", "QUANTITY", "PARAGRAPH", "1930 TARIFF ACT", "TRADE AGREEMENT",
    "(EXCEPT AS NOTED)", "BREEDING", "MEAT PRODUCTS"
]

def should_skip_line(line: str) -> bool:
    """Check if a line contains any unwanted keywords."""
    upper_line = line.upper()
    return any(keyword in upper_line for keyword in skip_keywords)

###############################################
# OCR Extraction: Raw Text
###############################################
def extract_ocr_text(pdf_path: str, start_page: int, end_page: int, ocr: PaddleOCR) -> List[str]:
    """
    Extract text lines from PDF using OCR.
    """
    logging.info(f"Extracting text from PDF pages {start_page}-{end_page}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Failed to open PDF: {e}")
        return []
    extracted_lines = []

    for page_number in range(start_page - 1, end_page):
        if page_number >= len(doc):
            logging.warning(f"Page {page_number + 1} does not exist. Skipping.")
            continue
        logging.info(f"Processing Page {page_number + 1}...")
        page = doc[page_number]
        pix = page.get_pixmap(dpi=300)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_np = np.array(image)
        image.show() #check image

        results = ocr.predict(image_np)
        for res in results:
          if 'rec_texts' in res:
            extracted_lines.extend(res['rec_texts'])

        print("Extracted lines:", extracted_lines)


    doc.close()
    logging.info("OCR text extraction completed.")
    return extracted_lines

###############################################
# OCR Words with Coordinates
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
# Build Rows: Minimal Logic + Paragraph Inheritance
###############################################
def build_rows(ocr_lines: list) -> list:
    """
    Build structured rows from OCR lines using regex and inheritance logic.
    """
    rows = []
    current_row = {
        "Schedule_A_Commodity_Number": "",
        "Commodity_Description": "",
        "Tariff_Paragraph": ""
    }
    last_paragraph = ""  # Store last encountered paragraph for inheritance

    def finalize_row():
        if (current_row["Schedule_A_Commodity_Number"] or
            current_row["Commodity_Description"] or
            current_row["Tariff_Paragraph"]):
            if not current_row["Tariff_Paragraph"].strip() and last_paragraph:
                current_row["Tariff_Paragraph"] = last_paragraph
            rows.append(current_row.copy())

    for line in ocr_lines:
        line = line.strip()
        if not line or should_skip_line(line):
            continue

        if commodity_regex.match(line):
            finalize_row()
            current_row = {
                "Schedule_A_Commodity_Number": line,
                "Commodity_Description": "",
                "Tariff_Paragraph": ""
            }
            continue

        if tariff_regex.match(line):
            current_row["Tariff_Paragraph"] = line
            last_paragraph = line
            continue

        if current_row["Commodity_Description"]:
            current_row["Commodity_Description"] += " " + line
        else:
            current_row["Commodity_Description"] = line

    finalize_row()
    return rows

def save_3col_results(ocr_lines: list, output_csv: str = output_csv_results) -> None:
    """
    Save structured OCR results to CSV.
    """
    table_rows = build_rows(ocr_lines)
    df = pd.DataFrame(table_rows, columns=[
        "Schedule_A_Commodity_Number",
        "Commodity_Description",
        "Tariff_Paragraph"
    ])
    df.to_csv(output_csv, index=False)
    logging.info(f"3-column CSV saved to: {output_csv}")

###############################################
# Main
###############################################
def main():
    # ocr = PaddleOCR(use_angle_cls=True, lang="en")
    ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False) #new ocr model


    ocr_texts = extract_ocr_text(pdf_path, start_page, end_page, ocr)
    save_3col_results(ocr_texts)
    extract_ocr_words_with_coords(pdf_path, start_page, end_page, ocr)

if __name__ == "__main__":
    main()