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
        image.show() #check image

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
    
def main():
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

if __name__ == "__main__":
    main()