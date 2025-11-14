#!/usr/bin/env python3
#python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
#python -m pip install paddleocr --quiet
#python -m pip install pymupdf --quiet
#python -m pip install opencv-python --quiet
#python -m pip install ultralytics --quiet  # For YOLOv8

import os
import pandas as pd
import json
from PIL import Image
import fitz  # PyMuPDF for PDF processing
from paddleocr import PaddleOCR

import numpy as np
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
import cv2

####
# Configuration
####
pdf_path = r'1950 Schedule A (scanned by ILL, Reed College).pdf'
output_word_coords = r'new-work/other/approach/output/ocr_word_coords.csv'
output_table_csv = r'new-work/other/approach/output/extracted_table_page_{page}.csv'
output_cells_json = r'new-work/other/approach/output/table_cells_page_{page}.json'
output_cells_image = r'new-work/other/approach/output/cells_visualization_page_{page}.png'
start_page = 29
end_page = 29

####
# Logging Setup
####
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

####
# Predefined Columns
####
COLUMN_DEFINITIONS = {
    0: {'name': 'Schedule_A_Number', 'x_range': (100, 280)},
    1: {'name': 'Commodity_Description', 'x_range': (280, 650)},
    2: {'name': 'Unit_of_Quantity', 'x_range': (650, 750)},
    3: {'name': 'Rate_1930_Tariff', 'x_range': (750, 1100)},
    4: {'name': 'Rate_Trade_Agreement', 'x_range': (1100, 1450)},
    5: {'name': 'Tariff_Paragraph', 'x_range': (1450, 1850)}
}

####
# Helper Functions
####

def assign_word_to_column(word_x: float) -> Tuple[int, str]:
    for col_idx, col_def in COLUMN_DEFINITIONS.items():
        x_min, x_max = col_def['x_range']
        if x_min <= word_x <= x_max:
            return col_idx, col_def['name']
    # fallback to nearest
    min_dist = float('inf')
    nearest_col = 0
    for col_idx, col_def in COLUMN_DEFINITIONS.items():
        x_min, x_max = col_def['x_range']
        center = (x_min + x_max) / 2
        dist = abs(word_x - center)
        if dist < min_dist:
            min_dist = dist
            nearest_col = col_idx
    return nearest_col, COLUMN_DEFINITIONS[nearest_col]['name']

def assign_word_to_row(word_y: float, row_boundaries: List[float]) -> int:
    if not row_boundaries:
        return 0
    min_dist = float('inf')
    nearest_row = 0
    for idx, row_y in enumerate(row_boundaries):
        dist = abs(word_y - row_y)
        if dist < min_dist:
            min_dist = dist
            nearest_row = idx
    return nearest_row

def detect_table_structure_from_words(words_df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    if words_df.empty:
        return [], []
    y_coords = [(row['TopLeft_Y'] + row['BottomLeft_Y']) / 2 for _, row in words_df.iterrows()]
    y_coords = sorted(y_coords)
    rows = []
    current_row = [y_coords[0]]
    for y in y_coords[1:]:
        if y - current_row[-1] < 15:
            current_row.append(y)
        else:
            rows.append(np.mean(current_row))
            current_row = [y]
    if current_row:
        rows.append(np.mean(current_row))
    columns = [col['x_range'][0] for col in COLUMN_DEFINITIONS.values()]
    return rows, columns

def extract_table_from_words(words_df: pd.DataFrame, page_num: int, column_ranges: Dict = None) -> pd.DataFrame:
    if words_df.empty:
        return pd.DataFrame()
    if column_ranges:
        for idx, row in words_df.iterrows():
            word_x = (row['TopLeft_X'] + row['TopRight_X']) / 2
            assigned = False
            for col_idx, col_range in column_ranges.items():
                if col_range['min_x'] <= word_x <= col_range['max_x']:
                    words_df.at[idx, 'Column_Index'] = col_idx
                    assigned = True
                    break
            if not assigned:
                min_dist = float('inf')
                nearest_col = 0
                for col_idx, col_range in column_ranges.items():
                    dist = abs(word_x - col_range['center'])
                    if dist < min_dist:
                        min_dist = dist
                        nearest_col = col_idx
                words_df.at[idx, 'Column_Index'] = nearest_col
    row_boundaries, _ = detect_table_structure_from_words(words_df)
    table_data = defaultdict(lambda: defaultdict(list))
    for _, word in words_df.iterrows():
        word_y = (word['TopLeft_Y'] + word['BottomLeft_Y']) / 2
        col_idx = int(word.get('Column_Index', 0))
        row_idx = assign_word_to_row(word_y, row_boundaries)
        table_data[row_idx][col_idx].append(word['Word'])
    rows_list = []
    num_cols = max(column_ranges.keys()) + 1 if column_ranges else 6
    for row_idx in sorted(table_data.keys()):
        row_dict = {}
        for col_idx in range(num_cols):
            col_name = COLUMN_DEFINITIONS.get(col_idx, {}).get('name', f'Column_{col_idx}')
            words = table_data[row_idx].get(col_idx, [])
            row_dict[col_name] = ' '.join(words).strip()
        rows_list.append(row_dict)
    table_df = pd.DataFrame(rows_list)
    table_df = table_df[(table_df != '').any(axis=1)]
    logging.info(f"Extracted table with {len(table_df)} rows and {len(table_df.columns)} columns")
    return table_df

####
# Placeholder Table Detection Functions
####

def detect_table_with_ppstructure(image_path: str, page_num: int) -> Tuple[List[Dict], Dict]:
    # Placeholder: Replace with actual PPStructure integration if installed
    return [], {}

def detect_table_with_line_detection(image_path: str) -> Tuple[List[Dict], Dict]:
    return [], {}

def detect_table_with_contours(image_path: str) -> Tuple[List[Dict], Dict]:
    return [], {}

def detect_table_cells_grid_based(image_path: str) -> Tuple[List[Dict], Dict]:
    # Simple fallback
    return [], {}

####
# OCR Extraction
####
def extract_ocr_words_with_coords(pdf_path: str, start_page: int, end_page: int, 
                                   ocr: PaddleOCR, output_csv: str = output_word_coords) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
    logging.info(f"Extracting words with coordinates from PDF pages {start_page}-{end_page}...")
    temp_image_dir = "temp_images"
    os.makedirs(temp_image_dir, exist_ok=True)
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Failed to open PDF: {e}")
        return pd.DataFrame(), {}

    extracted_data = []
    page_column_ranges = {}

    for page_number in range(start_page - 1, end_page):
        if page_number >= len(doc):
            logging.warning(f"Page {page_number + 1} does not exist. Skipping.")
            continue
        logging.info(f"\n{'='*60}\nProcessing Page {page_number + 1}\n{'='*60}")
        page = doc[page_number]
        pix = page.get_pixmap(dpi=300)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_np = np.array(image)
        temp_image_path = os.path.join(temp_image_dir, f'page_{page_number + 1}.png')
        image.save(temp_image_path)

        # Table detection (placeholder)
        cells, column_ranges = detect_table_with_ppstructure(temp_image_path, page_number + 1)
        if not cells or len(cells) < 10:
            cells, column_ranges = detect_table_with_line_detection(temp_image_path)
        if not cells or len(cells) < 10:
            cells, column_ranges = detect_table_with_contours(temp_image_path)
        if not cells or len(cells) < 10:
            cells, column_ranges = detect_table_cells_grid_based(temp_image_path)

        if column_ranges:
            page_column_ranges[page_number + 1] = column_ranges

        # OCR
        results = ocr.predict(image_np)[0]
        rec_texts = results.get("rec_texts", [])
        rec_polys = results.get("rec_polys", [])
        rec_scores = results.get("rec_scores", [])

        for i, word in enumerate(rec_texts):
            poly = rec_polys[i] if i < len(rec_polys) else [[0,0]]*4
            score = rec_scores[i] if i < len(rec_scores) else None
            word_x = (poly[0][0] + poly[2][0]) / 2
            col_idx, col_name = assign_word_to_column(word_x)
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
                "Page": page_number + 1,
                "Column_Index": col_idx,
                "Column_Name": col_name
            }
            extracted_data.append(row)

    doc.close()
    df = pd.DataFrame(extracted_data)
    if not df.empty:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        logging.info(f"✅ Word-coordinate CSV saved to: {output_csv}")
    return df, page_column_ranges

####
# Main Function
####
def main():
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    words_df, page_column_ranges = extract_ocr_words_with_coords(pdf_path, start_page, end_page, ocr)

    if not words_df.empty:
        for page_num in range(start_page, end_page + 1):
            page_words = words_df[words_df['Page'] == page_num]
            if not page_words.empty:
                print(f"\n{'='*60}\nEXTRACTING TABLE STRUCTURE FOR PAGE {page_num}\n{'='*60}\n")
                column_ranges = page_column_ranges.get(page_num, {})
                table_df = extract_table_from_words(page_words, page_num, column_ranges)
                if not table_df.empty:
                    table_csv_path = output_table_csv.format(page=page_num)
                    os.makedirs(os.path.dirname(table_csv_path), exist_ok=True)
                    table_df.to_csv(table_csv_path, index=False)
                    print(f"✅ Table CSV saved to: {table_csv_path}")
                    print(table_df.head(10).to_string())
                    print(f"Table shape: {table_df.shape[0]} rows × {table_df.shape[1]} columns")
                    for col in table_df.columns:
                        non_empty = (table_df[col] != '').sum()
                        print(f"  {col}: {non_empty}/{len(table_df)} rows")

    print(f"\n{'='*60}\n✅ PROCESSING COMPLETE!\n{'='*60}")
    print(f"Output files:")
    print(f"  - Raw OCR data: {output_word_coords}")
    print(f"  - Table data: {output_table_csv.format(page='*')}")

if __name__ == "__main__":
    main()
