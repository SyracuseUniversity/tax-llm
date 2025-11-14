#!/usr/bin/env python3
#python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
#python -m pip install paddleocr --quiet
#python -m pip install pymupdf --quiet

import os
import pandas as pd
import json
from PIL import Image
import fitz  # PyMuPDF for PDF processing
from paddleocr import PaddleOCR
import numpy as np
import logging
from typing import List, Dict, Tuple
import cv2

####
# Configuration
####
pdf_path = r'new-work/1950 Schedule A (scanned by ILL, Reed College).pdf'
output_word_coords = r'new-work/output/ocr_word_coords.csv'
output_cells_json = r'new-work/output/table_cells_page_{page}.json'
output_cells_image = r'new-work/output/cells_visualization_page_{page}.png'
temp_image_dir = r'new-work/temp'
start_page = 29
end_page = 29

####
# Logging Setup
####
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

####
# Enhanced Table Cell Detection
####
def detect_table_cells_rtdetr(image_path: str, page_num: int) -> Tuple[List[Dict], Dict]:
    """
    Detect table cells using RT-DETR-L model.
    """
    logging.info(f"Attempting RT-DETR-L cell detection from: {image_path}")
    
    try:
        from paddleocr import TableCellsDetection
        
        model = TableCellsDetection(model_name="RT-DETR-L_wired_table_cell_det")
        output = model.predict(image_path, threshold=0.3, batch_size=1)
        
        cells = []
        
        for res in output:
            # Save JSON output
            json_path = output_cells_json.format(page=page_num)
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            res.save_to_json(json_path)
            
            # Save visualization
            img_out_path = output_cells_image.format(page=page_num)
            res.save_to_img(os.path.dirname(img_out_path))
            logging.info(f"Saved cell visualization to: {img_out_path}")
            
            # Load and parse JSON
            with open(json_path, 'r') as f:
                detection_data = json.load(f)
            
            logging.info(f"RT-DETR-L JSON structure: {list(detection_data.keys())}")
            
            # Parse boxes - handle different JSON structures
            boxes = []
            scores = []
            
            if 'boxes' in detection_data:
                boxes = detection_data['boxes']
                scores = detection_data.get('scores', [1.0] * len(boxes))
            elif 'bbox' in detection_data:
                boxes = detection_data['bbox']
                scores = detection_data.get('scores', [1.0] * len(boxes))
            elif isinstance(detection_data, list):
                for item in detection_data:
                    if 'bbox' in item:
                        boxes.append(item['bbox'])
                        scores.append(item.get('score', 1.0))
            
            # Process each detected cell
            for i, box in enumerate(boxes):
                if len(box) >= 4:
                    x1, y1, x2, y2 = box[:4]
                    cell = {
                        'x': int(x1),
                        'y': int(y1),
                        'w': int(x2 - x1),
                        'h': int(y2 - y1),
                        'x_center': (x1 + x2) / 2,
                        'y_center': (y1 + y2) / 2,
                        'confidence': float(scores[i]) if i < len(scores) else 1.0
                    }
                    cells.append(cell)
        
        logging.info(f"✅ RT-DETR-L detected {len(cells)} cells")
        
        if cells and len(cells) >= 10:
            column_ranges = cluster_cells_into_columns(cells)
            return cells, column_ranges
        else:
            logging.warning(f"RT-DETR-L detected only {len(cells)} cells, falling back to grid-based")
            return detect_table_cells_grid_based(image_path)
    
    except Exception as e:
        logging.error(f"RT-DETR-L failed: {e}")
        import traceback
        traceback.print_exc()
        return detect_table_cells_grid_based(image_path)

def detect_table_cells_opencv(image_path: str) -> Tuple[List[Dict], Dict]:
    """
    Detect table cells using OpenCV line detection.
    More robust for structured tables with visible borders.
    """
    logging.info("Using OpenCV line-based cell detection...")
    
    try:
        # Read image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine lines
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out noise (too small cells)
            if w > 50 and h > 20 and w < img.shape[1] * 0.8:
                cell = {
                    'x': int(x),
                    'y': int(y),
                    'w': int(w),
                    'h': int(h),
                    'x_center': x + w / 2,
                    'y_center': y + h / 2,
                    'confidence': 1.0
                }
                cells.append(cell)
        
        logging.info(f"OpenCV detected {len(cells)} cells")
        
        if cells and len(cells) >= 10:
            column_ranges = cluster_cells_into_columns(cells)
            return cells, column_ranges
        else:
            logging.warning("OpenCV detected too few cells, falling back to grid-based")
            return detect_table_cells_grid_based(image_path)
    
    except Exception as e:
        logging.error(f"OpenCV detection failed: {e}")
        return detect_table_cells_grid_based(image_path)

def detect_table_cells_grid_based(image_path: str) -> Tuple[List[Dict], Dict]:
    """
    Fallback: Create a grid-based cell structure based on known column boundaries.
    Uses the exact boundaries from your debug output.
    """
    logging.info("Using grid-based cell detection with known boundaries...")
    
    # Load image to get dimensions
    img = Image.open(image_path)
    width, height = img.size
    
    # Define column boundaries based on page 29 debug output
    # cluster#0: xmid=235.5 [208.0, 263.0] → Commodity Number
    # cluster#1: xmid=535.0 [447.0, 623.0] → Commodity Description
    # cluster#2+3: xmid~1240 [1160.0, 1330.0] → Unit of Quantity
    # cluster#4: xmid=1441.0 [1424.0, 1458.0] → Rate of Duty 1930
    # cluster#5: xmid=1682.5 [1661.0, 1704.0] → Rate of Duty Trade Agreement
    # Rightmost → Tariff Paragraph
    
    column_boundaries = [
        (208, 447),    # Commodity Number
        (447, 1160),   # Commodity Description
        (1160, 1380),  # Unit of Quantity (merged cluster#2+3)
        (1380, 1600),  # Rate of Duty 1930 (cluster#4 expanded)
        (1600, 1850),  # Rate of Duty Trade Agreement (cluster#5 expanded)
        (1850, 2100)   # Tariff Paragraph
    ]
    
    # Create row boundaries
    row_height = 35
    row_start = 500  # Start after headers
    row_end = height - 100  # End before footer
    
    cells = []
    row_y = row_start
    row_idx = 0
    
    while row_y < row_end:
        for col_idx, (x_min, x_max) in enumerate(column_boundaries):
            cell = {
                'x': x_min,
                'y': row_y,
                'w': x_max - x_min,
                'h': row_height,
                'x_center': (x_min + x_max) / 2,
                'y_center': row_y + row_height / 2,
                'confidence': 1.0,
                'row': row_idx,
                'col': col_idx
            }
            cells.append(cell)
        
        row_y += row_height
        row_idx += 1
    
    logging.info(f"✅ Created {len(cells)} grid-based cells ({row_idx} rows × {len(column_boundaries)} columns)")
    
    # Create column ranges
    column_ranges = {}
    for idx, (x_min, x_max) in enumerate(column_boundaries):
        column_ranges[idx] = {
            'min_x': x_min,
            'max_x': x_max,
            'center': (x_min + x_max) / 2
        }
    
    return cells, column_ranges

def cluster_cells_into_columns(cells: List[Dict]) -> Dict[int, Dict]:
    """
    Cluster cells into columns based on X coordinates.
    """
    if not cells:
        return {}
    
    # Extract x_centers and sort
    x_centers = sorted([cell['x_center'] for cell in cells])
    
    # Remove duplicates within 10px
    unique_x = []
    for x in x_centers:
        if not unique_x or x - unique_x[-1] > 10:
            unique_x.append(x)
    
    # Group into columns (gap > 80px means new column)
    columns = []
    current_col = [unique_x[0]]
    
    for i in range(1, len(unique_x)):
        if unique_x[i] - current_col[-1] < 80:  # Same column
            current_col.append(unique_x[i])
        else:  # New column
            columns.append(current_col)
            current_col = [unique_x[i]]
    
    columns.append(current_col)
    
    # Create column ranges with padding
    column_ranges = {}
    for idx, col in enumerate(columns):
        min_x = min(col)
        max_x = max(col)
        padding = 40
        column_ranges[idx] = {
            'min_x': min_x - padding,
            'max_x': max_x + padding,
            'center': np.mean(col)
        }
    
    logging.info(f"✅ Identified {len(column_ranges)} columns from cells")
    for idx, col_range in column_ranges.items():
        logging.info(f"  Column {idx}: X=[{col_range['min_x']:.0f}, {col_range['max_x']:.0f}] center={col_range['center']:.0f}")
    
    return column_ranges

def determine_column_type(col_idx: int, total_cols: int) -> str:
    """
    Determine the semantic type of a column based on its position.
    """
    if total_cols >= 6:
        column_types = [
            'commodity_number',
            'commodity_description',
            'unit_of_quantity',
            'rate_duty_1930',
            'rate_duty_trade_agreement',
            'tariff_paragraph'
        ]
        return column_types[col_idx] if col_idx < len(column_types) else 'unknown'
    else:
        return f'column_{col_idx}'

def assign_word_to_cell(word_x: float, word_y: float, cells: List[Dict]) -> Dict:
    """
    Assign a word to the most appropriate cell based on coordinates.
    """
    # First, try exact match - word center inside cell
    for cell in cells:
        if (cell['x'] <= word_x <= cell['x'] + cell['w'] and
            cell['y'] <= word_y <= cell['y'] + cell['h']):
            return cell
    
    # If no exact match, find nearest cell
    min_dist = float('inf')
    nearest_cell = None
    
    for cell in cells:
        dist = np.sqrt((word_x - cell['x_center'])**2 + (word_y - cell['y_center'])**2)
        if dist < min_dist:
            min_dist = dist
            nearest_cell = cell
    
    # Only assign if reasonably close (within 200 pixels)
    return nearest_cell if min_dist < 200 else None

def assign_word_to_column(word_x: float, column_ranges: Dict[int, Dict]) -> Tuple[int, str]:
    """
    Assign a word to a column based on X coordinate.
    """
    # First, try exact match
    for col_idx, col_range in column_ranges.items():
        if col_range['min_x'] <= word_x <= col_range['max_x']:
            col_type = determine_column_type(col_idx, len(column_ranges))
            return col_idx, col_type
    
    # If no exact match, find nearest column
    min_dist = float('inf')
    nearest_col = None
    
    for col_idx, col_range in column_ranges.items():
        dist = abs(word_x - col_range['center'])
        if dist < min_dist:
            min_dist = dist
            nearest_col = col_idx
    
    if nearest_col is not None:
        col_type = determine_column_type(nearest_col, len(column_ranges))
        return nearest_col, col_type
    
    return None, None

####
# OCR Extraction with Cell Detection
####
def extract_ocr_words_with_coords(pdf_path: str, start_page: int, end_page: int, 
                                   ocr: PaddleOCR, output_csv: str = output_word_coords) -> None:
    """
    Extract words and their coordinates from PDF using OCR with enhanced cell detection.
    """
    logging.info(f"Extracting words with coordinates from PDF pages {start_page}-{end_page}...")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Failed to open PDF: {e}")
        return

    extracted_data = []
    os.makedirs(temp_image_dir, exist_ok=True)

    for page_number in range(start_page - 1, end_page):
        if page_number >= len(doc):
            logging.warning(f"Page {page_number + 1} does not exist. Skipping.")
            continue

        logging.info(f"\n{'='*60}")
        logging.info(f"Processing Page {page_number + 1}...")
        logging.info(f"{'='*60}")
        
        page = doc[page_number]
        pix = page.get_pixmap(dpi=300)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_np = np.array(image)
        
        # Save temporary image for cell detection
        temp_image_path = os.path.join(temp_image_dir, f'page_{page_number + 1}.png')
        image.save(temp_image_path)
        
        # Try multiple cell detection methods in order of preference
        cells, column_ranges = None, None
        
        # 1. Try RT-DETR-L
        try:
            cells, column_ranges = detect_table_cells_rtdetr(temp_image_path, page_number + 1)
        except Exception as e:
            logging.warning(f"RT-DETR-L failed: {e}")
        
        # 2. Try OpenCV if RT-DETR-L failed
        if not cells or len(cells) < 10:
            try:
                cells, column_ranges = detect_table_cells_opencv(temp_image_path)
            except Exception as e:
                logging.warning(f"OpenCV failed: {e}")
        
        # 3. Fallback to grid-based
        if not cells or len(cells) < 10:
            cells, column_ranges = detect_table_cells_grid_based(temp_image_path)
        
        # Save cell detection results
        cells_output = {
            'page': page_number + 1,
            'total_cells': len(cells),
            'cells_sample': cells[:10],  # Save first 10 cells as sample
            'columns': {k: {
                'min_x': float(v['min_x']),
                'max_x': float(v['max_x']),
                'center': float(v['center'])
            } for k, v in column_ranges.items()}
        }
        
        cells_json_path = output_cells_json.format(page=page_number + 1)
        os.makedirs(os.path.dirname(cells_json_path), exist_ok=True)
        with open(cells_json_path, 'w') as f:
            json.dump(cells_output, f, indent=2)
        logging.info(f"✅ Saved cell detection results to: {cells_json_path}")

        # Run OCR
        logging.info("Running OCR on page...")
        results = ocr.predict(image_np)[0]
        rec_texts = results.get("rec_texts", [])
        rec_polys = results.get("rec_polys", [])
        rec_scores = results.get("rec_scores", [])

        logging.info(f"✅ OCR extracted {len(rec_texts)} words")

        for i, word in enumerate(rec_texts):
            poly = rec_polys[i] if i < len(rec_polys) else [[None, None]] * 4
            score = rec_scores[i] if i < len(rec_scores) else None
            
            # Calculate word center
            word_x = (poly[0][0] + poly[2][0]) / 2
            word_y = (poly[0][1] + poly[2][1]) / 2
            
            # Assign word to cell (for cell-level info)
            assigned_cell = assign_word_to_cell(word_x, word_y, cells)
            
            # Assign word to column (more reliable for column classification)
            cell_column, column_type = assign_word_to_column(word_x, column_ranges)
            
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
                "Cell_X": assigned_cell['x'] if assigned_cell else None,
                "Cell_Y": assigned_cell['y'] if assigned_cell else None,
                "Cell_Width": assigned_cell['w'] if assigned_cell else None,
                "Cell_Height": assigned_cell['h'] if assigned_cell else None,
                "Cell_Column": cell_column,
                "Column_Type": column_type
            }
            extracted_data.append(row)

    doc.close()

    # Create DataFrame
    headers = [
        "Word", "Confidence",
        "TopLeft_X", "TopLeft_Y",
        "TopRight_X", "TopRight_Y",
        "BottomRight_X", "BottomRight_Y",
        "BottomLeft_X", "BottomLeft_Y",
        "Page",
        "Cell_X", "Cell_Y", "Cell_Width", "Cell_Height",
        "Cell_Column", "Column_Type"
    ]

    df = pd.DataFrame(extracted_data, columns=headers)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logging.info(f"\n✅ Word-coordinate CSV with cell info saved to: {output_csv}")
    
    # Log statistics
    logging.info(f"\n{'='*60}")
    logging.info(f"EXTRACTION STATISTICS")
    logging.info(f"{'='*60}")
    logging.info(f"Total words extracted: {len(df)}")
    logging.info(f"Words assigned to columns: {df['Cell_Column'].notna().sum()}")
    logging.info(f"Words NOT assigned: {df['Cell_Column'].isna().sum()}")
    
    if 'Column_Type' in df.columns and df['Column_Type'].notna().sum() > 0:
        logging.info(f"\n{'='*60}")
        logging.info("WORDS PER COLUMN TYPE:")
        logging.info(f"{'='*60}")
        for col_type, count in df['Column_Type'].value_counts().items():
            logging.info(f"  {col_type}: {count} words")
    
    # Show sample of assigned words per column
    logging.info(f"\n{'='*60}")
    logging.info("SAMPLE WORDS PER COLUMN:")
    logging.info(f"{'='*60}")
    for col_type in sorted(df['Column_Type'].dropna().unique()):
        sample_words = df[df['Column_Type'] == col_type]['Word'].head(5).tolist()
        logging.info(f"  {col_type}: {', '.join(sample_words)}")

def main():
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)

    # Extract OCR words with coordinates and cell detection
    print("\n" + "="*60)
    print("STARTING OCR EXTRACTION WITH ENHANCED CELL DETECTION")
    print("="*60 + "\n")
    extract_ocr_words_with_coords(pdf_path, start_page, end_page, ocr)
    print(f"\n✅ Raw OCR data with cell info saved to: {output_word_coords}")
    print(f"✅ Cell detection results saved to: {output_cells_json}")

if __name__ == "__main__":
    main()