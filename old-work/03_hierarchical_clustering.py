import pandas as pd
import numpy as np
import logging
import os

# =========================
# Configuration
# =========================
INPUT_CSV = "old-work/output/cleaned_classified_words.csv"
OUTPUT_CSV = "old-work/output/new_hierarchical_commodities.csv"
OUTPUT_TXT = "old-work/output/formatted_commodities.txt"
Y_PROXIMITY_THRESHOLD = 50  # Max vertical distance to consider lines as continuations

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

def combine_split_lines(data: pd.DataFrame, y_threshold: int = Y_PROXIMITY_THRESHOLD) -> pd.DataFrame:
    """
    Combine lines that are likely continuations of the previous line,
    based on indentation and vertical proximity.
    """
    processed_data = data.copy()
    skip_indices = set()
    for i in range(len(data) - 1):
        try:
            curr_x = data.iloc[i]['TopLeft_X']
            next_x = data.iloc[i+1]['TopLeft_X']
            curr_y = data.iloc[i]['TopLeft_Y']
            next_y = data.iloc[i+1]['TopLeft_Y']
            curr_desc = str(data.iloc[i]['Commodity Description']).strip() if pd.notna(data.iloc[i]['Commodity Description']) else ''
            next_desc = str(data.iloc[i+1]['Commodity Description']).strip() if pd.notna(data.iloc[i+1]['Commodity Description']) else ''
            y_proximity = next_y - curr_y
            if curr_desc and next_desc:
                if (next_x > curr_x) and (y_proximity < y_threshold) and not curr_desc.endswith(':'):
                    combined_desc = f"{curr_desc} {next_desc}"
                    processed_data.at[i, 'Commodity Description'] = combined_desc
                    skip_indices.add(i+1)
        except (IndexError, KeyError):
            continue
    # Remove combined rows
    result_data = processed_data.drop(index=list(skip_indices)).reset_index(drop=True)
    return result_data

def process_commodity_descriptions_by_pixels(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build hierarchical descriptions using indentation (X position) and parent-child logic.
    """
    data = data.sort_values(by=['Page', 'TopLeft_Y'])
    x_values = sorted(data['TopLeft_X'].unique())
    x_level_map = {x: level for level, x in enumerate(x_values)}
    current_texts = [None] * len(x_values)
    data['Hierarchical Description'] = None
    data['Is Parent'] = False

    for idx, row in data.iterrows():
        description = str(row['Commodity Description']).strip() if pd.notna(row['Commodity Description']) else ''
        if not description:
            continue
        x_coord = row['TopLeft_X']
        level = x_level_map[x_coord]
        current_texts[level] = description
        for i in range(level + 1, len(current_texts)):
            current_texts[i] = None
        is_parent = description.endswith(':')
        data.at[idx, 'Is Parent'] = is_parent
        if level == 0 and not is_parent:
            data.at[idx, 'Hierarchical Description'] = description
        elif not is_parent:
            parent_texts = [text for text in current_texts[:level] if text and text.endswith(':')]
            if parent_texts:
                hierarchical_desc = " ".join(parent_texts) + " " + description
                data.at[idx, 'Hierarchical Description'] = hierarchical_desc
            else:
                data.at[idx, 'Hierarchical Description'] = description
    # Keep only non-parent rows or those with a hierarchical description
    result = data[(~data['Is Parent']) | (data['Hierarchical Description'].notna())].copy()
    result = result.drop(columns=['Is Parent'])
    return result

def save_outputs(hierarchical_data: pd.DataFrame, csv_path: str, txt_path: str):
    """
    Save the processed data to CSV and the hierarchical descriptions to a text file.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    
    # Save CSV with UTF-8 encoding
    hierarchical_data.to_csv(csv_path, index=False, encoding='utf-8')
    
    # Save text file with UTF-8 encoding and comprehensive error handling
    with open(txt_path, "w", encoding='utf-8', errors='ignore') as f:
        for desc in hierarchical_data['Hierarchical Description'].dropna():
            try:
                # Convert to string and clean problematic characters
                clean_desc = str(desc).strip()
                
                # Remove or replace problematic Unicode characters
                # Replace common problematic characters
                clean_desc = clean_desc.replace('\u8bd5', '[CHAR]')  # Replace the specific character causing issues
                clean_desc = clean_desc.replace('\ufffd', '[REPLACEMENT]')  # Replace replacement characters
                
                # Remove any remaining non-printable characters except newlines and tabs
                clean_desc = ''.join(char for char in clean_desc if char.isprintable() or char in '\n\t')
                
                # Only write non-empty descriptions
                if clean_desc:
                    f.write(clean_desc + "\n")
                    
            except UnicodeEncodeError as e:
                logging.warning(f"Unicode encoding error for description: {e}")
                # Write a sanitized version
                try:
                    sanitized = str(desc).encode('ascii', errors='ignore').decode('ascii')
                    if sanitized.strip():
                        f.write(f"[SANITIZED]: {sanitized.strip()}\n")
                    else:
                        f.write("[UNICODE_ERROR]\n")
                except Exception:
                    f.write("[ENCODING_ERROR]\n")
            except Exception as e:
                logging.warning(f"Unexpected error writing description: {e}")
                f.write("[PROCESSING_ERROR]\n")
    
    logging.info(f"Processed data saved to: {csv_path}")
    logging.info(f"Formatted descriptions saved to: {txt_path}")

def print_sample(hierarchical_data: pd.DataFrame, n: int = 10):
    """
    Print a sample of the hierarchical descriptions.
    """
    print("\nSample of processed hierarchical descriptions:")
    for desc in hierarchical_data['Hierarchical Description'].dropna().head(n):
        print(desc)

# =========================
# Main Processing
# =========================

def main():
    # Load data
    data = pd.read_csv(INPUT_CSV)
    logging.info(f"Loaded {len(data)} rows from {INPUT_CSV}")

    # Combine split lines
    combined_data = combine_split_lines(data)
    logging.info(f"After combining split lines: {len(combined_data)} rows")

    # Build hierarchy
    hierarchical_data = process_commodity_descriptions_by_pixels(combined_data)
    logging.info(f"After hierarchy processing: {len(hierarchical_data)} rows")

    # Save outputs
    save_outputs(hierarchical_data, OUTPUT_CSV, OUTPUT_TXT)

    # Print sample
    print_sample(hierarchical_data)

if __name__ == "__main__":
    main()