import pandas as pd
import numpy as np
import logging
import re

# =========================
# Configuration
# =========================
INPUT_CSV = r"new-work/output/cleaned_classified_words.csv"
FINAL_TABLE_CSV = r"new-work/output/final-table.csv"
OUTPUT_TXT = r"new-work/output/formatted_commodities.txt"
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
    processed_data = data.copy().reset_index(drop=True)  # Reset index to avoid KeyError
    skip_indices = set()
    
    for i in range(len(processed_data) - 1):
        try:
            curr_x = processed_data.iloc[i]['TopLeft_X']
            next_x = processed_data.iloc[i+1]['TopLeft_X']
            curr_y = processed_data.iloc[i]['TopLeft_Y']
            next_y = processed_data.iloc[i+1]['TopLeft_Y']
            curr_desc = str(processed_data.iloc[i]['Commodity Description']).strip() if pd.notna(processed_data.iloc[i]['Commodity Description']) else ''
            next_desc = str(processed_data.iloc[i+1]['Commodity Description']).strip() if pd.notna(processed_data.iloc[i+1]['Commodity Description']) else ''
            y_proximity = next_y - curr_y
            
            if curr_desc and next_desc:
                if (next_x > curr_x) and (y_proximity < y_threshold) and not curr_desc.endswith(':'):
                    combined_desc = f"{curr_desc} {next_desc}"
                    processed_data.at[i, 'Commodity Description'] = combined_desc
                    skip_indices.add(i+1)
        except (IndexError, KeyError) as e:
            logging.warning(f"Error at index {i}: {e}")
            continue
    
    # Remove combined rows - only drop indices that actually exist
    valid_skip_indices = [idx for idx in skip_indices if idx < len(processed_data)]
    if valid_skip_indices:
        result_data = processed_data.drop(index=valid_skip_indices).reset_index(drop=True)
    else:
        result_data = processed_data.reset_index(drop=True)
    
    return result_data

def process_commodity_descriptions_by_pixels(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build hierarchical descriptions using indentation (X position) and parent-child logic.
    Enhanced to create cleaner, more targeted descriptions matching expected format.
    """
    data = data.sort_values(by=['Page', 'TopLeft_Y'])
    x_values = sorted(data['TopLeft_X'].unique())
    x_level_map = {x: level for level, x in enumerate(x_values)}
    current_texts = [None] * len(x_values)
    data['Is Parent'] = False

    for idx, row in data.iterrows():
        description = str(row['Commodity Description']).strip() if pd.notna(row['Commodity Description']) else ''
        if not description:
            continue
            
        # Apply OCR corrections
        description = apply_advanced_ocr_corrections(description)
        
        x_coord = row['TopLeft_X']
        level = x_level_map[x_coord]
        current_texts[level] = description
        
        # Clear deeper levels when we encounter a new item at this level
        for i in range(level + 1, len(current_texts)):
            current_texts[i] = None
        
        # Enhanced parent detection
        is_parent = (description.endswith(':') or 
                    level == 0 or  # Top-level items are often parents
                    x_coord < 100 or  # Far-left items are likely parents
                    any(pattern in description.lower() for pattern in 
                        ['cattle', 'sheep', 'lambs', 'animals', 'live', 'meat', 'poultry', 
                         'fresh', 'chilled', 'frozen', 'prepared', 'preserved']))
        
        data.at[idx, 'Is Parent'] = is_parent
        
        if level == 0 and not is_parent:
            # Top level non-parent items keep their original description
            data.at[idx, 'Commodity Description'] = description
        elif not is_parent:
            # Build targeted hierarchical description
            relevant_parents = []
            for i in range(level):
                if current_texts[i] and current_texts[i] != description:
                    parent_text = current_texts[i].rstrip(':').strip()
                    if parent_text and not is_noise_text(parent_text):
                        relevant_parents.append(parent_text)
            
            # Create concise description
            if relevant_parents:
                # Only use the most relevant parent (last one)
                main_parent = relevant_parents[-1] if relevant_parents else ""
                if main_parent and not description.startswith(main_parent):
                    hierarchical_desc = f"{main_parent}: {description}"
                else:
                    hierarchical_desc = description
                data.at[idx, 'Commodity Description'] = hierarchical_desc
            else:
                data.at[idx, 'Commodity Description'] = description
        else:
            # This is a parent item - keep it but might filter later
            data.at[idx, 'Commodity Description'] = description
    
    # Filter to keep only meaningful descriptions
    result = data[
        (~data['Is Parent']) |  # Keep all non-parent rows
        (data['Is Parent'] & data['Commodity Description'].str.contains(':', na=False))  # Keep parent rows with colons
    ].copy()
    
    result = result.drop(columns=['Is Parent'])
    return result

def apply_advanced_ocr_corrections(text):
    """Apply comprehensive OCR corrections for better text quality."""
    if not text:
        return ""
    
    # Common OCR corrections
    corrections = {
        'Catt1e': 'Cattle', 'p0unds': 'pounds', 'eachch': 'each', 'chi11ed': 'chilled',
        'fr0zen': 'frozen', 'P0u1try': 'Poultry', '1ive': 'live', '8eef': 'Beef',
        'Veach1': 'Veal', 'H0gs': 'Hogs', '60ats': 'Goats', '8aby': 'Baby',
        'guineachs': 'guineas', 'inc1uding': 'including', 'c0rned': 'corned',
        'Deachd': 'Dead', 'undressed': 'undressed', 'Turkeys': 'Turkeys',
        'Ducks': 'Ducks', '8irds': 'Birds', 'Mutt0n': 'Mutton', 'Reindeer': 'Reindeer',
        'meacht': 'meat', 'Venis0n': 'Venison', '1ess': 'less', 'm0re': 'more',
        'dairy': 'dairy', 'purp0ses': 'purposes', 'kidneys': 'kidneys',
        't0ngues': 'tongues', 'heachrts': 'hearts', '0ffa1': 'offal',
        'ca1ves': 'calves', 'n. 8. p. f': 'n. s. p. f', 'chicks': 'chicks',
        '0f': 'of', '0r': 'or', 'f0r': 'for', '（': '(', '）': ')',
        '1ess': 'less', 'and less': 'and less', 'or more': 'or more'
    }
    
    for incorrect, correct in corrections.items():
        text = text.replace(incorrect, correct)
    
    # Clean up extra spaces and dashes
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'-+', '-', text)   # Multiple dashes to single dash
    text = text.strip()
    
    return text

def is_noise_text(text):
    """Check if text is noise that should be filtered out."""
    noise_patterns = [
        r'^[A-Z0-9\s]+$',  # All caps abbreviations
        r'^\d+$',          # Just numbers
        r'^[^\w\s]+$',     # Just punctuation
        r'SCHEDULE A',
        r'COMMODITY',
        r'RATE OF DUTY',
        r'TARIFF',
        r'Group.*ANIMAL',
        r'ECONOMIC CLASS'
    ]
    
    for pattern in noise_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False

def save_outputs(hierarchical_data: pd.DataFrame, final_table_path: str, txt_path: str):
    """
    Save the processed data by updating the final table with hierarchical descriptions.
    """
    import os
    os.makedirs(os.path.dirname(final_table_path), exist_ok=True)
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    
    # Load the final table created by 02_commodity_number.py
    try:
        final_table = pd.read_csv(final_table_path)
        logging.info(f"Loaded final table with {len(final_table)} rows from {final_table_path}")
    except FileNotFoundError:
        logging.error(f"Final table not found at {final_table_path}. Please run 02_commodity_number.py first.")
        return
    
    # Debug: Check what columns are available
    logging.info(f"Hierarchical data columns: {list(hierarchical_data.columns)}")
    logging.info(f"Sample hierarchical data rows: {len(hierarchical_data)}")
    
    # Create a mapping of commodity numbers to descriptions
    commodity_desc_map = {}
    
    # Load the original cleaned data to get commodity numbers
    original_data = pd.read_csv(INPUT_CSV)
    
    # Group descriptions by Y-coordinate proximity to match with commodity numbers
    for idx, row in hierarchical_data.iterrows():
        description = str(row.get('Commodity Description', '')).strip()
        if not description or description == 'nan':
            continue
            
        # Find nearby commodity numbers in the original data using coordinates
        y_coord = row.get('TopLeft_Y', 0)
        tolerance = 30  # Y-coordinate tolerance
        
        # Find commodity numbers within Y-coordinate range
        nearby_commodities = original_data[
            (original_data['Commodity Number'].notna()) & 
            (original_data['Commodity Number'] != '') &
            (abs(original_data['TopLeft_Y'] - y_coord) <= tolerance)
        ]
        
        if len(nearby_commodities) > 0:
            # Get the closest commodity number
            closest_commodity = nearby_commodities.iloc[0]['Commodity Number']
            commodity_num = str(closest_commodity).strip()
            
            if commodity_num and commodity_num != 'nan':
                commodity_desc_map[commodity_num] = description
                logging.info(f"Mapped: {commodity_num} -> {description[:50]}...")
    
    logging.info(f"Created {len(commodity_desc_map)} commodity-description mappings")
    
    # Update the final table with hierarchical descriptions
    updated_count = 0
    
    logging.info("Debugging commodity mappings:")
    logging.info(f"Sample mappings: {list(commodity_desc_map.keys())[:5]}")
    logging.info(f"Sample final table commodity numbers: {final_table['SCHEDULE A COMMODITY NUMBER'].head().tolist()}")
    
    for idx, row in final_table.iterrows():
        commodity_num_formatted = str(row['SCHEDULE A COMMODITY NUMBER']).strip()
        
        # Convert "0106 000" to "10600.0" format to match our mappings
        # Remove spaces: "0106 000" -> "0106000"
        clean_num = commodity_num_formatted.replace(' ', '')
        
        # Convert "0106000" to "10600.0" format by removing leading zero and adjusting trailing zeros
        if len(clean_num) == 7:
            # "0106000" -> remove leading zero -> "106000"
            without_leading_zero = clean_num[1:]  # "106000"
            
            # For the format conversion, we need to match the pattern in our data
            # Our data has: 10600.0, 10700.0, 12000.0, 12200.0, etc.
            # Pattern: take first 3-5 significant digits, add trailing zeros to make it reasonable
            
            # Remove all trailing zeros first
            without_trailing_zeros = without_leading_zero.rstrip('0')  # "106", "12", etc.
            
            # Apply logic based on the length to match our data patterns
            if len(without_trailing_zeros) == 2:  # "12" -> should become "12000"
                converted_num = without_trailing_zeros + '000'
            elif len(without_trailing_zeros) == 3:  # "106" -> should become "10600"  
                converted_num = without_trailing_zeros + '00'
            elif len(without_trailing_zeros) == 4:  # "2540" -> should become "25400"
                converted_num = without_trailing_zeros + '0'
            else:  # 5+ digits or other cases
                converted_num = without_trailing_zeros
            
            # Add decimal: "10600" -> "10600.0"
            mapping_key = f"{converted_num}.0"
        else:
            mapping_key = clean_num
        
        logging.info(f"Converting {commodity_num_formatted} -> {mapping_key}")
        
        # Look for exact match
        if mapping_key in commodity_desc_map:
            matched_desc = commodity_desc_map[mapping_key]
            final_table.at[idx, 'COMMODITY DESCRIPTION AND ECONOMIC CLASS'] = matched_desc
            updated_count += 1
            logging.info(f"✅ Updated {commodity_num_formatted} -> {matched_desc[:50]}...")
        else:
            logging.info(f"❌ No match found for {commodity_num_formatted} (tried {mapping_key})")
    
    # Save updated final table (overwrite the original)
    final_table.to_csv(final_table_path, index=False)
    
    # Save descriptions to text file
    with open(txt_path, "w", encoding='utf-8') as f:
        for commodity_num, description in commodity_desc_map.items():
            f.write(f"{commodity_num}: {description}\n")
    
    logging.info(f"Updated final table saved to: {final_table_path}")
    logging.info(f"Updated {updated_count} commodity descriptions")
    logging.info(f"Formatted descriptions saved to: {txt_path}")

def print_sample(hierarchical_data: pd.DataFrame, n: int = 10):
    """
    Print a sample of the hierarchical descriptions.
    """
    print("\nSample of processed hierarchical descriptions:")
    
    # Load original data to get commodity numbers for context
    original_data = pd.read_csv(INPUT_CSV)
    
    valid_descriptions = hierarchical_data['Commodity Description'].dropna()
    valid_descriptions = [desc for desc in valid_descriptions if str(desc).strip()]
    
    for i, desc in enumerate(valid_descriptions[:n]):
        # Find the row with this description
        desc_row = hierarchical_data[hierarchical_data['Commodity Description'] == desc].iloc[0]
        y_coord = desc_row.get('TopLeft_Y', 0)
        
        # Find nearby commodity number
        tolerance = 30
        nearby_commodities = original_data[
            (original_data['Commodity Number'].notna()) & 
            (original_data['Commodity Number'] != '') &
            (abs(original_data['TopLeft_Y'] - y_coord) <= tolerance)
        ]
        
        commodity_num = "Unknown"
        if len(nearby_commodities) > 0:
            commodity_num = str(nearby_commodities.iloc[0]['Commodity Number']).strip()
        
        print(f"{commodity_num}: {desc}")

# =========================
# Main Processing
# =========================

def main():
    # Load data
    data = pd.read_csv(INPUT_CSV)
    logging.info(f"Loaded {len(data)} rows from {INPUT_CSV}")
    
    # Filter only rows with descriptions (ignore empty/nan descriptions)
    description_data = data[data['Commodity Description'].notna() & (data['Commodity Description'] != '')].copy()
    logging.info(f"Found {len(description_data)} rows with descriptions")

    # Combine split lines
    combined_data = combine_split_lines(description_data)
    logging.info(f"After combining split lines: {len(combined_data)} rows")

    # Build hierarchy - this updates the Description column directly
    hierarchical_data = process_commodity_descriptions_by_pixels(combined_data)
    logging.info(f"After hierarchy processing: {len(hierarchical_data)} rows")

    # Save outputs - now updates the final table with new column headers
    save_outputs(hierarchical_data, FINAL_TABLE_CSV, OUTPUT_TXT)

    # Print sample
    print_sample(hierarchical_data)
    
    print(f"\nFinal table with hierarchical descriptions updated in: {FINAL_TABLE_CSV}")
    print("The 'COMMODITY DESCRIPTION AND ECONOMIC CLASS' column has been populated.")

if __name__ == "__main__":
    main()