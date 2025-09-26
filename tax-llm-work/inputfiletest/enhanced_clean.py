"""
Enhanced cleaning script with dynamic coordinate mapping for column classification.
This script analyzes OCR coordinates dynamically to classify words into proper columns.
"""
import pandas as pd
import numpy as np
import re
import logging
import os

# Configuration
INPUT_CSV = r'new-work/output/ocr_word_coords.csv'
OUTPUT_CSV = r'new-work/output/cleaned_classified_words.csv'
DROP_ROWS_BEFORE = 14  # Number of initial rows to drop (headers/irrelevant)
EXCLUDE_FOOTER_Y_THRESHOLD = 2700  # Exclude rows with Y coordinates above this (footer area)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess(input_csv: str) -> pd.DataFrame:
    """Load the OCR CSV and perform initial cleaning."""
    df = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(df)} rows from {input_csv}")
    
    # Drop header rows
    if DROP_ROWS_BEFORE > 0:
        df = df.iloc[DROP_ROWS_BEFORE:].reset_index(drop=True)
        logging.info(f"After dropping {DROP_ROWS_BEFORE} header rows, {len(df)} rows remain.")
    
    # Exclude footer rows based on Y coordinate
    initial_count = len(df)
    df = df[df['TopLeft_Y'] < EXCLUDE_FOOTER_Y_THRESHOLD]
    excluded_count = initial_count - len(df)
    if excluded_count > 0:
        logging.info(f"Excluded {excluded_count} footer rows with Y >= {EXCLUDE_FOOTER_Y_THRESHOLD}, {len(df)} rows remain.")
    
    return df

def analyze_coordinate_zones(df):
    """Dynamically analyze coordinate patterns to identify zones for different data types."""
    zones = {}
    
    # 1. Find COMMODITY NUMBERS by pattern
    commodity_candidates = df[df['Word'].str.match(r'^\d{7}|\d{4}\s*\d{3}', na=False)]
    if len(commodity_candidates) > 0:
        x_coords = commodity_candidates['TopLeft_X'].values
        zones['commodity'] = {
            'min_x': np.min(x_coords) - 30,
            'max_x': np.max(x_coords) + 200,
            'center': np.median(x_coords)
        }
        logging.info(f"Commodity zone: X={zones['commodity']['min_x']:.0f}-{zones['commodity']['max_x']:.0f}")
    
    # 2. Find TARIFF PARAGRAPHS by pattern (rightmost)
    tariff_candidates = df[df['Word'].str.match(r'^70[0-9]|^71[0-9]|^1558', na=False)]
    if len(tariff_candidates) > 0:
        x_coords = tariff_candidates['TopLeft_X'].values
        zones['tariff'] = {
            'min_x': np.min(x_coords) - 50,
            'max_x': np.max(x_coords) + 100,
            'center': np.median(x_coords)
        }
        logging.info(f"Tariff zone: X={zones['tariff']['min_x']:.0f}-{zones['tariff']['max_x']:.0f}")
    
    # 3. Find UNITS by pattern
    unit_candidates = df[df['Word'].str.match(r'^(No|Lb|each|ea|N0|1b|Ib)\.?$', na=False, case=False)]
    if 'commodity' in zones and 'tariff' in zones:
        # Filter units between commodity and tariff
        unit_candidates = unit_candidates[
            (unit_candidates['TopLeft_X'] > zones['commodity']['max_x']) & 
            (unit_candidates['TopLeft_X'] < zones['tariff']['min_x'])
        ]
    
    if len(unit_candidates) > 0:
        x_coords = unit_candidates['TopLeft_X'].values
        
        # Remove outliers - focus on the main cluster of unit words
        # Most unit words should be clustered together
        q25, q75 = np.percentile(x_coords, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        # Filter out outliers
        filtered_coords = x_coords[(x_coords >= lower_bound) & (x_coords <= upper_bound)]
        
        if len(filtered_coords) > 0:
            # Use filtered coordinates for unit zone
            zones['unit'] = {
                'min_x': np.min(filtered_coords) - 30,
                'max_x': np.max(filtered_coords) + 50,  # Small buffer after main unit cluster
                'center': np.median(filtered_coords)
            }
            logging.info(f"Unit zone (filtered): X={zones['unit']['min_x']:.0f}-{zones['unit']['max_x']:.0f}")
            logging.info(f"Unit coordinates found: {sorted(set(x_coords))}")
            logging.info(f"Unit coordinates filtered: {sorted(set(filtered_coords))}")
            logging.info(f"Unit zone ends at {zones['unit']['max_x']:.0f}, rate area should start after this")
        else:
            # Fallback to all coordinates if filtering removes everything
            zones['unit'] = {
                'min_x': np.min(x_coords) - 30,
                'max_x': np.max(x_coords) + 50,
                'center': np.median(x_coords)
            }
            logging.info(f"Unit zone (unfiltered): X={zones['unit']['min_x']:.0f}-{zones['unit']['max_x']:.0f}")
            logging.info(f"Unit coordinates found: {sorted(set(x_coords))}")
    else:
        # If no units found, estimate a narrow zone
        if 'commodity' in zones and 'tariff' in zones:
            estimated_unit_x = zones['commodity']['max_x'] + (zones['tariff']['min_x'] - zones['commodity']['max_x']) * 0.4
            zones['unit'] = {
                'min_x': estimated_unit_x - 50,
                'max_x': estimated_unit_x + 50,
                'center': estimated_unit_x
            }
            logging.info(f"Unit zone estimated: X={zones['unit']['min_x']:.0f}-{zones['unit']['max_x']:.0f}")
    
    # 4. Calculate DESCRIPTION zone (between commodity and unit/tariff)
    if 'commodity' in zones:
        desc_start = zones['commodity']['max_x'] + 20
        if 'unit' in zones:
            desc_end = zones['unit']['min_x'] - 20
        elif 'tariff' in zones:
            desc_end = zones['tariff']['min_x'] - 20
        else:
            desc_end = desc_start + 800  # Fallback
        
        zones['description'] = {
            'min_x': desc_start,
            'max_x': desc_end,
            'center': (desc_start + desc_end) / 2
        }
        logging.info(f"Description zone: X={zones['description']['min_x']:.0f}-{zones['description']['max_x']:.0f}")
    
    # 5. Calculate RATE zones (between unit and tariff)
    if 'unit' in zones and 'tariff' in zones:
        rate_start = zones['unit']['max_x'] + 20
        rate_end = zones['tariff']['min_x'] - 20
        
        # Find actual words in the rate area (after units, before tariff)
        rate_area_words = df[
            (df['TopLeft_X'] > rate_start) & 
            (df['TopLeft_X'] < rate_end)
        ]
        
        if len(rate_area_words) > 0:
            rate_x_coords = rate_area_words['TopLeft_X'].values
            logging.info(f"Found {len(rate_area_words)} words in rate area between X={rate_start:.0f} and X={rate_end:.0f}")
            logging.info(f"Rate area coordinates: {sorted(set(rate_x_coords))}")
            logging.info(f"Sample rate words: {rate_area_words['Word'].head(10).tolist()}")
            
            # If we have enough rate data, try to split into two zones
            if len(rate_x_coords) > 3:
                # Sort coordinates and look for natural clustering
                sorted_coords = np.sort(rate_x_coords)
                
                # Try to find a gap to split into two rate columns
                # Calculate gaps between consecutive coordinates
                gaps = np.diff(sorted_coords)
                if len(gaps) > 0 and np.max(gaps) > 100:  # If there's a significant gap
                    # Find the largest gap
                    max_gap_idx = np.argmax(gaps)
                    split_point = (sorted_coords[max_gap_idx] + sorted_coords[max_gap_idx + 1]) / 2
                    
                    zones['rate_1930'] = {
                        'min_x': rate_start,
                        'max_x': split_point,
                        'center': (rate_start + split_point) / 2
                    }
                    
                    zones['rate_trade'] = {
                        'min_x': split_point,
                        'max_x': rate_end,
                        'center': (split_point + rate_end) / 2
                    }
                    
                    logging.info(f"Rate zones split at natural gap: 1930 at X={rate_start:.0f}-{split_point:.0f}, Trade at X={split_point:.0f}-{rate_end:.0f}")
                else:
                    # No clear gap, split evenly
                    rate_mid = (rate_start + rate_end) / 2
                    
                    zones['rate_1930'] = {
                        'min_x': rate_start,
                        'max_x': rate_mid,
                        'center': (rate_start + rate_mid) / 2
                    }
                    
                    zones['rate_trade'] = {
                        'min_x': rate_mid,
                        'max_x': rate_end,
                        'center': (rate_mid + rate_end) / 2
                    }
                    
                    logging.info(f"Rate zones split evenly: 1930 at X={rate_start:.0f}-{rate_mid:.0f}, Trade at X={rate_mid:.0f}-{rate_end:.0f}")
            else:
                # Few rate words, create single zone for 1930
                zones['rate_1930'] = {
                    'min_x': rate_start,
                    'max_x': rate_end,
                    'center': (rate_start + rate_end) / 2
                }
                logging.info(f"Single rate zone (1930): X={rate_start:.0f}-{rate_end:.0f}")
            
            # Log the actual rate words found
            rate_words = rate_area_words['Word'].tolist()
            logging.info(f"Found {len(rate_words)} words in rate area: {rate_words[:10]}...")  # Show first 10
        else:
            logging.info("No words found in calculated rate area")
    else:
        logging.info("Cannot calculate rate zones - missing unit or tariff zones")
    
    return zones

def clean_commodity_number(raw_number):
    """
    Clean and standardize a single commodity number.
    Expected format: 7 digits (e.g., 0010600)
    Returns None for OCR artifacts that should be removed.
    """
    if pd.isna(raw_number):
        return None
    
    # Convert to string and clean
    clean_num = str(raw_number).strip()
    
    # Remove spaces
    clean_num = clean_num.replace(' ', '')
    
    # Remove any non-digit characters except leading zeros
    clean_num = re.sub(r'[^0-9]', '', clean_num)
    
    # Filter out OCR artifacts (too short numbers that are likely fragments)
    if len(clean_num) <= 3:
        return None  # Filter out OCR artifacts
    
    # Handle different cases
    if len(clean_num) == 7:
        # Perfect - already 7 digits
        return clean_num
    elif len(clean_num) == 6:
        # Missing leading zero
        return '0' + clean_num
    elif len(clean_num) == 4:
        # Appears to be truncated - need to add trailing zeros
        # e.g., "0022" should become "0022000"
        return clean_num + '000'
    elif len(clean_num) == 5:
        # Could be missing leading zero and trailing zeros
        return '0' + clean_num + '00'
    elif len(clean_num) > 7:
        # Too long - truncate to 7 digits
        return clean_num[:7]
    else:
        # Other cases
        return clean_num

def classify_word_by_coordinates(word: str, x_coord: float, zones: dict):
    """
    Classify a word based on coordinate zones and content patterns.
    Returns tuple: (Commodity Number, Description, Unit, Rate 1930, Rate Trade, Tariff Paragraph)
    """
    word = str(word).strip()
    
    # Initialize all columns as None
    commodity_num = None
    description = None
    unit = None
    rate_1930 = None
    rate_trade = None
    tariff_para = None
    
    # 1. Check pattern-based classification first
    if re.match(r'^00\d{2}\s?\d{3}$', word):
        commodity_num = word
    elif re.match(r'^(70[0-9]|71[0-9]|1558)$', word):
        tariff_para = word
    else:
        # 2. Use coordinate-based classification
        
        # Check each zone
        if 'commodity' in zones and zones['commodity']['min_x'] <= x_coord <= zones['commodity']['max_x']:
            commodity_num = word
        elif 'tariff' in zones and zones['tariff']['min_x'] <= x_coord <= zones['tariff']['max_x']:
            tariff_para = word
        elif 'unit' in zones and zones['unit']['min_x'] <= x_coord <= zones['unit']['max_x']:
            unit = word
        elif 'rate_1930' in zones and zones['rate_1930']['min_x'] <= x_coord <= zones['rate_1930']['max_x']:
            rate_1930 = word
        elif 'rate_trade' in zones and zones['rate_trade']['min_x'] <= x_coord <= zones['rate_trade']['max_x']:
            rate_trade = word
        elif 'description' in zones and zones['description']['min_x'] <= x_coord <= zones['description']['max_x']:
            description = word
        else:
            # Fallback - likely description
            description = word
    
    return commodity_num, description, unit, rate_1930, rate_trade, tariff_para

def main():
    """Main processing function."""
    # Load and preprocess
    df = load_and_preprocess(INPUT_CSV)
    
    # Analyze coordinate zones dynamically
    zones = analyze_coordinate_zones(df)
    
    # Classify each word based on coordinates
    df[['Commodity Number', 'Commodity Description', 'Unit of Quantity', 
        'Rate of Duty 1930', 'Rate of Duty Trade Agreement', 'Tariff Paragraph']] = df.apply(
        lambda row: pd.Series(classify_word_by_coordinates(row['Word'], row['TopLeft_X'], zones)), 
        axis=1
    )
    
    # Clean commodity numbers
    logging.info("=== CLEANING COMMODITY NUMBERS ===")
    commodity_mask = df['Commodity Number'].notna()
    initial_commodity_count = commodity_mask.sum()
    logging.info(f"Found {initial_commodity_count} raw commodity numbers")
    
    # Apply commodity number cleaning
    df.loc[commodity_mask, 'Commodity Number'] = df.loc[commodity_mask, 'Commodity Number'].apply(clean_commodity_number)
    
    # Count final valid commodity numbers
    final_commodity_count = df['Commodity Number'].notna().sum()
    filtered_count = initial_commodity_count - final_commodity_count
    logging.info(f"Cleaned to {final_commodity_count} valid commodity numbers")
    if filtered_count > 0:
        logging.info(f"Filtered out {filtered_count} OCR artifacts")
    
    # Keep coordinate information
    output_df = df[['Commodity Number', 'Commodity Description', 'Unit of Quantity', 
                   'Rate of Duty 1930', 'Rate of Duty Trade Agreement', 'Tariff Paragraph',
                   'TopLeft_X', 'TopLeft_Y', 'TopRight_X', 'TopRight_Y', 
                   'BottomRight_X', 'BottomRight_Y', 'BottomLeft_X', 'BottomLeft_Y',
                   'Confidence', 'Page']]
    
    # Save cleaned output
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Saved classified data with coordinates to {OUTPUT_CSV}")
    
    # Show summary
    logging.info(f"Total rows: {len(output_df)}")
    logging.info(f"Rows with Commodity Number: {output_df['Commodity Number'].notna().sum()}")
    logging.info(f"Rows with Description: {output_df['Commodity Description'].notna().sum()}")
    logging.info(f"Rows with Units: {output_df['Unit of Quantity'].notna().sum()}")
    logging.info(f"Rows with Rate 1930: {output_df['Rate of Duty 1930'].notna().sum()}")
    logging.info(f"Rows with Rate Trade: {output_df['Rate of Duty Trade Agreement'].notna().sum()}")
    logging.info(f"Rows with Tariff Paragraph: {output_df['Tariff Paragraph'].notna().sum()}")
    
    return output_df

if __name__ == "__main__":
    main()
