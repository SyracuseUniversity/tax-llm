import json
import re
import pandas as pd
import numpy as np
from collections import Counter


# Configuration
RAW_CSV = r'new-work/output/ocr_word_coords.csv'  # Raw OCR data
CLEAN_CSV = r'new-work/output/cleaned_classified_words.csv'  # Path to your clean words CSV
OUTPUT_CSV = r'new-work/output/final-table.csv'

COLUMNS = [
    'SCHEDULE A COMMODITY NUMBER',
    'COMMODITY DESCRIPTION AND ECONOMIC CLASS',
    'UNIT OF QUANTITY',
    'RATE OF DUTY 1930',
    'RATE OF DUTY TRADE AGREEMENT',
    'TARIFF PARAGRAPH'
]

# Enhanced regex patterns for commodity numbers
commodity_patterns = [
    re.compile(r'\b(\d{4})\s*(\d{3})\b'),  # 4 digits + 3 digits with optional space
    re.compile(r'\b(\d{3})\s*(\d{4})\b'),  # 3 digits + 4 digits with optional space  
    re.compile(r'\b(\d{7})\b'),            # 7 digits together
    re.compile(r'\b(\d{4})\s+(\d{2})\s*(\d{1})\b'),  # 4 + 2 + 1 format
]

# OCR correction patterns for commodity numbers
ocr_corrections = {
    'O': '0',   # O to 0
    'o': '0',   # o to 0
    'l': '1',   # l to 1 (in numeric context)
    'I': '1',   # I to 1
    'S': '5',   # S to 5 (sometimes)
    'G': '6',   # G to 6 (sometimes)
    'B': '8',   # B to 8 (sometimes)
}


def clean_ocr_artifacts_in_number(text):
    """Clean OCR artifacts in commodity numbers."""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Apply OCR corrections for numbers
    for incorrect, correct in ocr_corrections.items():
        # Only apply in numeric context - when surrounded by digits or at start/end
        text = re.sub(rf'(?<=\d){incorrect}(?=\d)', correct, text)
        text = re.sub(rf'^{incorrect}(?=\d)', correct, text)
        text = re.sub(rf'(?<=\d){incorrect}$', correct, text)
    
    return text

def extract_commodity_number(text):
    """Extract and format commodity number from text using multiple patterns."""
    if not text or pd.isna(text):
        return None
    
    text = clean_ocr_artifacts_in_number(str(text))
    
    # Try each pattern
    for pattern in commodity_patterns:
        match = pattern.search(text)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                # Two groups found
                if len(groups[0]) == 4 and len(groups[1]) == 3:
                    commodity_num = f"{groups[0]} {groups[1]}"
                elif len(groups[0]) == 3 and len(groups[1]) == 4:
                    commodity_num = f"{groups[1]} {groups[0]}"  # Swap to correct order
                else:
                    continue
            elif len(groups) == 1 and len(groups[0]) == 7:
                # 7-digit number
                commodity_num = f"{groups[0][:4]} {groups[0][4:]}"
            elif len(groups) == 3:
                # 4+2+1 format, combine last two
                commodity_num = f"{groups[0]} {groups[1]}{groups[2]}"
            else:
                continue
            
            # Validate commodity number range
            if is_valid_commodity_number(commodity_num):
                return commodity_num
    
    # If no pattern matched but looks like commodity number
    digits_only = re.sub(r'[^\d]', '', text)
    if len(digits_only) == 7:
        commodity_num = f"{digits_only[:4]} {digits_only[4:]}"
        if is_valid_commodity_number(commodity_num):
            return commodity_num
    elif len(digits_only) == 6:
        commodity_num = f"{digits_only[:3]} {digits_only[3:]}"
        if is_valid_commodity_number(commodity_num):
            return commodity_num
    
    return None

def is_valid_commodity_number(commodity_num):
    """Validate if a commodity number is in the expected range for tariff schedules."""
    if not commodity_num:
        return False
    
    try:
        # Extract the first 4 digits for validation
        first_part = commodity_num.split()[0]
        if len(first_part) == 4 and first_part.isdigit():
            num = int(first_part)
            # Valid tariff commodity numbers typically start with 00-09
            # Numbers like 1000, 2007 are likely OCR errors from page references
            if 1 <= num <= 999:  # Valid range for tariff commodities
                return True
        return False
    except (ValueError, IndexError):
        return False

def analyze_coordinate_patterns(df):
    """Analyze coordinate patterns to improve commodity number detection."""
    # Find X coordinates where commodity numbers typically appear
    commodity_rows = df[df['Commodity_Number'].notna() & (df['Commodity_Number'] != '')]
    
    if len(commodity_rows) > 0:
        x_coords = commodity_rows['TopLeft_X'].values
        x_mode = Counter(x_coords).most_common(1)[0][0]
        x_tolerance = 20  # pixels
        
        print(f"Commodity numbers typically at X coordinate: {x_mode} (±{x_tolerance})")
        return x_mode, x_tolerance
    
    return 240, 20  # Default based on observed pattern

def enhance_commodity_detection(df):
    """Enhance commodity number detection using coordinate analysis."""
    x_mode, x_tolerance = analyze_coordinate_patterns(df)
    
    # Look for potential commodity numbers in the coordinate range
    potential_commodities = df[
        (df['TopLeft_X'] >= x_mode - x_tolerance) & 
        (df['TopLeft_X'] <= x_mode + x_tolerance)
    ].copy()
    
    enhanced_commodities = []
    
    for _, row in potential_commodities.iterrows():
        # Check both existing Commodity_Number and Description columns
        candidates = [
            row.get('Commodity_Number', ''),
            row.get('Description', ''),
            row.get('Word', '') if 'Word' in df.columns else ''
        ]
        
        for candidate in candidates:
            if candidate and str(candidate) != 'nan':
                commodity_num = extract_commodity_number(str(candidate))
                if commodity_num:
                    enhanced_commodities.append({
                        'commodity': commodity_num,
                        'y_coord': row['TopLeft_Y'],
                        'confidence': row.get('Confidence', 0.5)
                    })
                    break
    
    return enhanced_commodities

def extract_commodity_numbers_from_csv(csv_path):
    """
    Extract commodity numbers from the cleaned classified words CSV.
    Returns formatted commodity numbers for the first column.
    """
    df = pd.read_csv(csv_path)
    
    # Get all rows with commodity numbers (already cleaned by enhanced cleaning script)
    commodity_rows = df[df['Commodity Number'].notna()].copy()
    
    print(f"Found {len(commodity_rows)} rows with commodity numbers")
    
    # Extract unique commodity numbers
    commodity_numbers = []
    seen = set()
    
    for _, row in commodity_rows.iterrows():
        commodity_num = str(row['Commodity Number']).strip()
        if commodity_num and commodity_num != 'nan' and commodity_num not in seen:
            # Clean and format the commodity number
            clean_num = re.sub(r'[^\d]', '', commodity_num)  # Keep only digits
            
            if len(clean_num) == 7:
                # Format as "0010 600"
                formatted_num = f"{clean_num[:4]} {clean_num[4:]}"
            elif len(clean_num) == 6:
                # Add leading zero and format as "0010 600"  
                formatted_num = f"0{clean_num[:3]} {clean_num[3:]}"
            else:
                formatted_num = clean_num
            
            commodity_numbers.append(formatted_num)
            seen.add(commodity_num)
    
    # Sort by the numeric value to ensure proper order
    def sort_key(commodity_num):
        # Extract numeric part for sorting
        digits = re.sub(r'[^\d]', '', commodity_num)
        return int(digits) if digits else 0
    
    commodity_numbers.sort(key=sort_key)
    
    print(f"Extracted {len(commodity_numbers)} unique commodity numbers")
    print(f"Sample numbers: {commodity_numbers[:5]}")
    
    return commodity_numbers



def save_to_new_csv(commodity_numbers, output_csv, columns):
    """
    Save commodity numbers to a new CSV file with all column headers.
    Creates empty columns for future data population.
    """
    df = pd.DataFrame(columns=columns)
    
    # Fill first column with commodity numbers
    df[columns[0]] = commodity_numbers
    
    # Initialize other columns as empty (to be filled later)
    for col in columns[1:]:
        df[col] = ""
    
    df.to_csv(output_csv, index=False)




def main():
    commodity_numbers = extract_commodity_numbers_from_csv(CLEAN_CSV)
    save_to_new_csv(commodity_numbers, OUTPUT_CSV, COLUMNS)
    print(f"Created {OUTPUT_CSV} with {len(commodity_numbers)} commodity numbers.")
    print("Column structure created:")
    for i, col in enumerate(COLUMNS, 1):
        status = "POPULATED" if i == 1 else "EMPTY"
        print(f"  {i}. {col} - {status}")

if __name__ == "__main__":
    main()
