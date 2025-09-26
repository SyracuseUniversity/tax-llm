import pandas as pd
import re

# Updated to work with CSV files and new 6-column structure
CLEAN_CSV = r'new-work/output/cleaned_classified_words.csv'
FINAL_CSV = r'new-work/output/final-table.csv'

# Enhanced rate patterns to capture various formats including OCR artifacts
rate_patterns = [
    r'(\d+\.?\d*%)',           # Percentage rates (12%, 15%, etc.)
    r'(\d+½%)',                # Half percentages (12½%)
    r'(\d+%%\d*)',             # OCR artifacts like "12%%1"
    r'(Free\.?)',              # Free rates
    r'(\$\d+\.?\d*)',          # Dollar amounts
    r'(\d+\.\d+%)',            # Decimal percentages
    # OCR artifacts from page 28 data
    r'(2y21b)',                # OCR artifact for "2½¢ lb"
    r'(31b)',                  # OCR artifact for "3¢ lb"
    r'(1½)',                   # Half cent rates
    r'(\d+¢)',                 # Cent rates
    r'(\d+t)',                 # OCR "t" for "¢"
    r'(\d+plb)',               # OCR "plb" patterns
    r'(\d+each)',              # "each" rates
    r'(\$\d+\s+each)',         # Dollar each rates
]

def classify_rate_by_context(rate_text, context_text=""):
    """Classify rate as 1930 or trade agreement based on context."""
    rate_lower = rate_text.lower()
    context_lower = context_text.lower()
    
    # Trade agreement indicators
    trade_indicators = [
        'bound', 'gatt', 'agreement', 'u.k.', 'u. k.', 'can.', 'mex.', 
        'cuba', 'braz.', 'hond.', 'guat.', 'el salv.', 'c. rica',
        'para.', 'ecuad.', 'venz.', 'peru', 'arg.', 'neth.', 'colomb.'
    ]
    
    # Check if this is likely a trade agreement rate
    if any(indicator in context_lower for indicator in trade_indicators):
        return 'trade'
    elif '1930' in context_lower or 'tariff' in context_lower:
        return '1930'
    else:
        # Default to 1930 for simple rates without context
        return '1930'

def extract_rates_from_text(text):
    """Extract rate information from text using enhanced patterns."""
    if pd.isna(text) or not text:
        return ''
    
    text_str = str(text).strip()
    
    # Handle OCR artifacts first
    ocr_corrections = {
        '2y21b': '2½¢ lb',
        '31b': '3¢ lb',
        '1ye': '1½¢',
        '8plb': '8¢ lb',
        '2t': '2¢',
        '3t': '3¢',
        '4each': '4¢ each',
        '10lb': '10¢ lb',
        '6lb': '6¢ lb',
        '5lb': '5¢ lb',
        '7lb': '7¢ lb',
    }
    
    # Apply corrections
    for artifact, correction in ocr_corrections.items():
        if artifact in text_str:
            text_str = text_str.replace(artifact, correction)
    
    # Try each pattern
    for pattern in rate_patterns:
        match = re.search(pattern, text_str, re.IGNORECASE)
        if match:
            return match.group(0)
    
    # Check for dollar amounts with "each"
    dollar_each_match = re.search(r'\$(\d+\.?\d*)\s*each', text_str, re.IGNORECASE)
    if dollar_each_match:
        return f"${dollar_each_match.group(1)} each"
    
    return ''

def add_rates():
    """Add rate information to final table with new 6-column structure."""
    # Load data from CSV files
    df_clean = pd.read_csv(CLEAN_CSV)
    df_final = pd.read_csv(FINAL_CSV)
    
    # Ensure all expected columns exist
    expected_columns = [
        'SCHEDULE A COMMODITY NUMBER',
        'COMMODITY DESCRIPTION AND ECONOMIC CLASS', 
        'UNIT OF QUANTITY',
        'RATE OF DUTY 1930',
        'RATE OF DUTY TRADE AGREEMENT',
        'TARIFF PARAGRAPH'
    ]
    
    # Add missing columns if they don't exist
    for col in expected_columns:
        if col not in df_final.columns:
            df_final[col] = ''
    
    # Create mappings for rates based on coordinate proximity and Y-grouping
    commodity_rates_1930 = {}
    commodity_rates_trade = {}
    
    # Group data by Y-coordinate proximity to reconstruct logical rows
    tolerance = 30  # Y-coordinate tolerance for grouping
    
    # Sort by Y coordinate
    df_sorted = df_clean.sort_values(['TopLeft_Y', 'TopLeft_X']).reset_index(drop=True)
    
    # Group rows by Y proximity
    current_y_group = []
    current_y = None
    y_groups = []
    
    for _, row in df_sorted.iterrows():
        y_coord = row['TopLeft_Y']
        
        if current_y is None or abs(y_coord - current_y) <= tolerance:
            current_y_group.append(row)
            if current_y is None:
                current_y = y_coord
        else:
            if current_y_group:
                y_groups.append(current_y_group)
            current_y_group = [row]
            current_y = y_coord
    
    if current_y_group:
        y_groups.append(current_y_group)
    
    # Process each Y-group to extract rates and associate with commodities
    for group in y_groups:
        # Find commodity number in this group
        commodity_num = None
        for row in group:
            if pd.notna(row['Commodity Number']) and str(row['Commodity Number']).strip():
                commodity_num = str(row['Commodity Number']).strip()
                break
        
        if not commodity_num:
            continue
            
        # Extract rates from this group based on X coordinates
        group_1930_rates = []
        group_trade_rates = []
        
        for row in group:
            # Check Rate of Duty 1930 column first
            rate_1930 = str(row.get('Rate of Duty 1930', '')).strip()
            if rate_1930 and rate_1930 != 'nan':
                rate = extract_rates_from_text(rate_1930)
                if rate:
                    group_1930_rates.append(rate)
            
            # Check Rate of Duty Trade Agreement column
            rate_trade = str(row.get('Rate of Duty Trade Agreement', '')).strip()
            if rate_trade and rate_trade != 'nan':
                rate = extract_rates_from_text(rate_trade)
                if rate:
                    group_trade_rates.append(rate)
            
            # Also check general description for rate patterns
            word = str(row.get('Commodity Description', '')).strip()
            x_pos = row['TopLeft_X']
            
            # Extract rate if text contains rate information
            rate = extract_rates_from_text(word)
            if not rate:
                continue
            
            # Classify rate based on X position and context
            if 1350 <= x_pos <= 1700:
                # This is likely a 1930 rate
                rate_type = classify_rate_by_context(rate, word)
                if rate_type == '1930' or x_pos < 1550:
                    group_1930_rates.append(rate)
                else:
                    group_trade_rates.append(rate)
            elif 1700 <= x_pos <= 2150:
                # This is likely a trade agreement rate
                group_trade_rates.append(rate)
            else:
                # Default classification based on content
                rate_type = classify_rate_by_context(rate, word)
                if rate_type == 'trade':
                    group_trade_rates.append(rate)
                else:
                    group_1930_rates.append(rate)
        
        # Combine rates for this commodity
        if group_1930_rates:
            commodity_rates_1930[commodity_num] = ' '.join(group_1930_rates)
        if group_trade_rates:
            commodity_rates_trade[commodity_num] = ' '.join(group_trade_rates)
    
    # Also check for standalone rate information and try to associate with nearest commodity
    for _, row in df_clean.iterrows():
        word = str(row.get('Commodity Description', '')).strip()
        x_pos = row['TopLeft_X']
        y_pos = row['TopLeft_Y']
        
        # Skip if already processed in groups
        if pd.notna(row['Commodity Number']) and str(row['Commodity Number']).strip():
            continue
            
        rate = extract_rates_from_text(word)
        if not rate:
            continue
        
        # Find nearest commodity number (within reasonable Y distance)
        nearest_commodity = None
        min_distance = float('inf')
        
        for _, commodity_row in df_clean.iterrows():
            if pd.notna(commodity_row['Commodity Number']) and str(commodity_row['Commodity Number']).strip():
                commodity_y = commodity_row['TopLeft_Y']
                distance = abs(y_pos - commodity_y)
                if distance < min_distance and distance <= 50:  # Within 50 pixels
                    min_distance = distance
                    nearest_commodity = str(commodity_row['Commodity Number']).strip()
        
        if nearest_commodity:
            # Classify rate based on X position
            if 1350 <= x_pos <= 1700:
                if nearest_commodity not in commodity_rates_1930:
                    commodity_rates_1930[nearest_commodity] = rate
                else:
                    commodity_rates_1930[nearest_commodity] += f" {rate}"
            elif 1700 <= x_pos <= 2150:
                if nearest_commodity not in commodity_rates_trade:
                    commodity_rates_trade[nearest_commodity] = rate
                else:
                    commodity_rates_trade[nearest_commodity] += f" {rate}"
    
    # Update final table with extracted rates
    updated_1930 = 0
    updated_trade = 0
    
    for idx, row in df_final.iterrows():
        commodity_num_formatted = str(row['SCHEDULE A COMMODITY NUMBER']).strip()
        
        # Convert formatted number to match source data format (like we did in previous scripts)
        clean_num = commodity_num_formatted.replace(' ', '')
        if len(clean_num) == 7:
            without_leading_zero = clean_num[1:]
            without_trailing_zeros = without_leading_zero.rstrip('0')
            
            if len(without_trailing_zeros) == 2:
                converted_num = without_trailing_zeros + '000'
            elif len(without_trailing_zeros) == 3:
                converted_num = without_trailing_zeros + '00'
            elif len(without_trailing_zeros) == 4:
                converted_num = without_trailing_zeros + '0'
            else:
                converted_num = without_trailing_zeros
                
            mapping_key = f"{converted_num}.0"
        else:
            mapping_key = clean_num
        
        # Try both formats for matching
        for num_format in [commodity_num_formatted, clean_num, mapping_key]:
            # Add 1930 rates
            if num_format in commodity_rates_1930:
                df_final.at[idx, 'RATE OF DUTY 1930'] = commodity_rates_1930[num_format]
                updated_1930 += 1
                break
        
        for num_format in [commodity_num_formatted, clean_num, mapping_key]:
            # Add trade agreement rates  
            if num_format in commodity_rates_trade:
                df_final.at[idx, 'RATE OF DUTY TRADE AGREEMENT'] = commodity_rates_trade[num_format]
                updated_trade += 1
                break
    
    # Reorder columns to match new structure
    df_final = df_final[expected_columns]
    
    # Save updated data
    df_final.to_csv(FINAL_CSV, index=False)
    
    print(f"Updated rates in {FINAL_CSV}")
    print(f"1930 rates updated: {updated_1930}")
    print(f"Trade agreement rates updated: {updated_trade}")
    print(f"Found 1930 rates: {list(set(commodity_rates_1930.values()))}")
    print(f"Found trade rates: {list(set(commodity_rates_trade.values()))}")
    print(f"File now uses new 6-column structure: {expected_columns}")

def main():
    """
    Entry point for the script.
    """
    add_rates()

if __name__ == "__main__":
    main()
