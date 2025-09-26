import pandas as pd
import re

CLEAN_CSV = r'new-work/output/cleaned_classified_words.csv'
FINAL_CSV = r'new-work/output/final-table.csv'

# Pattern to identify units of quantity
unit_patterns = [
    r'\b(lb|lbs|pound|pounds)\b',
    r'\b(kg|kilogram|kilograms)\b',
    r'\b(ton|tons|tonne|tonnes)\b',
    r'\b(gal|gallon|gallons)\b',
    r'\b(qt|quart|quarts)\b',
    r'\b(pt|pint|pints)\b',
    r'\b(oz|ounce|ounces)\b',
    r'\b(cu\s?ft|cubic\s+feet?)\b',
    r'\b(sq\s?ft|square\s+feet?)\b',
    r'\b(linear\s+feet?|lin\s?ft)\b',
    r'\b(yard|yards|yd)\b',
    r'\b(meter|metres?|m)\b',
    r'\b(each|ea)\b',
    r'\b(dozen|doz)\b',
    r'\b(gross)\b',
    r'\b(case|cases)\b',
    r'\b(box|boxes)\b',
    r'\b(bag|bags)\b',
    r'\b(bale|bales)\b',
    r'\b(bundle|bundles)\b',
    r'\b(head|hd)\b',  # For livestock
    r'\b(no|number)\b',  # Sometimes "No" indicates quantity units
]

# Context-aware patterns for specific commodities
context_patterns = {
    'livestock': {
        'patterns': [r'cattle', r'sheep', r'lamb', r'swine', r'pig', r'horse', r'animal'],
        'units': ['No', 'Head']
    },
    'weight_based': {
        'patterns': [r'meat', r'beef', r'pork', r'carcass', r'dressed', r'fresh', r'frozen'],
        'units': ['Lb', 'Cwt']
    },
    'volume_based': {
        'patterns': [r'liquid', r'oil', r'milk', r'beverage'],
        'units': ['Gal', 'Qt']
    },
    'count_based': {
        'patterns': [r'eggs', r'birds', r'chickens', r'turkeys', r'ducks', r'poultry'],
        'units': ['No', 'Doz']
    }
}

def extract_unit_from_text(text, context_description=''):
    """
    Extract unit of quantity from description text using patterns and context.
    """
    if pd.isna(text) or not text:
        return ''
    
    text_lower = str(text).lower()
    context_lower = str(context_description).lower() if context_description else ''
    combined_text = f"{text_lower} {context_lower}".strip()
    
    # Check context-specific patterns
    for context_type, context_info in context_patterns.items():
        for pattern in context_info['patterns']:
            if re.search(pattern, combined_text, re.IGNORECASE):
                for unit in context_info['units']:
                    unit_pattern = rf'\b{re.escape(unit.lower())}\b'
                    if re.search(unit_pattern, combined_text, re.IGNORECASE):
                        return unit
                return context_info['units'][0]  # Default to the first unit if no match

    # General pattern matching
    for pattern in unit_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            matched_text = match.group(0).lower()
            return {
                'lb': 'Lb', 'lbs': 'Lb', 'pound': 'Lb', 'pounds': 'Lb',
                'no': 'No', 'number': 'No',
                'each': 'No', 'ea': 'No',
                'head': 'Head', 'hd': 'Head'
            }.get(matched_text, matched_text.capitalize())
    
    return ''

def infer_unit_from_commodity_type(commodity_num, description):
    """
    Infer unit based on commodity number patterns and description context.
    """
    if not commodity_num or not description:
        return ''
    
    description_lower = str(description).lower()
    
    # Livestock commodities typically use "No"
    if any(word in description_lower for word in ['cattle', 'sheep', 'lamb', 'swine', 'pig', 'horse', 'live']):
        return 'No'
    
    # Meat products typically use "Lb"
    if any(word in description_lower for word in ['meat', 'beef', 'pork', 'carcass', 'dressed', 'fresh', 'frozen']):
        return 'Lb'
    
    # Default fallback based on commodity number ranges
    try:
        num = int(commodity_num.replace(' ', ''))
        if 100 <= num <= 199:  # Livestock range (example)
            return 'No'
        elif 200 <= num <= 299:  # Meat products range (example)
            return 'Lb'
    except (ValueError, AttributeError):
        pass
    
    return ''

def add_units():
    """
    Main function to process and add units of quantity to the final table.
    """
    df_clean = pd.read_csv(CLEAN_CSV)
    df_final = pd.read_csv(FINAL_CSV)
    
    commodity_units = {}
    commodity_descriptions = {}

    # Collect descriptions for each commodity
    for _, row in df_clean.iterrows():
        commodity_num = str(row.get('Commodity Number', '')).strip()
        description = str(row.get('Commodity Description', '')).strip()
        if commodity_num and description:
            commodity_descriptions.setdefault(commodity_num, []).append(description)
    
    # Extract units with context awareness
    for _, row in df_clean.iterrows():
        description = str(row.get('Commodity Description', '')).strip()
        y_coord = row.get('TopLeft_Y', 0)
        tolerance = 30
        
        nearby_commodities = df_clean[
            (df_clean['Commodity Number'].notna()) &
            (abs(df_clean['TopLeft_Y'] - y_coord) <= tolerance)
        ]
        
        if not nearby_commodities.empty:
            commodity_num = str(nearby_commodities.iloc[0]['Commodity Number']).strip()
            context_descriptions = commodity_descriptions.get(commodity_num, [])
            full_context = ' '.join(context_descriptions)
            unit = extract_unit_from_text(description, full_context) or infer_unit_from_commodity_type(commodity_num, full_context)
            if unit:
                commodity_units[commodity_num] = unit
    
    # Update final table
    for idx, row in df_final.iterrows():
        commodity_num = str(row.get('SCHEDULE A COMMODITY NUMBER', '')).strip()
        unit = commodity_units.get(commodity_num, 'No')
        df_final.at[idx, 'UNIT OF QUANTITY'] = unit
    
    df_final.to_csv(FINAL_CSV, index=False)
    print(f"Updated {len(commodity_units)} units in {FINAL_CSV}")

def main():
    """
    Entry point for the script.
    """
    add_units()

if __name__ == "__main__":
    main()