import pandas as pd
import re
from typing import List, Dict, Any
from pathlib import Path

# =======================================
# CONFIGURATION
# =======================================
BASE_DIR = Path(__file__).parent  # folder containing your script

INPUT_CSV = BASE_DIR / "output/cleaned_classified_words.csv"
OUTPUT_CSV = BASE_DIR / "output/cleaned_flattened_table.csv"

# =======================================
# COLUMN DEFINITIONS
# =======================================
cols = [
    "Commodity Number",
    "Commodity Description",
    "Unit of Quantity",
    "Rate of Duty 1930 Tariff Act",
    "Rate of Duty Trade Agreement",
    "Tariff Paragraph",
]

# =======================================
# PATTERN DEFINITIONS
# =======================================
commodity_re = re.compile(r"^\s*\d{4}\s?\d{3}\s*$")

def is_commodity_number(text: str) -> bool:
    """Check if text is a commodity number"""
    return bool(commodity_re.match(text)) if text else False

def is_header(text: str) -> bool:
    """Detect headers such as all-caps phrases, section labels, or title-like words ending in ':'."""
    if not text or not isinstance(text, str):
        return False
    
    t = text.strip()

    # 1. Title-case or word ending with colon (e.g., "Cattle:")
    if t.endswith(":"):
        return True

    # 2. All-uppercase even if it's a single long token (commas, parentheses OK)
    if re.fullmatch(r"[A-Z][A-Z(),.\-_/]*", t):
        return True

    # 3. Original rule: multiple uppercase words
    clean = t.replace("*", "")
    words = clean.split()
    if len(words) >= 2:
        upper_count = sum(1 for w in words if w.isupper() and len(w) > 1)
        return upper_count >= 2
    
    return False

def clean_text(s: str) -> str:
    """Clean text by removing extra whitespace"""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s.strip())
    return s.strip()

# =======================================
# PROCESSING LOGIC
# =======================================
def process_row_by_row(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Process data row by row, cell by cell
    """
    output_rows = []
    current_record = {col: "" for col in cols}
    current_commodity = ""
    
    for _, row in df.iterrows():
        row_values = {col: clean_text(str(row[col])) for col in cols}
        
        # Check if this row contains a commodity number
        commodity_num = row_values["Commodity Number"]
        has_commodity = is_commodity_number(commodity_num)
        
        # Check if this row contains header text in any column
        has_header = is_header(row_values["Commodity Description"])

        
        # CASE 1: Header row - create separate header record
        if has_header and not has_commodity:
            # Save current record if it exists
            if any(current_record.values()):
                output_rows.append(current_record.copy())
                current_record = {col: "" for col in cols}
            
            # Create header record (only commodity description filled)
            header_record = {col: "" for col in cols}
            # Collect all header text from the row
            header_texts = []
            for col in cols:
                if col != "Commodity Number" and row_values[col]:
                    header_texts.append(row_values[col])
            if header_texts:
                header_record["Commodity Description"] = " ".join(header_texts)
                output_rows.append(header_record)
            continue
        
        # CASE 2: Commodity number row - start new record
        if has_commodity:
            # Save current record if it exists
            if any(current_record.values()):
                output_rows.append(current_record.copy())
            
            # Start new record
            current_record = {col: "" for col in cols}
            current_record["Commodity Number"] = commodity_num
            current_commodity = commodity_num
            
            # Add description if present in same row
            desc = row_values["Commodity Description"]
            if desc and not is_header(desc):
                current_record["Commodity Description"] = desc
            
            # Add other columns if present
            for col in cols[2:]:  # Skip Commodity Number and Description
                if row_values[col]:
                    current_record[col] = row_values[col]
            
            # If this record has data, save it immediately and start fresh
            if any(current_record[col] for col in cols if col != "Commodity Number"):
                output_rows.append(current_record.copy())
                current_record = {col: "" for col in cols}
                current_record["Commodity Number"] = current_commodity
            continue
        
        # CASE 3: Data row (no commodity, no header) - add to current record
        if current_commodity:  # Only process if we have an active commodity
            # For description - append to existing
            desc = row_values["Commodity Description"]
            if desc and not is_header(desc):
                if current_record["Commodity Description"]:
                    current_record["Commodity Description"] += " " + desc
                else:
                    current_record["Commodity Description"] = desc
            
            # For Unit of Quantity - handle multiple units
            unit = row_values["Unit of Quantity"]
            if unit:
                if current_record["Unit of Quantity"]:
                    # Multiple units - save current and start new line
                    output_rows.append(current_record.copy())
                    current_record = {col: "" for col in cols}
                    current_record["Commodity Number"] = current_commodity
                    current_record["Commodity Description"] = ""  # Don't repeat description
                    current_record["Unit of Quantity"] = unit
                else:
                    current_record["Unit of Quantity"] = unit
            
            # For Rate of Duty 1930 - handle multiple rates
            rate1930 = row_values["Rate of Duty 1930 Tariff Act"]
            if rate1930:
                if current_record["Rate of Duty 1930 Tariff Act"]:
                    # Multiple rates - save current and start new line
                    output_rows.append(current_record.copy())
                    current_record = {col: "" for col in cols}
                    current_record["Commodity Number"] = current_commodity
                    current_record["Commodity Description"] = ""
                    current_record["Rate of Duty 1930 Tariff Act"] = rate1930
                else:
                    current_record["Rate of Duty 1930 Tariff Act"] = rate1930
            
            # For Rate of Duty Trade Agreement - append or new line
            trade_rate = row_values["Rate of Duty Trade Agreement"]
            if trade_rate:
                if current_record["Rate of Duty Trade Agreement"]:
                    # Append if short, otherwise new line
                    if len(current_record["Rate of Duty Trade Agreement"]) + len(trade_rate) < 100:
                        current_record["Rate of Duty Trade Agreement"] += " " + trade_rate
                    else:
                        output_rows.append(current_record.copy())
                        current_record = {col: "" for col in cols}
                        current_record["Commodity Number"] = current_commodity
                        current_record["Commodity Description"] = ""
                        current_record["Rate of Duty Trade Agreement"] = trade_rate
                else:
                    current_record["Rate of Duty Trade Agreement"] = trade_rate
            
            # For Tariff Paragraph - replace existing
            tariff = row_values["Tariff Paragraph"]
            if tariff:
                current_record["Tariff Paragraph"] = tariff
    
    # Save the final record
    if any(current_record.values()):
        output_rows.append(current_record)
    
    return output_rows

def remove_duplicate_commodity_numbers(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Remove duplicate commodity numbers from subsequent lines of the same commodity group
    """
    if not rows:
        return []
    
    cleaned_rows = []
    last_commodity = ""
    
    for row in rows:
        current_row = row.copy()
        current_commodity = current_row["Commodity Number"]
        
        # If this is the same commodity as last row, clear the commodity number
        if current_commodity and current_commodity == last_commodity:
            current_row["Commodity Number"] = ""
        
        # Update last commodity if this row has one
        if current_commodity:
            last_commodity = current_commodity
        elif current_row["Commodity Description"] and is_header(current_row["Commodity Description"]):
            # Reset last commodity when we encounter a header
            last_commodity = ""
        
        cleaned_rows.append(current_row)
    
    return cleaned_rows

def remove_duplicate_descriptions(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Remove duplicate commodity descriptions from subsequent lines
    """
    if not rows:
        return []
    
    cleaned_rows = []
    last_description = ""
    last_commodity = ""
    
    for row in rows:
        current_row = row.copy()
        current_desc = current_row["Commodity Description"]
        current_commodity = current_row["Commodity Number"]
        
        # Only remove description if it's the same commodity and same description
        if (current_desc == last_description and 
            current_commodity == last_commodity and 
            current_commodity and  # Only for commodity rows, not headers
            any(current_row[col] for col in cols[2:])):  # Has data in other columns
            current_row["Commodity Description"] = ""
        
        # Update last description and commodity
        if current_desc and current_row["Commodity Description"]:  # Only if not cleared
            last_description = current_desc
            last_commodity = current_commodity
        elif not current_commodity and is_header(current_desc):  # Header row
            last_description = ""
            last_commodity = ""
        
        cleaned_rows.append(current_row)
    
    return cleaned_rows

# =======================================
# MAIN EXECUTION
# =======================================
def main():
    # Load raw data
    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    
    # Normalize column names
    col_map = {}
    available = {c.lower().strip(): c for c in df.columns}
    for desired in cols:
        key = desired.lower().strip()
        if key in available:
            col_map[desired] = available[key]
        else:
            df[desired] = ""
            col_map[desired] = desired
    
    df = df[[col_map[c] for c in cols]]
    df.columns = cols
    
    print(f"Loaded {len(df)} rows from input CSV")
    
    # Process data row by row
    output_rows = process_row_by_row(df)
    print(f"Generated {len(output_rows)} initial output rows")
    
    # Remove duplicate commodity numbers
    output_rows = remove_duplicate_commodity_numbers(output_rows)
    print(f"After removing duplicate commodity numbers: {len(output_rows)} rows")
    
    # Remove duplicate descriptions
    final_rows = remove_duplicate_descriptions(output_rows)
    print(f"After removing duplicate descriptions: {len(final_rows)} final rows")
    
    # Create final DataFrame
    final_df = pd.DataFrame(final_rows, columns=cols)
    
    # Remove completely empty rows
    final_df = final_df[final_df.astype(str).any(axis=1)]
    print(f"After removing empty rows: {len(final_df)} rows")
    
    # Save output
    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"✅ Cleaned flattened CSV saved to: {OUTPUT_CSV}")
    
    # Print sample for verification
    print("\nSample of first 10 rows:")
    print(final_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()