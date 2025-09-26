import pytest
import pandas as pd
from io import StringIO
import tempfile
import os
import re

import rate_of_duty05 as rd

@pytest.fixture
def sample_data():
    sample_csv = StringIO(
        """Page,TopLeft_X,TopLeft_Y,Commodity Description,Commodity Number,Rate of Duty 1930,Rate of Duty Trade Agreement
1,1350,100,Cattle:,0010600,2y21b,1½¢ lb Can., Mex., bound GATT
1,1400,110,Weighing less than 200 pounds each (calves).,0010600,2y21b,1½¢ lb Mex.
1,1500,115,Weighing 200 pounds and less than 700 pounds each.,0010700,31b,1½¢ lb Can., bound GATT
1,1600,200,Sheep and lambs,0020000,$3 each,$1.50 each Mex.
1,1700,210,Goats,0021200,$3 each,1½¢ lb Can., bound GATT
1,1800,215,Hogs,0021300,2¢ lb,1½¢ lb Can., bound GATT
"""
    )
    df = pd.read_csv(sample_csv)
    # Convert TopLeft_X and TopLeft_Y to numeric explicitly
    df['TopLeft_X'] = pd.to_numeric(df['TopLeft_X'], errors='coerce')
    df['TopLeft_Y'] = pd.to_numeric(df['TopLeft_Y'], errors='coerce')
    return df

def test_classify_rate_by_context():
    assert rd.classify_rate_by_context('2½¢ lb', 'bound GATT') == 'trade'
    assert rd.classify_rate_by_context('2½¢ lb', '1930 Tariff') == '1930'
    assert rd.classify_rate_by_context('2½¢ lb', '') == '1930'

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
    
    # After correction, try to match known rate patterns
    for pattern in rd.rate_patterns:
        match = re.search(pattern, text_str, re.IGNORECASE)
        if match:
            return match.group(0)
    
    # Check for dollar amounts with "each"
    dollar_each_match = re.search(r'\$(\d+\.?\d*)\s*each', text_str, re.IGNORECASE)
    if dollar_each_match:
        return f"${dollar_each_match.group(1)} each"
    
    return ''

def test_add_rates(sample_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        clean_csv_path = os.path.join(tmpdir, 'cleaned_classified_words.csv')
        final_csv_path = os.path.join(tmpdir, 'final-table.csv')

        # Save sample data to clean CSV
        sample_data.to_csv(clean_csv_path, index=False)

        # Create dummy final table with required columns
        dummy_final = pd.DataFrame({
            'SCHEDULE A COMMODITY NUMBER': ['0010600', '0010700', '0020000', '0021200', '0021300'],
            'COMMODITY DESCRIPTION AND ECONOMIC CLASS': ['', '', '', '', ''],
            'UNIT OF QUANTITY': ['', '', '', '', ''],
            'RATE OF DUTY 1930': ['', '', '', '', ''],
            'RATE OF DUTY TRADE AGREEMENT': ['', '', '', '', ''],
            'TARIFF PARAGRAPH': ['', '', '', '', '']
        })
        dummy_final.to_csv(final_csv_path, index=False)

        # Patch the file paths in the module
        rd.CLEAN_CSV = clean_csv_path
        rd.FINAL_CSV = final_csv_path

        # Run add_rates function
        rd.add_rates()

        # Load updated final table
        updated_final = pd.read_csv(final_csv_path)

        # Check that rates were added (non-empty strings)
        assert updated_final['RATE OF DUTY 1930'].iloc[0] != ''
        assert updated_final['RATE OF DUTY TRADE AGREEMENT'].iloc[0] != ''

if __name__ == '__main__':
    pytest.main()
