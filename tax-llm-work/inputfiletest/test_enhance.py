import os
import tempfile
import pandas as pd
import pytest
from enhanced_clean import (
    main,
    load_and_preprocess,
    analyze_coordinate_zones,
    classify_word_by_coordinates,
    clean_commodity_number
)

# Sample OCR CSV content simulating page_28.png
SAMPLE_OCR_CSV = """Word,TopLeft_X,TopLeft_Y,TopRight_X,TopRight_Y,BottomRight_X,BottomRight_Y,BottomLeft_X,BottomLeft_Y,Confidence,Page
Group,50,50,90,50,90,70,50,70,0.98,28
00,100,50,120,50,120,70,100,70,0.95,28
ANIMALS,150,50,220,50,220,70,150,70,0.97,28
AND,230,50,260,50,260,70,230,70,0.96,28
ANIMAL,270,50,320,50,320,70,270,70,0.94,28
PRODUCTS,,, ,
EDIBLE,350,50,400,50,400,70,350,70,0.96,28
0010600,50,150,120,150,120,170,50,170,0.99,28
Cattle,150,150,220,150,220,170,150,170,0.98,28
Weighing,150,180,220,180,220,200,150,200,0.97,28
less,230,180,270,180,270,200,230,200,0.96,28
than,280,180,320,180,320,200,280,200,0.95,28
200,330,180,370,180,370,200,330,200,0.94,28
pounds,380,180,440,180,440,200,380,200,0.93,28
each,450,180,490,180,490,200,450,200,0.92,28
No,50,220,70,220,70,240,50,240,0.99,28
Lb,80,220,100,220,100,240,80,240,0.98,28
2 1/2,150,220,200,220,200,240,150,240,0.97,28
lb,210,220,230,220,230,240,210,240,0.96,28
1 1/2,250,220,290,220,290,240,250,240,0.95,28
lb,300,220,320,220,320,240,300,240,0.94,28
Can.,330,220,370,220,370,240,330,240,0.93,28
Mex.,380,220,420,220,420,240,380,240,0.92,28
bound,430,220,470,220,470,240,430,240,0.91,28
GATT,480,220,520,220,520,240,480,240,0.90,28
701,600,220,630,220,630,240,600,240,0.99,28
"""

def test_clean_commodity_number():
    # Valid 7-digit number
    assert clean_commodity_number('0010600') == '0010600'
    # 6-digit number missing leading zero
    assert clean_commodity_number('010600') == '0010600'
    # 4-digit truncated number
    assert clean_commodity_number('0022') == '0022000'
    # 5-digit number
    assert clean_commodity_number('12345') == '01234500'
    # Too short (OCR artifact)
    assert clean_commodity_number('123') is None
    # Too long
    assert clean_commodity_number('123456789') == '1234567'
    # Number with spaces and non-digit chars
    assert clean_commodity_number(' 00 10600a ') == '0010600'


def test_classify_word_by_coordinates():
    zones = {
        'commodity': {'min_x': 40, 'max_x': 130},
        'description': {'min_x': 140, 'max_x': 300},
        'unit': {'min_x': 310, 'max_x': 400},
        'rate_1930': {'min_x': 410, 'max_x': 480},
        'rate_trade': {'min_x': 490, 'max_x': 550},
        'tariff': {'min_x': 560, 'max_x': 650}
    }

    # Commodity number in commodity zone
    result = classify_word_by_coordinates('0010600', 50, zones)
    assert result[0] == '0010600'  # Commodity Number

    # Description in description zone
    result = classify_word_by_coordinates('Cattle', 150, zones)
    assert result[1] == 'Cattle'  # Description

    # Unit in unit zone
    result = classify_word_by_coordinates('Lb', 350, zones)
    assert result[2] == 'Lb'  # Unit

    # Rate 1930 in rate_1930 zone
    result = classify_word_by_coordinates('69', 420, zones)
    assert result[3] == '69'  # Rate 1930

    # Rate Trade in rate_trade zone
    result = classify_word_by_coordinates('39', 500, zones)
    assert result[4] == '39'  # Rate Trade

    # Tariff paragraph in tariff zone
    result = classify_word_by_coordinates('701', 600, zones)
    assert result[5] == '701'  # Tariff Paragraph

    # Word outside zones defaults to description
    result = classify_word_by_coordinates('Unknown', 700, zones)
    assert result[1] == 'Unknown'


def test_analyze_coordinate_zones():
    # Create a sample DataFrame mimicking OCR output
    data = {
        'Word': ['0010600', '701', 'Lb', 'Cattle', '6%', '3%', '4%'],
        'TopLeft_X': [50, 600, 350, 150, 420, 430, 440],  # rate area words close together
        'TopLeft_Y': [100, 100, 100, 100, 100, 100, 100]
    }
    df = pd.DataFrame(data)

    zones = analyze_coordinate_zones(df)

    # Check that zones keys exist
    assert 'commodity' in zones
    assert 'tariff' in zones
    assert 'unit' in zones
    assert 'description' in zones
    assert 'rate_1930' in zones
    # 'rate_trade' may or may not be present depending on data
def test_load_and_preprocess():
    # Write sample CSV to temp file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as tmp:
        tmp.write(SAMPLE_OCR_CSV)
        tmp_path = tmp.name

    try:
        df = load_and_preprocess(tmp_path)
        # Check that rows are loaded and header rows dropped
        assert len(df) > 0
        # Check that no rows have TopLeft_Y >= 2700
        assert all(df['TopLeft_Y'] < 2700)
    finally:
        os.remove(tmp_path)


def test_main_function():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv_path = os.path.join(tmpdir, "sample_ocr.csv")
        output_csv_path = os.path.join(tmpdir, "cleaned_classified_words.csv")

        with open(input_csv_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_OCR_CSV)

        import enhanced_clean
        enhanced_clean.INPUT_CSV = input_csv_path
        enhanced_clean.OUTPUT_CSV = output_csv_path

        output_df = enhanced_clean.main()

        # Basic assertions
        assert not output_df.empty
        assert 'Commodity Number' in output_df.columns
        assert 'Commodity Description' in output_df.columns
        assert 'Unit of Quantity' in output_df.columns
        assert 'Rate of Duty 1930' in output_df.columns
        assert 'Rate of Duty Trade Agreement' in output_df.columns
        assert 'Tariff Paragraph' in output_df.columns

        # Print summary
        print("=== Main Function Test Output Summary ===")
        print(f"Total rows processed: {len(output_df)}")
        print(f"Rows with Commodity Number: {output_df['Commodity Number'].notna().sum()}")
        print(f"Rows with Description: {output_df['Commodity Description'].notna().sum()}")
        print(f"Rows with Units: {output_df['Unit of Quantity'].notna().sum()}")
        print(f"Rows with Rate 1930: {output_df['Rate of Duty 1930'].notna().sum()}")
        print(f"Rows with Rate Trade: {output_df['Rate of Duty Trade Agreement'].notna().sum()}")
        print(f"Rows with Tariff Paragraph: {output_df['Tariff Paragraph'].notna().sum()}")

        print("\nSample classified rows:")
        print(output_df.head(10))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'run_main':
        test_main_function()
    else:
        test_clean_commodity_number()
        test_classify_word_by_coordinates()
        test_analyze_coordinate_zones()
        test_load_and_preprocess()
        print("All unit tests passed.")