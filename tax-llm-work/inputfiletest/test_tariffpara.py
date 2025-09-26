import pytest
import pandas as pd
from io import StringIO
import tempfile
import os
import warnings

import tarrif_para06 as tp

@pytest.fixture
def sample_ocr_data():
    data = StringIO(
        """Commodity Number,TopLeft_Y,Tariff Paragraph
0106 000,100,701
0107 000,150,701
0020 000,200,702
0021 000,210,703
0021 000,215,704
"""
    )
    return pd.read_csv(data)

@pytest.fixture
def sample_final_table():
    data = StringIO(
        """SCHEDULE A COMMODITY NUMBER,COMMODITY DESCRIPTION AND ECONOMIC CLASS,UNIT OF QUANTITY,RATE OF DUTY 1930,RATE OF DUTY TRADE AGREEMENT,TARIFF PARAGRAPH
0106 000,,,,,
0107 000,,,,,
0020 000,,,,,
0021 000,,,,,
0021 000,,,,,
"""
    )
    return pd.read_csv(data)

def test_extract_tariff_ranges(sample_ocr_data):
    tariff_map = tp.extract_tariff_ranges(sample_ocr_data)
    assert isinstance(tariff_map, dict)
    assert any(isinstance(v, str) for v in tariff_map.values())

def test_apply_tariff_to_final_table(sample_ocr_data, sample_final_table):
    with tempfile.TemporaryDirectory() as tmpdir:
        ocr_csv_path = os.path.join(tmpdir, 'cleaned_classified_words.csv')
        final_csv_path = os.path.join(tmpdir, 'final-table.csv')

        sample_ocr_data.to_csv(ocr_csv_path, index=False)
        sample_final_table.to_csv(final_csv_path, index=False)

        tp.OCR_CSV = ocr_csv_path
        tp.FINAL_CSV = final_csv_path

        tariff_map = tp.extract_tariff_ranges(sample_ocr_data)

        # Suppress pandas dtype warning during assignment
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            tp.apply_tariff_to_final_table(tariff_map)

        updated_final = pd.read_csv(final_csv_path)
        # Check that tariff paragraphs column is not empty for some rows
        assert updated_final['TARIFF PARAGRAPH'].dropna().astype(bool).any()

if __name__ == '__main__':
    pytest.main()