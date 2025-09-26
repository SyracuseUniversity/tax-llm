import pytest
import os
import pandas as pd
from get_ocr_data import extract_ocr_words_with_coords
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="paddleocr")

def create_valid_dummy_pdf(pdf_path, num_pages=28):
    """
    Create a valid dummy PDF file with the specified number of pages using PyMuPDF.
    """
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page()  # Create a new page
        page.insert_text((72, 72), f"This is page {i + 1} of the test PDF.")  # Add text to each page
    doc.save(pdf_path)
    doc.close()
def test_extract_ocr_words_with_output_files(tmp_path):
    """
    Test the extract_ocr_words_with_coords function to ensure it generates the expected output files.

    Why:
        Validates that the function processes a PDF and generates the correct CSV outputs.

    How:
        - Creates a valid dummy PDF file with 28 pages.
        - Calls the function to generate OCR output.
        - Verifies the existence and contents of the output files.
    """
    # Ensure PaddleOCR is installed
    try:
        ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
    except ImportError:
        pytest.fail("PaddleOCR is not installed. Please install it using 'pip install paddleocr'.")

    # Mock input and output paths
    pdf_path = tmp_path / "sample.pdf"
    ocr_output_csv = tmp_path / "ocr_word_coords.csv"
    cleaned_output_csv = tmp_path / "cleaned_classified_words.csv"

    # Create a valid dummy PDF file with 28 pages
    create_valid_dummy_pdf(pdf_path, num_pages=28)

    # Call the function to extract OCR words
    extract_ocr_words_with_coords(
        pdf_path=pdf_path,
        start_page=28,
        end_page=28,
        ocr=ocr,
        output_csv=ocr_output_csv
    )

    # Check if the OCR output file is created
    assert ocr_output_csv.exists(), "OCR output CSV file was not created."

    # Verify the contents of the OCR output file
    ocr_df = pd.read_csv(ocr_output_csv)
    print(ocr_df)  # Debugging: Print the contents of the CSV file
    assert not ocr_df.empty, "OCR output CSV file is empty."
    assert "Word" in ocr_df.columns, "Expected column 'Word' not found in the OCR output CSV."
    assert "TopLeft_X" in ocr_df.columns, "Expected column 'TopLeft_X' not found in the OCR output CSV."
    assert "TopLeft_Y" in ocr_df.columns, "Expected column 'TopLeft_Y' not found in the OCR output CSV."

    # Simulate cleaning process (mocked for this test)
    ocr_df.to_csv(cleaned_output_csv, index=False)

    # Check if the cleaned output file is created
    assert cleaned_output_csv.exists(), "Cleaned output CSV file was not created."

    # Verify the contents of the cleaned output file
    cleaned_df = pd.read_csv(cleaned_output_csv)
    assert not cleaned_df.empty, "Cleaned output CSV file is empty."
    assert "Word" in cleaned_df.columns, "Expected column 'Word' not found in the cleaned output CSV."
    assert "TopLeft_X" in cleaned_df.columns, "Expected column 'TopLeft_X' not found in the cleaned output CSV."
    assert "TopLeft_Y" in cleaned_df.columns, "Expected column 'TopLeft_Y' not found in the cleaned output CSV."