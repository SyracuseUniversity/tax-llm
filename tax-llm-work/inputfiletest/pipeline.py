#!/usr/bin/env python3
"""
End-to-End Tax LLM Data Processing Pipeline

This script integrates multiple stages of tax-related data processing into a single pipeline. Each stage corresponds to a specific task, such as extracting OCR data, cleaning it, and processing various components like commodity numbers, hierarchical descriptions, units of quantity, rates of duty, and tariff paragraphs.

Main Stages
-----------
1. Extract OCR data (via 01_get_ocr_data.py).
2. Perform enhanced cleaning of OCR data (via 01b_enhanced_cleaning.py).
3. Process commodity numbers (via 02_commodity_number.py).
4. Process hierarchical descriptions (via 03_hierarchical_description.py).
5. Process units of quantity (via 04_unit_of_quantity.py).
6. Process rates of duty (via 05_rate_of_duty.py).
7. Process tariff paragraphs (via 06_tarrif_para.py).

Usage
-----
Run the script directly:
    python tax_llm_pipeline.py

Prerequisites
-------------
Ensure all the required modules are available and the individual scripts are functional.
"""

from __future__ import annotations

import os
from get_ocr_data import main as extract_ocr_data
from enhanced_clean import main as enhanced_cleaning
from commodity_number02 import main as process_commodity_numbers
from hierarchical_description03 import main as process_hierarchical_descriptions
from unit_of_quantity04 import main as process_units
from rate_of_duty05 import main as process_rates
from tarrif_para06 import main as process_tariff_paragraphs
import argparse
import logging

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the Tax LLM Data Processing Pipeline.")
    parser.add_argument("--input-dir", default="data/input", help="Directory for input files.")
    parser.add_argument("--output-dir", default="data/output", help="Directory for output files.")
    parser.add_argument("--skip-cleaning", action="store_true", help="Skip the cleaning stage.")
    parser.add_argument("--skip-commodity", action="store_true", help="Skip the commodity processing stage.")
    parser.add_argument("--skip-hierarchy", action="store_true", help="Skip the hierarchical description stage.")
    parser.add_argument("--skip-units", action="store_true", help="Skip the units of quantity stage.")
    parser.add_argument("--skip-rates", action="store_true", help="Skip the rates of duty stage.")
    parser.add_argument("--skip-tariff", action="store_true", help="Skip the tariff paragraphs stage.")
    return parser.parse_args()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    print("Step 1: Extracting OCR data...")
    extract_ocr_data()

    print("Step 2: Enhanced cleaning of OCR data...")
    enhanced_cleaning()

    print("Step 3: Processing commodity numbers...")
    process_commodity_numbers()

    print("Step 4: Processing hierarchical descriptions...")
    process_hierarchical_descriptions()

    print("Step 5: Processing units of quantity...")
    process_units()

    print("Step 6: Processing rates of duty...")
    process_rates()

    print("Step 7: Processing tariff paragraphs...")
    process_tariff_paragraphs()

    print("Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()