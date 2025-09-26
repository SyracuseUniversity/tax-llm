#!/usr/bin/env python3
"""
Tariff Data Processing Pipeline

This script orchestrates the end-to-end processing of tariff data by sequentially running
all the modular Python scripts you provided. It verifies script existence, runs each step,
logs progress and errors, and summarizes the pipeline run.

Environment variables can override default paths for input/output directories
and script locations.

Usage:
    python pipeline.py [--skip-get-ocr-data] [--skip-enhanced-clean] [--skip-commodity-number]
                   [--skip-hierarchical-description] [--skip-rate-of-duty] [--skip-tariff-paragraph]

Environment Variables:
    TARIFF_BASE_DIR           Base directory for output files (default: 'tax-llm/output')
    INPUT_OCR_DIR             Input OCR raw data directory (default: 'tax-llm/input/ocr_raw')
    INPUT_PDF_DIR             Input PDFs directory (default: 'tax-llm/input/pdfs')
    GET_OCR_DATA_SCRIPT       Path to get_ocr_data.py script
    ENHANCED_CLEAN_SCRIPT     Path to enhanced_clean.py script
    COMMODITY_NUMBER_SCRIPT   Path to commodity_number02.py script
    HIERARCHICAL_DESCRIPTION_SCRIPT Path to hierarchical_description03.py script
    RATE_OF_DUTY_SCRIPT       Path to rate_of_duty05.py script
    TARIFF_PARAGRAPH_SCRIPT   Path to tarrif_para06.py script
"""

import os
import sys
import argparse
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read environment variables or use defaults
BASE_DIR = os.getenv('TARIFF_BASE_DIR', 'tax-llm/output')
INPUT_OCR_DIR = os.getenv('INPUT_OCR_DIR', 'tax-llm/input/ocr_raw')
INPUT_PDF_DIR = os.getenv('INPUT_PDF_DIR', 'tax-llm/input/pdfs')

SCRIPTS = {
    'get_ocr_data': os.getenv('GET_OCR_DATA_SCRIPT', 'get_ocr_data.py'),
    'enhanced_clean': os.getenv('ENHANCED_CLEAN_SCRIPT', 'enhanced_clean.py'),
    'commodity_number': os.getenv('COMMODITY_NUMBER_SCRIPT', 'commodity_number02.py'),
    'hierarchical_description': os.getenv('HIERARCHICAL_DESCRIPTION_SCRIPT', 'hierarchical_description03.py'),
    'rate_of_duty': os.getenv('RATE_OF_DUTY_SCRIPT', 'rate_of_duty05.py'),
    'tariff_paragraph': os.getenv('TARIFF_PARAGRAPH_SCRIPT', 'tarrif_para06.py'),
}

def ensure_directories():
    """
    Ensure the base output directory and required input directories exist.
    Creates the directories if they do not exist.
    """
    for directory in [BASE_DIR, INPUT_OCR_DIR, INPUT_PDF_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

def run_script(script_name):
    """
    Run a script by name from the SCRIPTS dictionary.
    Checks if the script file exists before running.
    Captures and logs output and errors.
    Returns True if successful, False otherwise.
    """
    script_path = SCRIPTS[script_name]
    if not os.path.isfile(script_path):
        logging.warning(f"Script file {script_path} not found. Skipping {script_name} step.")
        return False

    logging.info(f"Running {script_path}...")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Error running {script_path}:\n{result.stderr}")
        return False
    else:
        logging.info(f"{script_path} completed successfully.")
        if result.stdout.strip():
            logging.info(result.stdout.strip())
        return True

def main():
    parser = argparse.ArgumentParser(description="Run the full tariff data processing pipeline.")
    parser.add_argument('--skip-get-ocr-data', action='store_true', help='Skip get_ocr_data.py step')
    parser.add_argument('--skip-enhanced-clean', action='store_true', help='Skip enhanced_clean.py step')
    parser.add_argument('--skip-commodity-number', action='store_true', help='Skip commodity_number02.py step')
    parser.add_argument('--skip-hierarchical-description', action='store_true', help='Skip hierarchical_description03.py step')
    parser.add_argument('--skip-rate-of-duty', action='store_true', help='Skip rate_of_duty05.py step')
    parser.add_argument('--skip-tariff-paragraph', action='store_true', help='Skip tarrif_para06.py step')
    args = parser.parse_args()

    logging.info("Starting tariff data processing pipeline...")
    ensure_directories()

    steps = [
        ('get_ocr_data', args.skip_get_ocr_data),
        ('enhanced_clean', args.skip_enhanced_clean),
        ('commodity_number', args.skip_commodity_number),
        ('hierarchical_description', args.skip_hierarchical_description),
        ('rate_of_duty', args.skip_rate_of_duty),
        ('tariff_paragraph', args.skip_tariff_paragraph),
    ]

    results = {}

    for step_name, skip in steps:
        if skip:
            logging.info(f"Skipping {step_name} step as requested.")
            results[step_name] = 'skipped'
            continue

        success = run_script(step_name)
        results[step_name] = 'success' if success else 'failed'

    logging.info("Pipeline run summary:")
    for step, status in results.items():
        logging.info(f"  {step}: {status}")

    if any(status == 'failed' for status in results.values()):
        logging.error("One or more pipeline steps failed. See logs for details.")
        sys.exit(1)

    logging.info("Pipeline completed successfully.")


if __name__ == '__main__':
    main()