# Digitizing Tax Tariff Documents - Automated Data Extraction Pipeline

## Overview

This project provides an automated solution for digitizing complex tariff documents from the early/mid 20th century. The system extracts structured data from scanned PDFs containing hierarchical commodity descriptions and associated tariff rates, transforming them into usable digital formats (CSV).

The pipeline consists of four main components that work sequentially:

1. **OCR Data Extraction (`get_ocr_data.py`)** - Extracts text and precise coordinates from PDFs
2. **Word Classification (`classifying_words.py`)** - Classifies extracted words into commodity numbers, descriptions, and tariff paragraphs
3. **Hierarchical Processing (`hierarchical_clustering.py`)** - Reconstructs the hierarchical relationships between commodity descriptions
4. **Tariff Paragraph Assignment (`tarrif_paragraphs.py`)** - Intelligently associates tariff paragraphs with their corresponding commodities

---

## Detailed Workflow

### 1. OCR Data Extraction (`get_ocr_data.py`)

**Purpose:** Extract text content with precise positional coordinates from PDF documents.

**Key Features:**

- Uses PaddleOCR for high-accuracy text recognition
- Captures exact coordinates (x,y positions) of each word
- Handles multi-page PDF documents
- Filters out header/footer content and irrelevant sections

**How it works:**

- Opens the PDF file using PyMuPDF (`fitz`)
- Processes each page to create high-resolution images (300 DPI)
- Uses PaddleOCR to detect text and bounding boxes
- Stores each word with its:
  - Text content
  - Confidence score
  - Coordinate positions (top-left, bottom-right, etc.)
  - Page number

**Output:** `ocr_word_coords.csv` containing all extracted words with their metadata

---

### 2. Word Classification (`classifying_words.py`)

**Purpose:** Classify extracted words into three categories:

- Commodity Numbers (e.g., "0010 600")
- Commodity Descriptions (e.g., "Cattle: Weighing less than 200 pounds")
- Tariff Paragraphs (e.g., "(2)")

**Key Features:**

- Uses regular expressions for precise pattern matching
- Handles edge cases and special characters
- Cleans and filters the data

**How it works:**

- Loads the OCR output data
- Applies classification rules:
  - Commodity numbers match pattern `00XX YYY` (e.g., "0010 600")
  - Tariff paragraphs are typically 1-4 digits, sometimes in parentheses
  - All other text becomes commodity descriptions
- Performs data cleaning:
  - Removes rows with only numbers
  - Combines short tariff codes with their descriptions
  - Filters out unwanted rows

**Output:** `cleaned_classified_words.csv` with classified data

---

### 3. Hierarchical Processing (`hierarchical_clustering.py`)

**Purpose:** Reconstruct the hierarchical relationships between commodity descriptions based on their indentation levels.

**Key Features:**

- Uses pixel-level coordinates to determine hierarchy
- Combines split lines into complete descriptions
- Identifies parent-child relationships
- Preserves the original document structure

**How it works:**

- Sorts data by page and vertical position
- Analyzes horizontal (X) positions to determine indentation levels
- Processes each line to:
  - Combine continuation lines (based on Y proximity)
  - Build hierarchical descriptions by combining parent and child items
  - Identify parent items (ending with ":")
- Removes redundant parent rows while preserving hierarchy information

**Output:**

- `new_hierarchical_commodities.csv` - Processed data with hierarchical descriptions
- `formatted_commodities.txt` - Clean hierarchical descriptions for reference

---

### 4. Tariff Paragraph Assignment (`tarrif_paragraphs.py`)

**Purpose:** Intelligently associate tariff paragraphs with their corresponding commodities.

**Key Features:**

- Uses spatial proximity to match tariff paragraphs
- Handles cases where paragraphs appear after descriptions
- Preserves page boundaries for accurate matching

**How it works:**

- For each commodity description:
  - First checks for nearby existing tariff paragraphs (within 5 pixels vertically)
  - If none found, looks for the next tariff paragraph below the description
  - Calculates a reasonable vertical range to assign the tariff paragraph
- Fills forward commodity numbers to maintain consistency
- Removes empty rows and cleans the final output

**Output:** `final_tables.csv` - The fully processed data with all relationships intact

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- `pip` package manager


---
# Usage Instructions
### 1. Prepare Your PDF
Place your tariff document PDF in the project directory. The default filename expected is: 1950 Schedule A (no OCR).pdf

If your PDF has a different name, modify the pdf_path variable in get_ocr_data.py.

### 2. Configure Page Range
Edit the following variables in get_ocr_data.py to specify which pages to process:

start_page = 28  # First page to process (1-based index)
end_page = 28    # Last page to process


### 3. Run the Pipeline
Execute the scripts in order:

    python get_ocr_data.py
    python classifying_words.py
    python hierarchical_clustering.py
    python tarrif_paragraphs.py

### 4. View Results
The final output will be saved as: final_tables.csv

Intermediate files are also created at each step for debugging:

    1. ocr_word_coords.csv - Raw OCR output
    2. cleaned_classified_words.csv - After classification
    3. new_hierarchical_commodities.csv - After hierarchy processing

---
### Installation Steps

Clone the repository:

```bash
git clone [repository-url]
cd [repository-directory]

conda create -n tariffdocs -c conda-forge -y python
pip install pandas numpy paddleocr paddlepaddle Pillow torch PyMuPDF
```
