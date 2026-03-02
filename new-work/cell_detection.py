import pandas as pd
import os
from pathlib import Path
import numpy as np

#Configuration
BASE_DIR = Path(__file__).parent
input_csv = BASE_DIR / "output/ocr_word_coords.csv"
output_table_csv = BASE_DIR / "output/detected_table.csv"

# Load word coordinates from CSV
def load_word_coords(csv_path: str) -> pd.DataFrame:
    """
    Load word coordinates from a CSV file.
    """
    df = pd.read_csv(csv_path)

    # Compute word center points
    df["center_x"] = (
        df["TopLeft_X"] + df["TopRight_X"] +
        df["BottomRight_X"] + df["BottomLeft_X"]
    ) / 4 # Compute center x

    df["center_y"] = (
        df["TopLeft_Y"] + df["TopRight_Y"] +
        df["BottomRight_Y"] + df["BottomLeft_Y"]
    ) / 4 # Compute center y

    return df

def assign_rows(df: pd.DataFrame, y_threshold: int = 15) -> pd.DataFrame:
    """
    Assign row numbers based on vertical proximity.
    """

    df = df.sort_values("center_y").reset_index(drop=True)

    current_row = 0
    row_assignments = []
    last_y = None

    for _, row in df.iterrows():
        if last_y is None:
            row_assignments.append(current_row)
            last_y = row["center_y"]
            continue

        if abs(row["center_y"] - last_y) > y_threshold:
            current_row += 1
            last_y = row["center_y"]

        row_assignments.append(current_row)

    df["row_id"] = row_assignments
    return df

def assign_columns(df: pd.DataFrame, x_threshold: int = 30) -> pd.DataFrame:
    """
    Detect column centers based on clustering of X positions.
    """

    # Sort by x
    df = df.sort_values("center_x").reset_index(drop=True)

    column_centers = []

    for x in df["center_x"]:
        placed = False

        for i, center in enumerate(column_centers):
            if abs(x - center) < x_threshold:
                column_centers[i] = (center + x) / 2
                placed = True
                break

        if not placed:
            column_centers.append(x)

    # Now assign col_id based on closest center
    col_ids = []

    for x in df["center_x"]:
        distances = [abs(x - c) for c in column_centers]
        col_ids.append(np.argmin(distances))

    df["col_id"] = col_ids

    return df

def build_table(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure Word column is string and drop NaNs
    df["Word"] = df["Word"].fillna("").astype(str)

    grouped = (
        df.groupby(["row_id", "col_id"])["Word"]
        .apply(lambda words: " ".join(w for w in words if w.strip() != ""))
        .reset_index()
    )

    table = grouped.pivot(
        index="row_id",
        columns="col_id",
        values="Word"
    )

    table = table.sort_index().reset_index(drop=True)
    return table

def detect_table_from_ocr(csv_path: str, output_path: str):
    df = load_word_coords(csv_path)

    df = assign_rows(df, y_threshold=18)
    df = assign_columns(df, x_threshold=30)

    table = build_table(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    table.to_csv(output_path, index=False)

    print(f"Table saved to: {output_path}")

def main():
    print("Detecting table structure...")
    detect_table_from_ocr(input_csv, output_table_csv)

if __name__ == "__main__":
    main()