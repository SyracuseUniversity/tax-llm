import pandas as pd
import numpy as np
import logging
import os

# =========================
# Configuration
# =========================
INPUT_CSV = 'old-work/output/new_hierarchical_commodities.csv'
OUTPUT_CSV = 'old-work/output/final_tables.csv'
Y_PROXIMITY = 5  # Pixel threshold for "nearby" tariff paragraph

# =========================
# Logging Setup
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fill_tariff_paragraphs_from_commodity(df: pd.DataFrame, y_proximity: int = Y_PROXIMITY) -> pd.DataFrame:
    """
    For each commodity, assign the nearest tariff paragraph below it (within y_proximity pixels),
    or the next available tariff paragraph below if none is close.
    """
    df = df.copy()
    commodity_indices = df[df["Hierarchical Description"].notna()].index

    for i, idx in enumerate(commodity_indices):
        comm_y = df.loc[idx, "BottomRight_Y"]
        comm_page = df.loc[idx, "Page"]

        # Step 1: Check for nearby existing Tariff Paragraph
        nearby_mask = (
            (df["Tariff Paragraph"].notna()) &
            (df["Page"] == comm_page) &
            (df["BottomRight_Y"].between(comm_y, comm_y + y_proximity))
        )
        if df[nearby_mask].shape[0] > 0:
            nearest_tariff_para = df[nearby_mask].iloc[0]["Tariff Paragraph"]
            df.loc[idx, "Tariff Paragraph"] = nearest_tariff_para
            continue  # Already filled

        # Step 2: Look for the next available tariff paragraph below
        tariff_rows = df.loc[idx + 1 :]
        tariff_rows = tariff_rows[tariff_rows["Tariff Paragraph"].notna()]

        if not tariff_rows.empty:
            tariff_idx = tariff_rows.index[0]
            tariff_page = df.loc[tariff_idx, "Page"]
            tariff_y = df.loc[tariff_idx, "BottomRight_Y"]
            distance = abs(tariff_y - comm_y)

            y_min = comm_y
            y_max = df.loc[tariff_idx, "BottomRight_Y"] + distance + 10
            mask = (
                (df["BottomRight_Y"] >= y_min) &
                (df["BottomRight_Y"] <= y_max) &
                (df["Tariff Paragraph"].isna()) &
                (df["Page"] == tariff_page)
            )
            df.loc[mask, "Tariff Paragraph"] = df.loc[tariff_idx, "Tariff Paragraph"]

    return df

def main():
    # Load data
    df = pd.read_csv(INPUT_CSV)
    logging.info(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # Drop old description column if present
    if 'Commodity Description' in df.columns:
        df = df.drop(['Commodity Description'], axis=1)

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Ensure BottomRight_Y is numeric
    df["BottomRight_Y"] = pd.to_numeric(df["BottomRight_Y"], errors="coerce")

    # Apply the logic
    df_filled = fill_tariff_paragraphs_from_commodity(df)

    # Forward-fill missing Commodity Numbers
    df_filled['Commodity Number'] = df_filled['Commodity Number'].ffill()

    # Fill empty Commodity Number at start if needed
    if pd.isna(df_filled['Commodity Number'].iloc[0]):
        df_filled.loc[0, 'Commodity Number'] = df_filled['Commodity Number'].bfill().iloc[0]

    # Drop rows without a hierarchical description
    df_filled = df_filled.dropna(subset=['Hierarchical Description'])

    # Save the result
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_filled.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Saved final table to {OUTPUT_CSV}")

    # Show a preview
    logging.info("\n" + str(df_filled[['Commodity Number', 'Hierarchical Description', 'Tariff Paragraph']].head(10)))

if __name__ == "__main__":
    main()