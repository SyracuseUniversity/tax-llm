import numpy as np
from paddleocr import PaddleOCR, TableCellsDetection
import pandas as pd
import json

model = TableCellsDetection(model_name="RT-DETR-L_wired_table_cell_det")
output = model.predict("./img28.jpg", threshold=0.3, batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./new-work/output/")
    res.save_to_json("./new-work/output/res.json")

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

ocr_results = ocr.predict("./img28.jpg")[0]

with open("./new-work/output/res.json") as f:
    cells = json.load(f)['boxes']

cell_texts = []

for cell in cells:
    xmin, ymin, xmax, ymax = cell['coordinate']
    words_in_cell = []

    for i, word in enumerate(ocr_results['rec_texts']):
        poly = ocr_results['rec_polys'][i]
        center_x = sum(p[0] for p in poly)/4
        center_y = sum(p[1] for p in poly)/4

        if xmin <= center_x <= xmax and ymin <= center_y <= ymax:
            words_in_cell.append(word)

    # even if empty, append an empty string
    cell_texts.append(" ".join(words_in_cell))

# Build DataFrame after finishing the loop
df = pd.DataFrame({
    "cell_text": cell_texts,
    "xmin": [c['coordinate'][0] for c in cells],
    "ymin": [c['coordinate'][1] for c in cells],
    "xmax": [c['coordinate'][2] for c in cells],
    "ymax": [c['coordinate'][3] for c in cells]
})

# Assign row IDs by vertical proximity
df = df.sort_values("ymin").reset_index(drop=True)
current_row = 0
row_ids = []
last_y = None
y_threshold = 10  # adjust for your spacing

for y in df['ymin']:
    if last_y is None or abs(y - last_y) > y_threshold:
        current_row += 1
    row_ids.append(current_row)
    last_y = y

df['row_id'] = row_ids

# Assign column IDs within each row
def assign_cols(subdf, x_thresh=10):
    subdf = subdf.sort_values("xmin").reset_index(drop=True)
    col_id = 0
    last_x = None
    col_ids = []
    for x in subdf['xmin']:
        if last_x is None or abs(x - last_x) > x_thresh:
            col_id = col_id + 1
        col_ids.append(col_id)
        last_x = x
    subdf['col_id'] = col_ids
    return subdf

df = df.groupby('row_id').apply(assign_cols).reset_index(drop=True)

table = df.pivot_table(
    index='row_id',
    columns='col_id',
    values='cell_text',
    aggfunc=lambda x: " ".join(x)  # join texts if duplicates
)
table.to_csv("./new-work/output/final_table.csv", index=False)