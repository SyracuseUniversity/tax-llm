import pandas as pd
import re
from typing import List

# =======================================
# CONFIGURATION
# =======================================
INPUT_CSV = r"new-work/output/cleaned_classified_words.csv"
OUTPUT_CSV = r"new-work/output/final_table.csv"

# =======================================
# COLUMN DEFINITIONS
# =======================================
cols = [
    "Commodity Number",
    "Commodity Description",
    "Unit of Quantity",
    "Rate of Duty 1930 Tariff Act",
    "Rate of Duty Trade Agreement",
    "Tariff Paragraph",
]

# =======================================
# HEADER SPLITTING HELPERS
# =======================================
HEADER_SUB_RE = re.compile(r'''
    (?P<header>
      (?=(?:.*[A-Z]){3,})            # at least 3 uppercase letters somewhere
      [A-Z0-9\(\)\[\],\.\-:&\s]{4,}  # allowed uppercase + punctuation + spaces, length >=4
    )
''', re.VERBOSE)

def split_headers_in_text(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    parts = []
    remaining = s
    while remaining:
        m = HEADER_SUB_RE.search(remaining)
        if not m:
            parts.append(remaining.strip())
            break
        start, end = m.span('header')
        before = remaining[:start].strip()
        header = m.group('header').strip()
        after = remaining[end:].strip()
        if before:
            parts.append(before)
        parts.append(header)
        remaining = after
    return [p for p in parts if p]

def is_header_text(s: str) -> bool:
    if not s or not isinstance(s, str):
        return False
    letters = re.sub(r'[^A-Za-z]', '', s)
    if len(letters) < 3:
        return False
    return not re.search(r'[a-z]', s) and re.search(r'[A-Z].*[A-Z].*[A-Z]', s)

# =======================================
# LOAD RAW DATA
# =======================================
df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")

# Normalize column names
col_map = {}
available = {c.lower().strip(): c for c in df.columns}
for desired in cols:
    key = desired.lower().strip()
    if key in available:
        col_map[desired] = available[key]
    else:
        df[desired] = ""
        col_map[desired] = desired
df = df[[col_map[c] for c in cols]]
df.columns = cols

# =======================================
# SPLIT INTO FRAGMENTS
# =======================================
fragments = []
for _, row in df.iterrows():
    part_lists = {}
    max_parts = 1
    for c in cols:
        text = (row[c] or "").strip()
        if c == "Commodity Number":
            part_lists[c] = [text] if text else [""]
        else:
            parts = split_headers_in_text(text) or [""]
            part_lists[c] = parts
            max_parts = max(max_parts, len(parts))
    for i in range(max_parts):
        frag = {c: (part_lists[c][i] if i < len(part_lists[c]) else "") for c in cols}
        fragments.append(frag)

# =======================================
# FLATTENED RECORDS
# =======================================
def join_desc(a, b):
    if not a: return b
    if not b: return a
    return (a + " " + b).strip()

commodity_re = re.compile(r"^\s*\d{4}\s?\d{3}\s*$")
rows_out = []
current = {c: "" for c in cols}

for frag in fragments:
    code = frag["Commodity Number"].strip()
    desc_frag = frag["Commodity Description"].strip()
    is_header_frag = is_header_text(desc_frag) and not commodity_re.match(desc_frag)

    # Start new record when new commodity number
    if code and commodity_re.match(code):
        if any(current[c] for c in cols if c != "Commodity Number") or current["Commodity Number"]:
            rows_out.append(current)
        current = {c: "" for c in cols}
        current["Commodity Number"] = code

    # Header fragment as its own row
    if is_header_frag:
        if any(current[c] for c in cols if c != "Commodity Number") or current["Commodity Number"]:
            rows_out.append(current)
        header_row = {c: "" for c in cols}
        header_row["Commodity Description"] = desc_frag
        rows_out.append(header_row)
        current = {c: "" for c in cols}
        continue

    # Append data
    if desc_frag:
        current["Commodity Description"] = join_desc(current["Commodity Description"], desc_frag)

    # ✅ Same splitting logic for Unit + Rate columns
    for col in [
        "Unit of Quantity",
        "Rate of Duty 1930 Tariff Act",
        "Rate of Duty Trade Agreement",
    ]:
        val = frag[col].strip()
        if val:
            # if current row already has a value → push it, start new row with only this col filled
            if current[col]:
                rows_out.append(current.copy())
                new_row = {c: "" for c in cols}
                new_row[col] = val
                rows_out.append(new_row)
                continue
            else:
                current[col] = val

    # Add Tariff Paragraph if present
    val = frag["Tariff Paragraph"].strip()
    if val:
        current["Tariff Paragraph"] = join_desc(current["Tariff Paragraph"], val)

# Append final
if any(current[c] for c in cols if c != "Commodity Number") or current["Commodity Number"]:
    rows_out.append(current)

# =======================================
# CLEAN OUTPUT
# =======================================
def clean_text(s):
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

cleaned_df = pd.DataFrame(rows_out, columns=cols)
for c in cols:
    cleaned_df[c] = cleaned_df[c].apply(clean_text)

# =======================================
# SAVE OUTPUT
# =======================================
cleaned_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"✅ Cleaned flattened CSV saved to: {OUTPUT_CSV}")
