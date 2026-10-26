"""Download CFPB complaint-narrative archives and build the healthcare-adjacent analysis sample.

Usage: python src/01_download_and_prepare.py
Streams each monthly archive, keeps complaints with a narrative that mentions a medical-care term,
then draws a reproducible random sample (<= MAX_ROWS).
"""
import urllib.request, pandas as pd
from config import *

RAW.mkdir(parents=True, exist_ok=True)
for fn in ARCHIVE_FILES:
    dest = RAW / fn
    if not dest.exists():
        print("downloading", fn)
        urllib.request.urlretrieve(BASE_URL + fn, dest)

cols = ["Date received", "Product", "Sub-product", "Issue", "Sub-issue", "Consumer complaint narrative",
        "Company", "State", "Submitted via", "Company response to consumer", "Timely response?", "Complaint ID"]
parts, n_all, n_narr = [], 0, 0
for fn in ARCHIVE_FILES:
    for ch in pd.read_csv(RAW / fn, usecols=cols, dtype=str, chunksize=200_000):
        n_all += len(ch)
        ch = ch[ch["Consumer complaint narrative"].notna()]
        n_narr += len(ch)
        m = ch["Consumer complaint narrative"].str.contains(MEDICAL_REGEX, case=False, regex=True)
        parts.append(ch[m])
df = pd.concat(parts, ignore_index=True).drop_duplicates("Complaint ID")
print(f"records scanned: {n_all:,}; with narrative: {n_narr:,}; narrative mentions medical term: {len(df):,}")
if len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=SEED)
df = df.rename(columns={"Consumer complaint narrative": "narrative"}).reset_index(drop=True)
df.to_csv(PROCESSED, index=False)
print(f"saved {len(df):,} rows -> {PROCESSED}")
print(df["Product"].value_counts().head(10))
