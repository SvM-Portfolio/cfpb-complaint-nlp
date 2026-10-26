from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
PROCESSED = DATA / "medical_complaints.csv.gz"
SEED = 42
BASE_URL = "https://files.consumerfinance.gov/f/documents/"
# CFPB stopped publishing narratives in the live database on 2026-08-14; previously published
# narratives are in the FOIA Reading Room archive (the exports below, Sep 2023 - Aug 2026).
ARCHIVE_FILES = [
    "CCDB_Export_5_September_2023_through_March_2024.zip",
    "CCDB_Export_6_April_2024_through_July_2024.zip",
    "CCDB_Export_7_August_2024_through_October_2024.zip",
    "CCDB_Export_8_November_2024_through_December_2024.zip",
    "CCDB_Export_9_January_2025_through_February_2025.zip",
    "CCDB_Export_10_March_2025_through_April_2025.zip",
    "CCDB_Export_11_May_2025_through_June_2025.zip",
    "CCDB_Export_12_July_2025_through_August_2025.zip",
    "CCDB_Export_13_September_2025_through_October_2025.zip",
    "CCDB_Export_14_November_2025_through_December_2025.zip",
    "CCDB_Export_15_January_2026_through_February_2026.zip",
    "CCDB_Export_16_March_2026.zip",
    "CCDB_Export_17_April_2026.zip",
    "CCDB_Export_18_May_2026.zip",
    "CCDB_Export_19_June_2026.zip",
    "CCDB_Export_20_July_2026.zip",
    "CCDB_Export_21_August_2026.zip",
]
# Healthcare-adjacent filter: narrative must mention a medical-care term
MEDICAL_REGEX = (r"\b(?:medical|hospital|ambulance|clinic|physician|doctor|healthcare|health care|"
                 r"emergency room|surgery|radiology|urgent care|anesthesi\w+|dental|dentist|pharmacy|prescription|medicaid|medicare|health insurance)\b")
MAX_ROWS = 40000
