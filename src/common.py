import re
import pandas as pd
from config import *

PRODUCT_MAP = {
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Debt collection": "Debt collection",
    "Credit card": "Credit card",
    "Checking or savings account": "Bank account",
    "Mortgage": "Mortgage",
}


def clean_text(t: str) -> str:
    """Normalise a CFPB narrative for modelling.
    - CFPB redacts personal info as XXXX / XX/XX/XXXX and dollar amounts as {$123.00}: remove / tokenise them
    - lowercase, drop URLs, digits and punctuation, collapse whitespace
    """
    t = str(t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\{\$[\d,\.]+\}", " dollaramount ", t)
    t = re.sub(r"\bX{2,}(?:/X{2,})*\b", " ", t)           # XXXX, XX/XX/XXXX redactions
    t = t.lower()
    t = re.sub(r"[^a-z\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_raw():
    return pd.read_csv(PROCESSED)


def load_clean():
    return pd.read_csv(DATA / "clean.csv.gz", parse_dates=["date"])


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    print("saved", name)
