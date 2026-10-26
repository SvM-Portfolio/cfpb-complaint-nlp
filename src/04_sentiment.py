"""Sentiment scoring with (1) VADER (lexicon, fast) and (2) DistilBERT SST-2 (transformer).
Writes outputs/sentiment.csv. Run once (DistilBERT on CPU takes ~10-15 min for ~20k narratives)."""
import re, time, numpy as np, pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from common import *

df = load_clean()[["Complaint ID", "narrative"]]
sia = SentimentIntensityAnalyzer()

def prep(t):  # keep sentence punctuation for VADER, but remove redaction tokens
    t = re.sub(r"\{\$[\d,\.]+\}", " money ", str(t)); t = re.sub(r"\bX{2,}(?:/X{2,})*\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def vader(t):
    t = prep(t)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if len(s) > 3] or [t]
    return np.mean([sia.polarity_scores(s)["compound"] for s in sents])

t0 = time.time()
df["text"] = df["narrative"].map(prep)
df["vader_sent_mean"] = df["text"].map(vader)            # mean of sentence-level compound scores (-1..1)
df["vader_doc"] = df["text"].map(lambda t: sia.polarity_scores(t)["compound"])
print(f"VADER done in {time.time()-t0:.0f}s")

from transformers import pipeline
clf = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=-1)
order = df["text"].str.len().sort_values().index          # length-sorted batching = much faster on CPU
res = {}
texts = df.loc[order, "text"].tolist()
t0 = time.time(); out = []
for i in range(0, len(texts), 512):
    out += clf(texts[i:i+512], truncation=True, max_length=256, batch_size=32)
    if (i // 512) % 5 == 0: print(f"distilbert {i}/{len(texts)}  {time.time()-t0:.0f}s", flush=True)
sc = pd.Series([o["score"] if o["label"] == "POSITIVE" else -o["score"] for o in out], index=order)
df["distilbert_signed"] = sc.reindex(df.index)           # +p(positive) or -p(negative)
df["distilbert_pos"] = (df["distilbert_signed"] > 0).astype(int)
df.drop(columns=["text", "narrative"]).to_csv(OUT / "sentiment.csv", index=False)
print(f"DistilBERT done in {time.time()-t0:.0f}s; saved outputs/sentiment.csv")
