"""Text cleaning + exploratory data analysis. Writes data/clean.csv.gz and EDA figures."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd, numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from common import *

sns.set_theme(style="whitegrid")
df = load_raw()
n0 = len(df)
df["date"] = pd.to_datetime(df["Date received"])
df["clean"] = df["narrative"].map(clean_text)
df["n_words"] = df["clean"].str.split().str.len()
df["product_group"] = df["Product"].map(PRODUCT_MAP).fillna("Other products")
df["is_medical_debt"] = (df["Sub-product"] == "Medical debt").astype(int)
# drop empty / very short narratives and exact duplicate texts (templated letters inflate classifier scores)
df = df[df["n_words"] >= 15]
n_short = n0 - len(df)
df = df.drop_duplicates("clean")
n_dups = n0 - n_short - len(df)
df = df.reset_index(drop=True)
df.to_csv(DATA / "clean.csv.gz", index=False)
print(f"rows loaded {n0:,}; dropped {n_short:,} very short (<15 words) and {n_dups:,} exact-duplicate texts -> {len(df):,} analysis rows")
print(f"date range {df.date.min().date()} to {df.date.max().date()}; median words {df.n_words.median():.0f}; mean {df.n_words.mean():.0f}")
print("\nProduct group share:\n", (df.product_group.value_counts(normalize=True)*100).round(1).to_string())
print(f"\nMedical-debt sub-product share: {df.is_medical_debt.mean()*100:.1f}%  (n={df.is_medical_debt.sum():,})")
print("\nTop issues:\n", df.Issue.value_counts().head(10).to_string())
print("\nTop companies:\n", df.Company.value_counts().head(10).to_string())

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
df.product_group.value_counts().plot.barh(ax=axes[0, 0], color="#2b6cb0"); axes[0, 0].invert_yaxis(); axes[0, 0].set_title("Complaints by product group")
df.Issue.value_counts().head(10).plot.barh(ax=axes[0, 1], color="#c05621"); axes[0, 1].invert_yaxis(); axes[0, 1].set_title("Top 10 issues (consumer-selected)")
axes[0, 1].set_yticklabels([l.get_text()[:45] for l in axes[0, 1].get_yticklabels()])
m = df.set_index("date").resample("MS").size()
m.plot(ax=axes[1, 0], marker="o", color="#2f855a"); axes[1, 0].set_title("Monthly complaint volume (medical-related narratives)"); axes[1, 0].set_xlabel("")
sns.histplot(df.n_words.clip(upper=1500), bins=50, ax=axes[1, 1], color="#6b46c1"); axes[1, 1].set_title("Narrative length (words, clipped at 1500)")
savefig(fig, "01_eda_overview.png")

# top terms overall and top terms for medical-debt vs other
cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=20, max_features=20000)
X = cv.fit_transform(df.clean); terms = np.array(cv.get_feature_names_out())
freq = np.asarray(X.sum(0)).ravel()
top = pd.Series(freq, index=terms).sort_values(ascending=False).head(25)
fig, ax = plt.subplots(figsize=(8, 7)); top[::-1].plot.barh(ax=ax, color="#2b6cb0"); ax.set_title("Top 25 terms / bigrams (all narratives)")
savefig(fig, "02_top_terms_overall.png")
print("\nTop 15 terms:", ", ".join(top.index[:15]))
