"""Topic modelling with NMF on TF-IDF to surface contact drivers / friction points.
k is chosen by gensim c_v coherence over a grid; topics are then labelled by a human from top words + sample narratives
(labels live in topic_labels.json after manual review)."""
import json, os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from common import *

sns.set_theme(style="whitegrid")
df = load_clean()
DOMAIN_STOP = {"dollaramount", "xxxx", "don", "ve", "just", "like", "said", "told", "called", "would", "also", "one", "please", "im", "ive",
               "didn", "dont", "get", "got", "did", "does", "doesn", "isn", "wasn", "ll", "re", "us", "even", "still", "make", "made", "sure",
               "going", "way", "want", "need", "know", "use", "used", "case", "ago", "day", "days", "month", "months", "year", "years",
               "time", "times", "consumer", "reporting", "agency", "report", "complaint", "company", "information"}
stop = list(ENGLISH_STOP_WORDS | DOMAIN_STOP)
vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=15, max_df=0.35, sublinear_tf=True, max_features=30000, token_pattern=r"(?u)\b[a-z][a-z']{2,}\b")
X = vec.fit_transform(df["clean"]); terms = np.array(vec.get_feature_names_out())
print("TF-IDF matrix:", X.shape)

# coherence on unigram-tokenised corpus
tok = [[w for w in d.split() if w not in stop and len(w) > 2] for d in df["clean"]]
dic = Dictionary(tok); dic.filter_extremes(no_below=15, no_above=0.35)
def top_words(H, n=10):
    return [[w for w in terms[np.argsort(r)[::-1]] if " " not in w][:n] for r in H]
rows = []
for k in [6, 8, 10, 12, 14, 16]:
    m = NMF(n_components=k, init="nndsvd", random_state=SEED, max_iter=400).fit(X)
    cm = CoherenceModel(topics=top_words(m.components_), texts=tok, dictionary=dic, coherence="c_v", processes=1).get_coherence()
    rows.append(dict(k=k, coherence_cv=round(cm, 4), recon_err=round(m.reconstruction_err_, 3))); print(rows[-1], flush=True)
sel = pd.DataFrame(rows); sel.to_csv(OUT / "topic_k_selection.csv", index=False)
# k=6 has the top c_v but collapses into 2 giant generic topics; for an actionable driver list we pick the best-coherence k >= 10
K = int(os.environ.get("N_TOPICS", sel[sel.k >= 10].sort_values("coherence_cv", ascending=False).iloc[0].k))
print("chosen K =", K)
nmf = NMF(n_components=K, init="nndsvd", random_state=SEED, max_iter=600)
W = nmf.fit_transform(X); H = nmf.components_
df["topic"] = W.argmax(1); df["topic_strength"] = W.max(1) / W.sum(1).clip(min=1e-9)
tw = [[w for w in terms[np.argsort(r)[::-1]]][:12] for r in H]
labels_path = os.path.join(os.path.dirname(__file__), "topic_labels.json")
labels = json.load(open(labels_path)) if os.path.exists(labels_path) else {}
themes = {int(k): v["theme"] for k, v in labels.items()}
labels = {int(k): v["label"] for k, v in labels.items()}
if not labels: labels = {i: f"Topic {i}" for i in range(K)}; themes = dict(labels)
df["topic_label"] = df["topic"].map(labels); df["theme"] = df["topic"].map(themes)
print("\nTOPICS (top terms, share of complaints):")
share = df.topic.value_counts(normalize=True).sort_index()
for i in range(K):
    print(f"[{i}] {labels.get(i, '?')}  ({share.get(i, 0)*100:.1f}%, n={int((df.topic==i).sum()):,}): {', '.join(tw[i])}")
if os.environ.get("SHOW_SAMPLES"):
    for i in range(K):
        print(f"\n--- TOPIC {i} samples")
        for t in df[df.topic == i].sort_values("topic_strength", ascending=False).head(2).narrative: print("   *", t[:350].replace("\n", " "))
df[["Complaint ID", "topic", "topic_label", "theme", "topic_strength"]].to_csv(OUT / "topics_assigned.csv", index=False)
json.dump({"K": K, "top_terms": {i: tw[i] for i in range(K)}, "share": {int(k): float(v) for k, v in share.items()}}, open(OUT / "topics.json", "w"), indent=1)

print("\nTHEME share (%):\n", (df.theme.value_counts(normalize=True)*100).round(1).to_string())
# figures
order = share.sort_values().index
fig, ax = plt.subplots(figsize=(10, 0.5 * K + 2))
ax.barh([labels[i] for i in order], share[order] * 100, color="#2b6cb0"); ax.set_xlabel("% of complaints (dominant topic)")
for y, v in enumerate(share[order] * 100): ax.text(v + 0.2, y, f"{v:.1f}%", va="center")
ax.set_title(f"Topic prevalence - {K} NMF topics across {len(df):,} medical-related complaints"); savefig(fig, "07_topic_prevalence.png")

fig, axes = plt.subplots((K + 3) // 4, 4, figsize=(20, 3.2 * ((K + 3) // 4)))
for i, ax in enumerate(axes.ravel()):
    if i >= K: ax.axis("off"); continue
    w = H[i][np.argsort(H[i])[::-1][:8]][::-1]; ax.barh(tw[i][:8][::-1], w, color="#c05621"); ax.set_title(labels[i][:42], fontsize=10); ax.tick_params(labelsize=8)
savefig(fig, "08_topic_top_terms.png")
