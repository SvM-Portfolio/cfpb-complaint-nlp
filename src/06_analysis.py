"""Sentiment by topic/segment, time trends, repeat-complaint language, company comparison, outcomes by driver."""
import re, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, numpy as np, pandas as pd
from scipy import stats
from common import *

sns.set_theme(style="whitegrid")
df = load_clean()
s = pd.read_csv(OUT / "sentiment.csv"); t = pd.read_csv(OUT / "topics_assigned.csv")
df = df.merge(s, on="Complaint ID").merge(t, on="Complaint ID")
df["quarter"] = df["date"].dt.to_period("Q").astype(str); df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
df["segment"] = np.where(df.is_medical_debt == 1, "Medical-debt complaints", "Other medical-mentioning complaints")
print(f"n = {len(df):,}")

# ---------------- sentiment ----------------
print("\n== SENTIMENT ==")
print(f"VADER (mean sentence compound): mean={df.vader_sent_mean.mean():.3f}, median={df.vader_sent_mean.median():.3f}; share < 0: {(df.vader_sent_mean<0).mean()*100:.1f}%")
print(f"DistilBERT SST-2: share classified NEGATIVE = {(1-df.distilbert_pos.mean())*100:.1f}%; mean signed prob = {df.distilbert_signed.mean():.3f}")
rho, p = stats.spearmanr(df.vader_sent_mean, df.distilbert_signed)
print(f"Spearman correlation VADER vs DistilBERT: rho={rho:.3f} (p={p:.2g})")
agree = ((df.vader_sent_mean < 0) == (df.distilbert_signed < 0)).mean()
print(f"Sign agreement (neg vs not-neg): {agree*100:.1f}%")
print("\nBy theme:"); g = df.groupby("theme").agg(n=("theme", "size"), vader=("vader_sent_mean", "mean"), bert_neg_pct=("distilbert_pos", lambda x: (1 - x.mean()) * 100)).sort_values("vader")
print(g.round(3).to_string())
H, pk = stats.kruskal(*[x.vader_sent_mean.values for _, x in df.groupby("theme")]); print(f"Kruskal-Wallis across themes (VADER): H={H:.1f}, p={pk:.2g}")
print("\nBy segment:"); print(df.groupby("segment").agg(n=("segment", "size"), vader=("vader_sent_mean", "mean"), bert_neg_pct=("distilbert_pos", lambda x: (1 - x.mean()) * 100)).round(3).to_string())
a, b = df[df.is_medical_debt == 1].vader_sent_mean, df[df.is_medical_debt == 0].vader_sent_mean
print(f"Mann-Whitney medical-debt vs other (VADER): p={stats.mannwhitneyu(a, b).pvalue:.2g}; mean diff={a.mean()-b.mean():+.3f}")
print("\nBy product group:"); print(df.groupby("product_group").agg(n=("theme", "size"), vader=("vader_sent_mean", "mean"), bert_neg_pct=("distilbert_pos", lambda x: (1 - x.mean()) * 100)).round(3).to_string())

order = g.index.tolist()
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
sns.boxplot(data=df, y="theme", x="vader_sent_mean", order=order, ax=ax[0], color="#90cdf4", showfliers=False); ax[0].axvline(0, color="k", lw=.8)
ax[0].set_title("Sentiment by complaint theme (VADER, mean sentence score)"); ax[0].set_xlabel("<- more negative     more positive ->"); ax[0].set_ylabel("")
bn = g["bert_neg_pct"].reindex(order); ax[1].barh(order, bn, color="#c05621"); ax[1].set_title("% narratives classified NEGATIVE (DistilBERT)"); ax[1].set_yticklabels([]); ax[1].set_xlim(0, 100)
for y, v in enumerate(bn): ax[1].text(v + 1, y, f"{v:.0f}%", va="center")
savefig(fig, "09_sentiment_by_theme.png")
tl = df.groupby("topic_label").vader_sent_mean.mean().sort_values()
fig, ax = plt.subplots(figsize=(9, 5.5)); ax.barh(tl.index, tl.values, color=["#c05621" if v < tl.median() else "#2b6cb0" for v in tl.values]); ax.axvline(0, color="k", lw=.8)
ax.set_title("Mean VADER sentiment by NMF topic (lower = angrier / more negative)"); savefig(fig, "10_sentiment_by_topic.png")

# ---------------- trends ----------------
print("\n== TRENDS ==")
q = df.groupby("quarter").size(); print("Complaints per quarter:\n", q.to_string())
sh = pd.crosstab(df.quarter, df.theme, normalize="index") * 100
print("\nTheme share by quarter (%):\n", sh.round(1).to_string())
mdq = df.groupby("quarter").is_medical_debt.mean() * 100; print("\nMedical-debt sub-product share by quarter (%):\n", mdq.round(1).to_string())
fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
sh.plot.area(ax=ax[0], colormap="tab20", alpha=.9); ax[0].set_ylabel("% of complaints"); ax[0].set_title("Complaint theme mix over time (by quarter)"); ax[0].legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=8)
sq = df.groupby("quarter").vader_sent_mean.mean(); print("\nMean VADER by quarter:\n", sq.round(3).to_string()); ax[1].plot(sq.index, sq.values, marker="o", label="mean VADER sentiment")
ax2 = ax[1].twinx(); ax2.bar(q.index, q.values, alpha=.2, color="gray"); ax2.set_ylabel("complaints (bars)"); ax[1].set_ylabel("mean sentiment"); ax[1].set_title("Sentiment and volume by quarter"); ax[1].tick_params(axis="x", rotation=45)
savefig(fig, "11_trends_by_quarter.png")

# ---------------- repeat-contact language ----------------
print("\n== REPEAT-COMPLAINT ANALYSIS ==")
REPEAT = re.compile(r"\b(again|repeatedly|multiple times|several times|many times|numerous times|second time|third time|over and over|yet again|"
                    r"previously (?:disputed|filed|submitted|contacted|complained|called|sent)|already (?:disputed|filed|submitted|called|contacted|sent|complained)|"
                    r"as i (?:stated|mentioned|said) (?:before|previously)|last (?:month|year) i (?:disputed|filed|complained|called))\b", re.I)
df["repeat_lang"] = df["narrative"].map(lambda x: int(bool(REPEAT.search(x))))
print(f"Overall: {df.repeat_lang.mean()*100:.1f}% of narratives contain explicit repeat-contact language ({df.repeat_lang.sum():,} of {len(df):,})")
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d, (c + r) / d
def rate_table(col, minn=1):
    r = df.groupby(col).agg(n=("repeat_lang", "size"), k=("repeat_lang", "sum")); r = r[r.n >= minn]
    r["repeat_pct"] = r.k / r.n * 100; ci = [wilson(k, n) for k, n in zip(r.k, r.n)]; r["ci_lo"] = [c[0] * 100 for c in ci]; r["ci_hi"] = [c[1] * 100 for c in ci]
    return r.sort_values("repeat_pct", ascending=False)
byt = rate_table("theme"); print("\nRepeat-language rate by theme:\n", byt.round(1).to_string())
byp = rate_table("product_group"); print("\nRepeat-language rate by product group:\n", byp.round(1).to_string())
df["company_short"] = df.Company.str.replace(r",? (INC\.?|N\.A\.|LLC|CORPORATION|HOLDINGS|INTERMEDIATE)\b.*", "", regex=True).str.title()
topc = df.Company.value_counts(); topc = topc[topc >= 250].index
bc = rate_table("company_short", 1); bc = bc.loc[[c for c in bc.index if c in df[df.Company.isin(topc)].company_short.unique()]]
print("\nRepeat-language rate by company (>=250 complaints):\n", bc.round(1).to_string())
chi = stats.chi2_contingency(pd.crosstab(df[df.Company.isin(topc)].company_short, df[df.Company.isin(topc)].repeat_lang)); print(f"Chi-square test, repeat language differs across these companies: chi2={chi[0]:.1f}, p={chi[1]:.2g}")
# repeat language -> sentiment
print(f"\nMean VADER: repeat-language {df[df.repeat_lang==1].vader_sent_mean.mean():.3f} vs other {df[df.repeat_lang==0].vader_sent_mean.mean():.3f}")
# 'problem with investigation of an existing problem' = formally a repeat-contact issue
INV = "Problem with a company's investigation into an existing problem"
inv = df.groupby("company_short").agg(n=("Issue", "size"), inv_pct=("Issue", lambda x: (x == INV).mean() * 100)); inv = inv[inv.n >= 250].sort_values("inv_pct", ascending=False)
print(f"\nShare of complaints selecting '{INV}' (company-level, >=250):\n", inv.round(1).to_string())
print(f"Overall share selecting that issue: {(df.Issue==INV).mean()*100:.1f}%")
ci_pairs = df.groupby(["company_short", "Issue"]).size().sort_values(ascending=False).head(10); print("\nTop 10 company x issue pairs (repeat-issue hot spots):\n", ci_pairs.to_string())

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
bt = byt.sort_values("repeat_pct"); ax[0].barh(bt.index, bt.repeat_pct, xerr=[bt.repeat_pct - bt.ci_lo, bt.ci_hi - bt.repeat_pct], color="#2b6cb0"); ax[0].set_title("Narratives with explicit repeat-contact language, by theme (95% CI)"); ax[0].set_xlabel("%")
bb = bc.sort_values("repeat_pct"); ax[1].barh(bb.index, bb.repeat_pct, xerr=[bb.repeat_pct - bb.ci_lo, bb.ci_hi - bb.repeat_pct], color="#c05621"); ax[1].set_title("... by company (>=250 complaints, 95% CI)"); ax[1].set_xlabel("%")
savefig(fig, "12_repeat_complaints.png")

# ---------------- outcomes by driver ----------------
print("\n== COMPANY RESPONSE BY THEME ==")
df["relief"] = df["Company response to consumer"].str.contains("relief", case=False, na=False).astype(int)
oc = df.groupby("theme").agg(n=("relief", "size"), relief_pct=("relief", lambda x: x.mean() * 100), explanation_pct=("Company response to consumer", lambda x: (x == "Closed with explanation").mean() * 100)); print(oc.round(1).sort_values("relief_pct", ascending=False).to_string())
print("\nResponse mix overall (%):\n", (df["Company response to consumer"].value_counts(normalize=True) * 100).round(1).to_string())
fig, ax = plt.subplots(figsize=(9, 4.5)); o = oc.sort_values("relief_pct"); ax.barh(o.index, o.relief_pct, color="#2f855a"); ax.set_title("% complaints closed with monetary/non-monetary relief, by theme"); ax.set_xlabel("%")
savefig(fig, "13_relief_by_theme.png")
df[["Complaint ID", "theme", "topic_label", "vader_sent_mean", "distilbert_signed", "repeat_lang", "relief"]].to_csv(OUT / "complaint_level_scores.csv.gz", index=False)

# ---------------- healthcare-financing deep dive ----------------
print("\n== HEALTHCARE FINANCING (CareCredit-style cards) ==")
cc = df[df["narrative"].str.contains(r"care ?credit", case=False, regex=True)]
print(f"Narratives mentioning CareCredit: {len(cc):,} ({len(cc)/len(df)*100:.1f}% of sample)")
print("Product:", cc.product_group.value_counts().to_dict())
print("Top issues:\n", cc.Issue.value_counts().head(5).to_string())
di = cc["narrative"].str.contains(r"deferred interest|promotional|no interest|0% interest|retroactive", case=False, regex=True)
print(f"Of these, {di.mean()*100:.1f}% mention deferred / promotional / retroactive interest terms")
print(f"Mean VADER {cc.vader_sent_mean.mean():.3f} vs overall {df.vader_sent_mean.mean():.3f}; repeat-language {cc.repeat_lang.mean()*100:.1f}% vs overall {df.repeat_lang.mean()*100:.1f}%")
md = df[df.is_medical_debt == 1]
ins = md["narrative"].str.contains(r"insurance", case=False)
print(f"\nMedical-debt complaints (n={len(md):,}): {ins.mean()*100:.1f}% mention insurance; "
      f"{md['narrative'].str.contains(r'never (?:received|got|notified)|did not receive|no notice|wasn.t notified|without notice', case=False, regex=True).mean()*100:.1f}% say they never received a bill/notice; "
      f"{md['narrative'].str.contains(r'no surprises act|surprise bill|out of network|out-of-network', case=False, regex=True).mean()*100:.1f}% mention surprise / out-of-network billing; "
      f"{md['narrative'].str.contains(r'already paid|paid (?:it |this |the bill |the debt |in full|off)|i paid|was paid', case=False, regex=True).mean()*100:.1f}% say they already paid the bill.")
