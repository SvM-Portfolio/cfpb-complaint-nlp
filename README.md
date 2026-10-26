# What are people actually complaining about? NLP on 19,758 medical-related consumer complaints

Text analytics on real consumer complaint narratives to find **friction points and contact drivers**,
quantify **repeat-contact language**, and compare **sentiment** across themes, products, companies and time.
Healthcare angle: the sample is every public CFPB complaint narrative that mentions a medical-care term
(hospital, medical, doctor, dental, pharmacy, …). That surfaces **medical-debt, medical-billing and
healthcare-financing** problems (e.g. CareCredit-type cards) alongside general credit/banking issues.

> Built with scikit-learn, NMF topic modelling, VADER and DistilBERT. Everything below was produced by the code in `src/`;
> logs of the actual runs are in `outputs/*_log.txt`.

## 1. Business questions
1. What are the main **contact drivers** in medical-related complaints, and how big is each?
2. Can we **auto-route** a complaint (product / issue / medical-debt) from its free text alone?
3. Which drivers are the **angriest**, and which companies/themes show **repeat contact**?
4. How is the mix changing over time?

## 2. Data
* **Source:** CFPB Consumer Complaint Database - narratives archive (public domain), CFPB FOIA Reading Room:
  <https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/>
* **Important:** on 2026-08-14 the CFPB stopped publishing narratives in the live API / bulk CSV, so the pipeline downloads the
  FOIA-archive monthly exports (Sep 2023 - Aug 2026) from `files.consumerfinance.gov` automatically.
* **Funnel (actual run):** 13,495,502 complaints scanned → 2,323,440 have a consumer narrative → **21,611** narratives mention a
  medical-care term (regex in `src/config.py`) → drop 87 very short (<15 words) and 1,766 exact-duplicate texts →
  **19,758 analysis complaints** (2023-09-01 to 2026-07-29; median 233 words/narrative).
* **Mix:** Debt collection 34.9%, Credit reporting 28.8%, Credit card 11.5%, Bank account 9.1%, Mortgage 4.1%, other products 11.5%.
  26.6% (5,257) are filed under the *Medical debt* sub-product.
* No data was fabricated or simulated. Narratives are redacted by CFPB (`XXXX`), which the cleaner handles.

## 3. Methods
| Step | Script | What |
|---|---|---|
| Download + filter | `01_download_and_prepare.py` | Streams monthly archives, keeps narratives mentioning medical terms |
| Cleaning + EDA | `02_clean_and_eda.py` | Lowercase, strip redaction tokens/URLs/punctuation, `{$123}`→`dollaramount`, dedupe |
| Classification | `03_classify.py` | TF-IDF (1-2 grams) + Logistic Regression / LinearSVC. Stratified 80/20 split; C tuned with 5-fold CV on **train only**; test set used once. Majority-class baseline. |
| Sentiment | `04_sentiment.py` | VADER (mean of sentence scores) and DistilBERT SST-2 (first 256 tokens), CPU |
| Topics | `05_topics.py` | NMF on TF-IDF; k chosen by gensim c_v coherence; topics named by reading top words + sample complaints |
| Analysis | `06_analysis.py` | Sentiment by theme/segment, quarterly trends, repeat-contact language (regex lexicon + Wilson CIs, chi-square), company comparison, outcomes by theme |

## 4. Results (real outputs)

### 4.1 Classification (held-out test set, n = 3,952; 5-fold CV on 15,806 train rows)
| Task | Model | CV macro-F1 | Test accuracy | Test macro-F1 | Majority baseline macro-F1 |
|---|---|---|---|---|---|
| Product group (6 classes) | Logistic Regression (C=20) | 0.832 ± 0.005 | **0.839** | **0.831** | 0.086 |
| Product group | LinearSVC (C=1) | 0.832 ± 0.007 | 0.838 | 0.829 | |
| Issue group (top-8 issues + Other = 9 classes) | LinearSVC (C=1) | 0.589 ± 0.008 | **0.650** | **0.602** | 0.057 |
| Issue group | Logistic Regression (C=20) | 0.597 ± 0.008 | 0.648 | 0.599 | |

Product-group per-class F1 (LogReg): Mortgage 0.88, Debt collection 0.87, Credit reporting 0.85, Credit card 0.83, Bank account 0.81, Other products 0.75.
Issue-group is harder (the issue is chosen by the consumer from overlapping menu options): best classes are *Improper use of your report* (F1 0.80)
and *Purchase problem on statement* (0.69); weakest are *False statements or representation* (0.29) and *Written notification about debt* (0.44).

**Medical-debt complaint detection (binary, 26.6% positives):** Logistic Regression - 5-fold CV ROC-AUC 0.955; **test ROC-AUC 0.964**,
PR-AUC 0.887, accuracy 0.906; for the medical-debt class precision 0.781, recall 0.897, F1 0.835.
Top predictive terms: *debt, collection, insurance, billing, medical debt, visit, hospital, patient, covered, payment plan*.

Figures: `03_confusion_matrix_product.png`, `04_confusion_matrix_issue.png`, `05_roc_pr_medical_debt.png`, `06_top_terms_by_product.png`.

### 4.2 Topics / contact drivers (NMF, k = 12)
c_v coherence by k: 6 → 0.707, 8 → 0.693, 10 → 0.675, **12 → 0.703**, 14 → 0.682, 16 → 0.673. k = 6 scored marginally higher but merged
most complaints into two generic topics; I chose the best-coherence k ≥ 10 for an actionable driver list.

| Theme (topics grouped) | Share of complaints | Topics |
|---|---|---|
| **Medical billing & collections** | **42.4%** | Hospital & insurance billing sent to collections (23.5%); Medical-debt validation & FCRA/FDCPA disputes (18.9%) |
| **Payment hardship** | 18.2% | Loans, mortgage, late payments, hardship plans |
| **Account access & funds** | 13.6% | Bank/card account access, ATM and funds problems |
| **Dispute handling** | 12.1% | Dispute not properly reviewed / documentation ignored |
| **Templated statute-citation letters** | 7.7% | 6 small topics (0.5-3.1% each): copy-paste letters citing FCRA/privacy statutes |
| **Identity theft & fraud** | 6.0% | Fraudulent accounts / inquiries |

Figures: `07_topic_prevalence.png`, `08_topic_top_terms.png`.

### 4.3 Sentiment
* VADER (mean sentence compound): overall mean **-0.061**; 63.2% of narratives net-negative.
* DistilBERT SST-2 labels **98.6%** of narratives NEGATIVE - complaints are negative almost by definition, so the transformer has almost no
  discriminating power here. It correlates only weakly with VADER (Spearman ρ = 0.26; sign agreement 64.1%). **VADER sentence-mean is therefore used for comparisons**, and DistilBERT is reported as a sanity check.
* Most negative themes (VADER mean): Dispute handling -0.110, Identity theft & fraud -0.092, Medical billing & collections -0.088, Account access -0.069.
  Payment hardship (+0.010) and templated letters (+0.029) read as neutral (formal or polite wording). Differences across themes are significant (Kruskal-Wallis p < 1e-300).
* Medical-debt complaints are more negative than the rest: -0.098 vs -0.048 (Mann-Whitney p = 7.5e-69).
* By product: Bank account -0.107, Debt collection -0.100 are the most negative; Credit card -0.010, Credit reporting -0.024 the least.
* Caveat: absolute VADER differences are small (≈0.05-0.1 on a -1…1 scale); read them as relative rankings.

Figures: `09_sentiment_by_theme.png`, `10_sentiment_by_topic.png`.

### 4.4 Trends (quarterly)
* Volume peaked at 2,383 (2025 Q3) vs 1,371-1,711 per full quarter from 2023 Q4 to 2024 Q4; 2026 Q3 is a partial quarter (July only, 272).
* **Medical billing & collections** rose from ~35% of complaints (2023 Q3) to 51-54% (2025 Q1-Q2), then eased to 31% (2026 Q3 partial).
* **Templated statute-citation letters** collapsed from 25.6% (2023 Q3) to 3.1% (2025 Q1) and ~0% in 2026.
* **Dispute handling** (credit-card/bank dispute denials) grew from 2.4% to 32.7% of the mix (partly a mix effect as templated letters disappeared).
* Medical-debt sub-product share is stable: 23-30% every quarter.
* Mean VADER sentiment drifted from -0.021 (2023 Q3) to a low of -0.104 (2025 Q4); -0.081 to -0.092 in 2026. Note part of this is the shrinking share of polite/neutral templated letters.

Figure: `11_trends_by_quarter.png`.

### 4.5 Repeat-complaint analysis
There is no consumer ID in the data, so true repeat filers cannot be linked. Two proxies are used:
1. **Explicit repeat-contact language** ("again", "multiple times", "second time", "already disputed", …): **29.5%** (5,836) of narratives.
   By theme: Account access & funds 42.3%, Dispute handling 36.3%, Payment hardship 33.6%, Templated letters 27.8%, Medical billing 23.1%, Identity theft 22.1%.
   By product: Mortgage 44.9%, Bank account 38.9%, Credit card 36.8%, Credit reporting 25.2%, Debt collection 24.3%.
   By company (≥250 complaints): Citibank 47.4%, Wells Fargo 40.2%, JPMorgan Chase 40.0%, Bank of America 34.2% … Equifax 24.0%
   (χ² across companies = 170.8, p = 8e-31). 95% Wilson CIs are in `outputs/06_analysis_log.txt` and the figure.
2. **Issue = "Problem with a company's investigation into an existing problem"** (a formal "my earlier dispute failed" signal): 7.6% overall, but
   **25.3% Experian, 22.7% Equifax, 22.3% TransUnion** vs ≤5% for every bank/lender with ≥250 complaints.

Figure: `12_repeat_complaints.png`.

### 4.6 Healthcare-financing findings
* **Medical-debt complaints (n = 5,257):** 34.0% mention insurance; 16.2% say they never received a bill or notice; 14.6% say they already paid; only 2.7% mention surprise / out-of-network billing.
* **CareCredit-style cards:** 555 narratives (2.8%) mention CareCredit; 86% are credit-card complaints; top issues are *Fees or interest* (139) and
  *Problem with a purchase shown on your statement* (128); 32.3% reference deferred/promotional/retroactive interest.
* Share closed with relief (monetary or non-monetary): Templated letters 43.1%, Identity theft 31.9%, Account access 21.3%, Medical billing 21.2%, Dispute handling 16.3%, Payment hardship 16.3%. 76.8% of all complaints close "with explanation" only.

## 5. Findings and recommendations (for a business audience)
1. **Medical billing is the biggest single driver (42% of complaints; 23.5% are plain billing/insurance errors that end in collections).**
   Roughly 1 in 6 medical-debt complainants says they never got a bill. → Fix upstream: confirm insurance billing was attempted before
   referral to collections, and send/verify a first notice with an easy dispute path.
2. **Pause collections reporting while a dispute or insurance appeal is open.** Medical-debt complainants are measurably angrier than others, and 14.6% say they already paid.
3. **The credit bureaus have a "failed first fix" problem.** About one in four of their complaints is about an investigation of an existing dispute, vs ≤5% at banks.
   → Track re-dispute rate as a KPI and audit investigation quality (the 2nd contact is the expensive one).
4. **Banks/card issuers show the most explicit repeat-contact language (Citi 47%, Wells 40%, Chase 40%).** → Build "same issue within 30 days" flags and first-contact-resolution ownership; mine call/chat transcripts the same way.
5. **Healthcare financing products (CareCredit-type cards) generate fees/interest and deferred-interest complaints.** → Plain-language disclosure at the point of sale, aimed at clinics.
6. **Operationally, an NLP triage model is viable:** product routing at ~84% accuracy (macro-F1 0.83) and medical-debt flagging at ROC-AUC 0.96 from text alone, with no manual labelling.
7. **~8% of narratives are copy-paste statute letters** (down from ~26% in 2023); they read as neutral and need a different handling path (they are not "angry customers").

## 6. Limitations
* **Self-selected, one-sided sample.** CFPB complaints are unverified consumer allegations; they say nothing about rates of problems in the whole population.
  Only complaints with a published narrative (consumer consent) are included.
* **Medical filter is keyword-based** (e.g. "medical", "hospital", "dental"); it also catches complaints where the term appears incidentally (e.g. "medical emergency" causing mortgage hardship).
  The data is **financial-services**, not clinical patient-experience text, so conclusions concern medical-related financial friction.
* **Random train/test split:** near-duplicate template letters (only *exact* duplicates removed) can leak between splits and slightly inflate scores; a time-based split was not run.
* **Issue labels** are picked by consumers from overlapping menus; ~0.60 macro-F1 partly reflects label noise, not only model weakness.
* **Medical-debt label** = CFPB sub-product chosen at filing; the model detects what *consumers file as* medical debt.
* **Sentiment:** DistilBERT-SST2 saturates on complaint text (98.6% negative); VADER is a lexicon, ignores sarcasm and legal language, and 256-token truncation applies to DistilBERT.
* **Repeat-contact language** is a regex proxy (can over/under-count; negated mentions like "not again" are counted); no consumer ID exists to link repeat filers.
* **Topic names are my interpretation** of NMF word lists; NMF is deterministic here (`random_state=42`, NNDSVD init) but other k values would split topics differently.
* The theme shift in 2025-26 (templated letters vanish, dispute-handling grows) may reflect changes in consumer behaviour/consent, CFPB handling, or complaint-writing tools - the data cannot say which.
* 2026 Q3 holds only July; narratives were removed from the live database on 2026-08-14.

## 7. Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU torch is fine
bash run_all.sh                          # ~30 min on 8 cores (download ~1.1 GB, DistilBERT ~17 min)
```
Or run steps individually: `cd src && python 01_download_and_prepare.py && python 02_clean_and_eda.py && python 03_classify.py && python 04_sentiment.py && python 05_topics.py && python 06_analysis.py`.
Raw archives go to `data/raw/` (git-ignored). Seeds are fixed (`SEED = 42`).

## 8. Repo layout
```
src/            01-06 scripts, config.py, common.py, topic_labels.json (human-assigned names)
outputs/        figures/*.png, *_log.txt (real run logs), classification_results.json, topics.json, scores CSVs
data/           medical_complaints.csv.gz (21,611 filtered rows), clean.csv.gz (analysis table); raw/ is git-ignored
INTERVIEW_NOTES.md   RESUME_BULLETS.txt   run_all.sh   requirements.txt
```
