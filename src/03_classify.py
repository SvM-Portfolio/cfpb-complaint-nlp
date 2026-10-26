"""Supervised text classification: TF-IDF + LinearSVC / Logistic Regression.
Tasks: (A) product group, (B) issue group, (C) binary 'medical debt' detection (ROC-AUC).
Proper protocol: stratified 80/20 train/test split; hyper-parameters chosen by 5-fold CV on TRAIN only;
test set touched once at the end."""
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score,
                             roc_curve, average_precision_score, precision_recall_curve, ConfusionMatrixDisplay)
from common import *

sns.set_theme(style="whitegrid")
df = load_clean()
results = {}

def tfidf():
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5, max_df=0.5, sublinear_tf=True, max_features=100_000)

def run_multiclass(name, y, fig_name):
    print(f"\n{'='*70}\nTASK: {name}\n{'='*70}")
    Xtr, Xte, ytr, yte = train_test_split(df["clean"], y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"train={len(Xtr):,} test={len(Xte):,}; classes={y.nunique()}")
    print("class balance (%):", (y.value_counts(normalize=True) * 100).round(1).to_dict())
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = {
        "Majority baseline": (DummyClassifier(strategy="most_frequent"), None),
        "LogReg": (LogisticRegression(max_iter=2000, class_weight="balanced"), {"clf__C": [1, 5, 20]}),
        "LinearSVC": (LinearSVC(class_weight="balanced"), {"clf__C": [0.1, 0.3, 1]}),
    }
    fitted, rows = {}, []
    for mname, (est, grid) in models.items():
        pipe = Pipeline([("tfidf", tfidf()), ("clf", est)])
        if grid:
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=4).fit(Xtr, ytr)
            best, cvs, cvstd, par = gs.best_estimator_, gs.best_score_, gs.cv_results_["std_test_score"][gs.best_index_], gs.best_params_
        else:
            best = pipe.fit(Xtr, ytr); sc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1_macro"); cvs, cvstd, par = sc.mean(), sc.std(), {}
        pred = best.predict(Xte)
        rows.append(dict(model=mname, params=par, cv_macro_f1=round(cvs, 4), cv_std=round(cvstd, 4),
                         test_accuracy=round(accuracy_score(yte, pred), 4), test_macro_f1=round(f1_score(yte, pred, average="macro"), 4),
                         test_weighted_f1=round(f1_score(yte, pred, average="weighted"), 4)))
        fitted[mname] = (best, pred)
    res = pd.DataFrame(rows); print(res.to_string(index=False))
    best_name = res[res.model != "Majority baseline"].sort_values("test_macro_f1", ascending=False).iloc[0].model
    best, pred = fitted[best_name]
    print(f"\nBest model on CV/test: {best_name}\n")
    rep = classification_report(yte, pred, digits=3); print(rep)
    # confusion matrix (row-normalised)
    labels = sorted(y.unique())
    cm = confusion_matrix(yte, pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 5, 1.0 * len(labels) + 4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=[l[:28] for l in labels], yticklabels=[l[:40] for l in labels], ax=ax, cbar=False)
    ax.set_title(f"{name}: confusion matrix on held-out test set\n({best_name}, row-normalised = recall)"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=40, ha="right"); savefig(fig, fig_name)
    # top features per class
    vec, clf = best.named_steps["tfidf"], best.named_steps["clf"]
    fn = np.array(vec.get_feature_names_out()); tops = {}
    for i, c in enumerate(clf.classes_):
        tops[c] = fn[np.argsort(clf.coef_[i])[::-1][:12]].tolist()
        print(f"  top terms [{c[:45]}]: {', '.join(tops[c][:10])}")
    results[name] = dict(table=res.to_dict("records"), best=best_name, report=rep, top_terms=tops, n_train=len(Xtr), n_test=len(Xte))
    return tops

# (A) product group
tA = run_multiclass("Product group", df["product_group"], "03_confusion_matrix_product.png")

# (B) issue group: top-8 issues + Other
top_issues = df.Issue.value_counts().head(8).index
df["issue_group"] = np.where(df.Issue.isin(top_issues), df.Issue, "Other issue")
tB = run_multiclass("Issue group", df["issue_group"], "04_confusion_matrix_issue.png")

# (C) binary: medical-debt complaint?
print(f"\n{'='*70}\nTASK: Medical-debt complaint detection (binary)\n{'='*70}")
y = df["is_medical_debt"]
Xtr, Xte, ytr, yte = train_test_split(df["clean"], y, test_size=0.2, stratify=y, random_state=SEED)
pipe = Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
gs = GridSearchCV(pipe, {"clf__C": [1, 5, 20]}, cv=cv, scoring="roc_auc", n_jobs=4).fit(Xtr, ytr)
best = gs.best_estimator_
p = best.predict_proba(Xte)[:, 1]; pred = (p >= 0.5).astype(int)
auc, ap = roc_auc_score(yte, p), average_precision_score(yte, p)
print(f"train={len(Xtr):,} test={len(Xte):,}; positive rate={y.mean()*100:.1f}%; best C={gs.best_params_['clf__C']}")
print(f"5-fold CV ROC-AUC (train): {gs.best_score_:.4f}")
print(f"TEST ROC-AUC={auc:.4f}  PR-AUC(avg precision)={ap:.4f}  accuracy={accuracy_score(yte, pred):.4f}")
rep = classification_report(yte, pred, target_names=["other", "medical debt"], digits=3); print(rep)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fpr, tpr, _ = roc_curve(yte, p); ax[0].plot(fpr, tpr, lw=2, label=f"LogReg (AUC={auc:.3f})"); ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].set_title("ROC - medical-debt complaint detection"); ax[0].legend()
pr, rc, _ = precision_recall_curve(yte, p); ax[1].plot(rc, pr, lw=2, label=f"AP={ap:.3f}"); ax[1].axhline(yte.mean(), color="k", ls="--", label="baseline (prevalence)"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall"); ax[1].legend()
savefig(fig, "05_roc_pr_medical_debt.png")
fn = np.array(best.named_steps["tfidf"].get_feature_names_out()); coef = best.named_steps["clf"].coef_[0]
print("Top terms -> medical debt:", ", ".join(fn[np.argsort(coef)[::-1][:15]]))
print("Top terms -> other:", ", ".join(fn[np.argsort(coef)[:15]]))
results["Medical debt (binary)"] = dict(cv_auc=gs.best_score_, test_auc=auc, test_pr_auc=ap, report=rep)

# top-term figure: 6 product groups x top terms
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, (c, terms) in zip(axes.ravel(), tA.items()):
    ax.barh(range(10)[::-1], range(10, 0, -1), color="#2b6cb0"); ax.set_yticks(range(10)[::-1]); ax.set_yticklabels(terms[:10]); ax.set_xticks([]); ax.set_title(c)
fig.suptitle("Most discriminative terms per product group (LinearSVC/LogReg coefficients, ranked)", y=1.02)
savefig(fig, "06_top_terms_by_product.png")
json.dump(results, open(OUT / "classification_results.json", "w"), indent=1, default=str)
