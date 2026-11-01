"""Positive control: deliberately leak target-day tweets.

Uses tweets posted between close(d-1) and close(d), i.e. *during* the move being
predicted. A working text pipeline should score far above chance here; if it
does, the near-chance results in the honest setting are not caused by broken
features. Writes results/positive_control.csv.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

import features as F

ROOT = Path(__file__).parent
S, tw, lex = F.load()
LAYER = int(os.environ.get("EMB_LAYER", 4))
emb = np.load(F.DATA / f"roberta_emb_L{LAYER}.npy").astype(np.float32)

leak = []
for stock, g in S.groupby("stock"):
    p = pd.read_csv(F.STOCKNET / "price" / "raw" / f"{stock}.csv",
                    parse_dates=["Date"]).sort_values("Date")["Date"].reset_index(drop=True)
    prev_of = dict(zip(p.iloc[1:].values, p.iloc[:-1].values))
    st = tw[tw["stock"] == stock]
    t, ids = st["created_at"].values, st["tid"].values
    for i, d in zip(g.index, g["date"].values):
        lo = prev_of[d] + np.timedelta64(20, "h")
        hi = d + np.timedelta64(20, "h")
        leak.append((i, ids[np.searchsorted(t, lo):np.searchsorted(t, hi)].tolist()))
S["tids_leak"] = pd.Series(dict(leak))
S["tids_leak_last"] = S["tids_leak"]
keep = (S["tids_leak"].map(len) > 0).values
print(f"samples with target-day tweets: {keep.sum():,} of {len(S):,}")

from sklearn.feature_extraction.text import TfidfVectorizer

y = S["label"].values
tr = ((S["split"] == "train").values & keep)
va = ((S["split"] == "val").values & keep)
te = ((S["split"] == "test").values & keep)
texts = F.window_texts(S, tw, "leak")
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000, sublinear_tf=True)
vec.fit([texts[i] for i in np.where(tr)[0]])
rows = []
for name, X in [("Lexicon (target-day tweets)", F.lexicon_features(S, lex, "leak").values),
                ("TF-IDF (target-day tweets)", vec.transform(texts)),
                ("RoBERTa (target-day tweets)", F.roberta_features(S, emb, "leak"))]:
    if not hasattr(X, "tocsr"):
        X = StandardScaler().fit(X[tr]).transform(X)
    else:
        X = X.tocsr()
    best = max(((accuracy_score(y[va], m.predict(X[va])), C, m) for C in
                (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
                for m in [LogisticRegression(C=C, max_iter=3000).fit(X[tr], y[tr])]),
               key=lambda z: z[0])
    pred = best[2].predict(X[te])
    rows.append(dict(model=name, C=best[1], n_test=int(te.sum()),
                     test_acc=accuracy_score(y[te], pred),
                     test_mcc=matthews_corrcoef(y[te], pred),
                     majority_acc=max(y[te].mean(), 1 - y[te].mean())))
    print(rows[-1])
(ROOT / "results" / f"L{LAYER}").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / f"L{LAYER}" / "positive_control.csv", index=False)
