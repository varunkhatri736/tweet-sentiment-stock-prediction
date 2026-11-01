"""Train and evaluate every model; write results/*.csv and predictions.

Model selection uses the validation split only; the test split is touched once
per final configuration. Neural models are run with 5 seeds.

Usage: python models.py
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

import features as F

ROOT = Path(__file__).parent
LAYER = int(os.environ.get("EMB_LAYER", 4))  # which RoBERTa layer's embeddings
RES = ROOT / "results" / f"L{LAYER}"
RES.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)

S, tw, lex = F.load()
emb = np.load(F.DATA / f"roberta_emb_L{LAYER}.npy")
y = S["label"].values
tr, va, te = (S["split"] == s for s in ("train", "val", "test"))
tr, va, te = tr.values, va.values, te.values
PREDS = {}   # name -> test-set probability of "up"
ROWS = []


def score(name, group, p_val, p_test, extra=None):
    yv, yt = y[va], y[te]
    row = dict(model=name, group=group,
               val_acc=accuracy_score(yv, p_val > .5), val_mcc=matthews_corrcoef(yv, p_val > .5),
               test_acc=accuracy_score(yt, p_test > .5),
               test_mcc=matthews_corrcoef(yt, p_test > .5),
               test_pred_up=float((p_test > .5).mean()))
    row.update(extra or {})
    ROWS.append(row)
    PREDS[name] = p_test
    print(f"{name:42s} val acc {row['val_acc']:.3f} mcc {row['val_mcc']:+.3f} | "
          f"test acc {row['test_acc']:.3f} mcc {row['test_mcc']:+.3f}", flush=True)


def logreg(name, group, X, Cs=(0.001, 0.01, 0.1, 1.0, 10.0), scale=True):
    X = np.asarray(X, dtype=np.float64)
    if scale:
        sc = StandardScaler().fit(X[tr])
        X = sc.transform(X)
    best = None
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=3000).fit(X[tr], y[tr])
        pv = m.predict_proba(X[va])[:, 1]
        acc = accuracy_score(y[va], pv > .5)
        if best is None or acc > best[0]:
            best = (acc, C, m)
    _, C, m = best
    score(name, group, m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1],
          {"C": C})
    return m


# ------------------------------------------------------------------ baselines
score("Majority class (always up)", "Baseline", np.ones(va.sum()), np.ones(te.sum()))
rng = np.random.default_rng(0)
score("Random guess", "Baseline", rng.random(va.sum()), rng.random(te.sum()))

P = F.price_features(S)
logreg("Price-only LogReg", "Price only", P)

# ------------------------------------------------------------------ text: strict
for setting in ("strict", "calendar"):
    tag = "" if setting == "strict" else " [calendar cutoff]"
    Lx = F.lexicon_features(S, lex, setting)
    logreg(f"Lexicon sentiment LogReg{tag}", "Lexicon sentiment", Lx)
    logreg(f"Lexicon sentiment + price LogReg{tag}", "Lexicon sentiment",
           np.hstack([Lx, P]))

    texts = F.window_texts(S, tw, setting)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000,
                          sublinear_tf=True)
    Xtr = vec.fit_transform([texts[i] for i in np.where(tr)[0]])
    Xall = vec.transform(texts)
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        m = LogisticRegression(C=C, max_iter=3000).fit(Xtr, y[tr])
        acc = accuracy_score(y[va], m.predict_proba(Xall[va])[:, 1] > .5)
        if best is None or acc > best[0]:
            best = (acc, C, m)
    _, C, m = best
    score(f"TF-IDF LogReg{tag}", "Bag-of-words", m.predict_proba(Xall[va])[:, 1],
          m.predict_proba(Xall[te])[:, 1], {"C": C})

    Rf = F.roberta_features(S, emb, setting)
    logreg(f"RoBERTa (frozen) LogReg{tag}", "RoBERTa", Rf,
           Cs=(0.0001, 0.001, 0.01, 0.1))
    logreg(f"RoBERTa (frozen) + price LogReg{tag}", "RoBERTa", np.hstack([Rf, P]),
           Cs=(0.0001, 0.001, 0.01, 0.1))


# ------------------------------------------------------------------ LSTMs
class SeqModel(nn.Module):
    def __init__(self, use_text, use_lex, d_text=768, d_lex=8, d_price=3, hid=32,
                 proj=32, drop=0.3):
        super().__init__()
        self.use_text, self.use_lex = use_text, use_lex
        d_in = d_price
        if use_text:
            self.text_proj = nn.Sequential(nn.Dropout(drop), nn.Linear(d_text, proj),
                                           nn.Tanh())
            d_in += proj
        if use_lex:
            d_in += d_lex
        self.lstm = nn.LSTM(d_in, hid, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hid, 1))

    def forward(self, text, lexd, price):
        parts = [price]
        if self.use_text:
            parts.append(self.text_proj(text))
        if self.use_lex:
            parts.append(lexd)
        h, _ = self.lstm(torch.cat(parts, -1))
        return self.head(h[:, -1]).squeeze(-1)


def train_seq(use_text, use_lex, seed, epochs=30, lr=1e-3, wd=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = SeqModel(use_text, use_lex)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    idx = np.where(tr)[0]
    yt = torch.tensor(y, dtype=torch.float32)
    best = (-1, None, None)
    for ep in range(epochs):
        m.train()
        rng = np.random.default_rng(seed * 100 + ep)
        for b in np.array_split(rng.permutation(idx), len(idx) // 128):
            opt.zero_grad()
            loss = lossf(m(T[b], Ld[b], Pr[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(T[va], Ld[va], Pr[va])).numpy()
            acc = accuracy_score(y[va], pv > .5)
            if acc > best[0]:  # early stopping on validation accuracy
                pt = torch.sigmoid(m(T[te], Ld[te], Pr[te])).numpy()
                best = (acc, pv, pt)
    return best


T_np, L_np, P_np = F.sequence_inputs(S, emb, lex)
# standardise with training statistics
for arr in (T_np, L_np, P_np):
    mu = arr[tr].reshape(-1, arr.shape[-1]).mean(0)
    sd = arr[tr].reshape(-1, arr.shape[-1]).std(0) + 1e-6
    arr -= mu
    arr /= sd
T, Ld, Pr = (torch.tensor(a) for a in (T_np, L_np, P_np))

for name, use_text, use_lex in [("LSTM price-only", False, False),
                                ("LSTM price + lexicon", False, True),
                                ("LSTM price + RoBERTa", True, False),
                                ("LSTM price + RoBERTa + lexicon", True, True)]:
    t0 = time.time()
    runs = [train_seq(use_text, use_lex, seed) for seed in range(5)]
    pv = np.mean([r[1] for r in runs], 0)   # seed-ensemble probabilities
    pt = np.mean([r[2] for r in runs], 0)
    accs = [accuracy_score(y[te], r[2] > .5) for r in runs]
    mccs = [matthews_corrcoef(y[te], r[2] > .5) for r in runs]
    score(name, "LSTM", pv, pt, {"seed_test_acc_mean": np.mean(accs),
                                 "seed_test_acc_sd": np.std(accs),
                                 "seed_test_mcc_mean": np.mean(mccs),
                                 "seed_test_mcc_sd": np.std(mccs)})
    print(f"   per-seed test acc {np.mean(accs):.3f}±{np.std(accs):.3f}, "
          f"mcc {np.mean(mccs):+.3f}±{np.std(mccs):.3f}  ({time.time() - t0:.0f}s)")

pd.DataFrame(ROWS).to_csv(RES / "results.csv", index=False)
out = S.loc[te, ["stock", "date", "ret", "label"]].reset_index(drop=True)
for k, v in PREDS.items():
    out[k] = v
out.to_parquet(RES / "test_predictions.parquet")
print("saved results")
