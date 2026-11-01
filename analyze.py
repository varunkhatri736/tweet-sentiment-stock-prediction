"""Date-clustered bootstrap CIs and paired comparisons on the test set.

Writes results/results_with_ci.csv and results/paired_vs_price.csv.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

RES = Path(__file__).parent / "results" / f"L{int(os.environ.get('EMB_LAYER', 4))}"
P = pd.read_parquet(RES / "test_predictions.parquet")
R = pd.read_csv(RES / "results.csv")
y = P["label"].values
dates = P["date"].values
udates = np.unique(dates)
groups = [np.where(dates == d)[0] for d in udates]
rng = np.random.default_rng(42)
B = 1000
boot_idx = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
            for _ in range(B)]


def ci(stat):
    vals = np.array([stat(ix) for ix in boot_idx])
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


rows = []
for name in R["model"]:
    pred = (P[name].values > .5).astype(int)
    acc_lo, acc_hi = ci(lambda ix: (pred[ix] == y[ix]).mean())
    mcc_lo, mcc_hi = ci(lambda ix: matthews_corrcoef(y[ix], pred[ix]))
    rows.append(dict(model=name, acc_ci_lo=acc_lo, acc_ci_hi=acc_hi,
                     mcc_ci_lo=mcc_lo, mcc_ci_hi=mcc_hi))
out = R.merge(pd.DataFrame(rows), on="model")
out.to_csv(RES / "results_with_ci.csv", index=False)

base = (P["Price-only LogReg"].values > .5).astype(int)
maj = np.ones_like(y)
pair = []
for name in R["model"]:
    pred = (P[name].values > .5).astype(int)
    for ref_name, ref in (("Price-only LogReg", base), ("Majority class (always up)", maj)):
        if name == ref_name:
            continue
        diffs = np.array([(pred[ix] == y[ix]).mean() - (ref[ix] == y[ix]).mean()
                          for ix in boot_idx])
        pair.append(dict(model=name, vs=ref_name,
                         acc_diff=(pred == y).mean() - (ref == y).mean(),
                         diff_ci_lo=np.percentile(diffs, 2.5),
                         diff_ci_hi=np.percentile(diffs, 97.5),
                         p_not_better=(diffs <= 0).mean()))
pd.DataFrame(pair).to_csv(RES / "paired.csv", index=False)

# Exploratory: is a model more accurate on its most confident 20% of calls?
conf = []
for name in R["model"]:
    p = P[name].values
    if np.unique(p).size < 10:   # constant/baseline predictors
        continue
    top = np.abs(p - .5) >= np.quantile(np.abs(p - .5), .8)
    acc_top = ((p[top] > .5) == y[top]).mean()
    lo, hi = ci(lambda ix: (((p[ix] > .5) == y[ix])[top[ix]]).mean())
    conf.append(dict(model=name, n_top=int(top.sum()), acc_top20=acc_top,
                     ci_lo=lo, ci_hi=hi, up_rate_top20=y[top].mean()))
pd.DataFrame(conf).to_csv(RES / "confidence_top20.csv", index=False)
pd.set_option("display.width", 200)
print(out[["model", "test_acc", "acc_ci_lo", "acc_ci_hi", "test_mcc", "mcc_ci_lo",
           "mcc_ci_hi"]].round(3).to_string(index=False))
print()
print(pd.DataFrame(pair).round(3).to_string(index=False))
print()
print(pd.DataFrame(conf).round(3).to_string(index=False))
print(f"\n{len(udates)} test trading days")
