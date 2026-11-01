"""How much directional information do StockNet tweets carry at all?

Measures (1) the share of multi-ticker / templated tweets and (2) the rank
correlation between a stock-day's mean tweet sentiment and that day's return,
using tweets from the *same* UTC day (leaky on purpose: an upper bound) and from
the previous day. Writes results/data_quality.json.
"""
import json
import re
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

import features as F

S, tw, lex = F.load()
tw = tw.assign(vader=lex["vader_compound"].values, d=tw["created_at"].dt.normalize())
cash = tw["text"].map(lambda t: len(set(re.findall(r"\$[A-Za-z][A-Za-z.\-]{0,5}", t))))
tw["multi"] = cash >= 3
tw["has_link"] = tw["text"].str.contains("http")
tw["neutral"] = tw["vader"].abs() < 0.05

out = {"n_tweets": int(len(tw)),
       "share_3plus_cashtags": float(tw["multi"].mean()),
       "share_with_link": float(tw["has_link"].mean()),
       "share_vader_neutral": float(tw["neutral"].mean())}


def corr(sub, shift_days):
    g = sub.groupby(["stock", "d"])["vader"].mean().rename("v").reset_index()
    m = S.assign(d=S["date"] - pd.Timedelta(days=shift_days)).merge(g, on=["stock", "d"])
    r, p = spearmanr(m["v"], m["ret"])
    return {"spearman": float(r), "p": float(p), "n": int(len(m))}


for label, sub in [("all", tw), ("single_ticker_only", tw[cash <= 1])]:
    out[f"same_day_{label}"] = corr(sub, 0)
    out[f"prev_day_{label}"] = corr(sub, 1)
res = Path(__file__).parent / "results"
res.mkdir(exist_ok=True)
(res / "data_quality.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
