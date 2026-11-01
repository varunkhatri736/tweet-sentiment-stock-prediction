"""Sanity checks that guard against look-ahead leakage. Run: python test_pipeline.py"""
import numpy as np
import pandas as pd

import features as F

S, tw, _ = F.load(with_lexicon=False)
created = tw.set_index("tid")["created_at"]
stock_of = tw.set_index("tid")["stock"]
rng = np.random.default_rng(0)
checked = 0
_cache = {}


def load_prices(stock):
    if stock not in _cache:
        _cache[stock] = pd.read_csv(
            F.STOCKNET / "price" / "raw" / f"{stock}.csv",
            parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    return _cache[stock]


for i in rng.choice(len(S), 3000, replace=False):
    s = S.iloc[i]
    prices = load_prices(s.stock)
    j = prices.index[prices["Date"] == s.date][0]
    prev = prices.at[j - 1, "Date"]
    # 1. label is the close(d-1)->close(d) move
    r = prices.at[j, "Adj Close"] / prices.at[j - 1, "Adj Close"] - 1
    assert abs(r - s.ret) < 1e-9 and s.label == int(r >= 0.0055)
    # 2. strict tweets were all posted before 20:00 UTC on d-1
    if len(s.tids_strict):
        assert created[s.tids_strict].max() < prev + pd.Timedelta(hours=20)
    # 3. calendar tweets were all posted before the UTC end of d-1
    if len(s.tids_calendar):
        assert created[s.tids_calendar].max() < prev + pd.Timedelta(days=1)
    # 4. tweets belong to the right stock
    assert (stock_of[s.tids_strict] == s.stock).all()
    # 5. per-day groups partition the strict window
    assert sorted(sum(map(list, s.tids_by_day), [])) == sorted(s.tids_strict)
    # 6. past returns end at d-1, not d
    assert abs(s.past_rets[-1] - prices["Adj Close"].pct_change().iloc[j - 1]) < 1e-6
    checked += 1

# 7. splits are chronological and disjoint
d = S.groupby("split")["date"].agg(["min", "max"])
assert d.loc["train", "max"] < d.loc["val", "min"] <= d.loc["val", "max"] < d.loc["test", "min"]
print(f"all leakage checks passed on {checked} random samples")
print(d)
