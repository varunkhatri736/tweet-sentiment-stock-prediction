"""Build the leakage-safe StockNet movement-prediction dataset.

Inputs : StockNet raw tweets (tweet/raw/<TICKER>/<date>, one JSON per line)
         StockNet raw prices (price/raw/<TICKER>.csv, Yahoo daily OHLCV)
Outputs: data/tweets.parquet   one row per (stock, tweet) after cleaning/dedup
         data/samples.parquet  one row per (stock, target trading day)

Label (Xu & Cohen, 2018): r_d = AdjClose_d / AdjClose_{d-1} - 1
    up   if r_d >=  0.55%
    down if r_d <= -0.50%
    otherwise the sample is dropped (too small to call).

Tweet cutoff — the key leakage control. For target day d we only use tweets
posted at or before the *close of the previous trading day* d-1, i.e. information
a trader holding over close(d-1) -> close(d) could actually have had.
  * "strict"   : tweets before 20:00 UTC on d-1 (16:00 EDT / 15:00 EST, always
                 at or before the 4pm ET close).
  * "calendar" : all tweets dated <= d-1 in UTC (StockNet-style alignment). This
                 admits tweets from the close until midnight UTC (7-8 pm ET),
                 e.g. reactions to after-hours earnings, which a trade placed
                 at close(d-1) could not have used.
The lag window is the 5 trading days ending at d-1 (StockNet's default).
Splits are chronological and follow StockNet exactly.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "stocknet-dataset")
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

START, END = pd.Timestamp("2014-01-01"), pd.Timestamp("2016-01-01")
SPLITS = [("train", "2014-01-01", "2015-08-01"),
          ("val", "2015-08-01", "2015-10-01"),
          ("test", "2015-10-01", "2016-01-01")]
UP, DOWN = 0.0055, -0.005
LAG = 5
STRICT_CUTOFF_HOUR_UTC = 20

URL_RE = re.compile(r"https?://\S+")
USER_RE = re.compile(r"@\w+")


def clean(text: str) -> str:
    text = text.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    text = URL_RE.sub("http", text)
    text = USER_RE.sub("@user", text)
    return " ".join(text.split())


# ---------------------------------------------------------------- tweets
rows = []
for tdir in sorted((ROOT / "tweet" / "raw").iterdir()):
    for f in sorted(tdir.iterdir()):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            rows.append((tdir.name, t["id_str"], t["created_at"], t["text"],
                         t["user"]["id_str"]))
tw = pd.DataFrame(rows, columns=["stock", "tweet_id", "created_at", "raw_text", "user_id"])
n_raw = len(tw)
tw["created_at"] = pd.to_datetime(tw["created_at"], format="%a %b %d %H:%M:%S %z %Y",
                                  utc=True).dt.tz_localize(None)
tw = tw.drop_duplicates(["stock", "tweet_id"])
n_unique = len(tw)
tw["text"] = tw["raw_text"].map(clean)
tw["date"] = tw["created_at"].dt.normalize()
# Spam/bot control: identical cleaned text for the same stock on the same day
# is kept once (StockNet contains many copy-paste bot posts).
tw = tw.drop_duplicates(["stock", "date", "text"])
tw = tw[(tw["created_at"] >= START - pd.Timedelta(days=14)) & (tw["created_at"] < END)]
tw = tw.sort_values(["stock", "created_at", "tweet_id"], kind="mergesort").reset_index(drop=True)
tw["tid"] = np.arange(len(tw))
tw[["tid", "stock", "tweet_id", "user_id", "created_at", "date", "text"]].to_parquet(
    OUT / "tweets.parquet")
print(f"tweets: raw lines {n_raw:,} -> unique (stock,id) {n_unique:,} -> "
      f"after same-day duplicate-text removal {len(tw):,}")

# ---------------------------------------------------------------- prices
samples = []
for stock in sorted(tw["stock"].unique()):
    p = pd.read_csv(ROOT / "price" / "raw" / f"{stock}.csv", parse_dates=["Date"])
    p = p.sort_values("Date").dropna(subset=["Adj Close"]).reset_index(drop=True)
    p["ret"] = p["Adj Close"].pct_change()
    p["logvol"] = np.log1p(p["Volume"])
    # intraday range normalised by previous close (a volatility proxy)
    p["range"] = (p["High"] - p["Low"]) / p["Close"].shift(1)
    st = tw[tw["stock"] == stock]
    t_times = st["created_at"].values
    t_ids = st["tid"].values
    for i in range(LAG + 21, len(p)):
        d = p.at[i, "Date"]
        if not (START <= d < END):
            continue
        r = p.at[i, "ret"]
        if np.isnan(r) or (DOWN < r < UP):
            continue
        prev = p.at[i - 1, "Date"]               # d-1 (last trading day)
        first = p.at[i - LAG, "Date"]            # first day of the lag window
        # window start: the close-time cutoff of the trading day *before* the
        # window, so consecutive windows tile without gaps or overlap.
        before_first = p.at[i - LAG - 1, "Date"]
        lo_strict = before_first + pd.Timedelta(hours=STRICT_CUTOFF_HOUR_UTC)
        hi_strict = prev + pd.Timedelta(hours=STRICT_CUTOFF_HOUR_UTC)
        lo_cal = first
        hi_cal = prev + pd.Timedelta(days=1)
        def ids_between(lo, hi):
            a = np.searchsorted(t_times, np.datetime64(lo), "left")
            b = np.searchsorted(t_times, np.datetime64(hi), "left")
            return t_ids[a:b].tolist()
        # per-day tweet groups inside the strict window (for sequence models)
        day_groups = []
        for k in range(LAG, 0, -1):
            lo_k = p.at[i - k - 1, "Date"] + pd.Timedelta(hours=STRICT_CUTOFF_HOUR_UTC)
            hi_k = p.at[i - k, "Date"] + pd.Timedelta(hours=STRICT_CUTOFF_HOUR_UTC)
            day_groups.append(ids_between(lo_k, hi_k))
        rets = p.loc[i - 20:i - 1, "ret"].values  # 20 past returns, ends at d-1
        samples.append(dict(
            stock=stock, date=d, ret=r, label=int(r >= UP),
            tids_strict=ids_between(lo_strict, hi_strict),
            tids_calendar=ids_between(lo_cal, hi_cal),
            tids_by_day=day_groups,
            tids_strict_last=day_groups[-1],
            tids_calendar_last=ids_between(prev, hi_cal),
            past_rets=rets.astype(np.float32),
            past_range=p.loc[i - LAG:i - 1, "range"].values.astype(np.float32),
            past_logvol=p.loc[i - 20:i - 1, "logvol"].values.astype(np.float32),
        ))

S = pd.DataFrame(samples)
S["split"] = ""
for name, a, b in SPLITS:
    S.loc[(S["date"] >= a) & (S["date"] < b), "split"] = name
S["n_strict"] = S["tids_strict"].map(len)
S["n_calendar"] = S["tids_calendar"].map(len)
# StockNet keeps only samples with tweets in the lag window; we require at least
# one tweet under the *strict* cutoff so both settings use identical samples.
S = S[S["n_strict"] > 0].reset_index(drop=True)
S.to_parquet(OUT / "samples.parquet")

summ = S.groupby("split").agg(n=("label", "size"), up_rate=("label", "mean"),
                              tweets_strict=("n_strict", "mean"),
                              tweets_calendar=("n_calendar", "mean"),
                              stocks=("stock", "nunique"))
print(summ.loc[["train", "val", "test"]].round(3).to_string())
summ.to_csv(OUT / "split_summary.csv")
