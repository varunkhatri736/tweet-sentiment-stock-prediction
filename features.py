"""Turn per-tweet features into per-sample feature matrices.

Every function takes the samples frame S and a `setting` ("strict" or
"calendar") naming which tweet window to aggregate over.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
STOCKNET = Path(os.environ.get("STOCKNET_DIR", Path(__file__).parent / "stocknet-dataset"))


def load(with_lexicon=True):
    S = pd.read_parquet(DATA / "samples.parquet")
    tw = pd.read_parquet(DATA / "tweets.parquet")
    if not with_lexicon:  # lexicon.parquet is only written by embed_tweets.py
        return S, tw, None
    lex = pd.read_parquet(DATA / "lexicon.parquet")
    assert len(lex) == len(tw)
    return S, tw, lex


def price_features(S):
    """Only uses prices up to and including d-1."""
    R = np.stack(S["past_rets"].values)          # 20 returns, last = d-1
    V = np.stack(S["past_logvol"].values)
    G = np.stack(S["past_range"].values)          # 5 ranges
    feats = {f"ret_lag{k}": R[:, -k] for k in range(1, 6)}
    feats.update(
        mom5=R[:, -5:].sum(1), mom20=R.sum(1),
        vol20=R.std(1), vol5=R[:, -5:].std(1),
        range_mean5=G.mean(1), range_last=G[:, -1],
        volu_surprise=V[:, -1] - V[:, :-1].mean(1),
    )
    return pd.DataFrame(feats).fillna(0.0)


LEX_COLS = ["vader_compound", "vader_pos", "vader_neg", "lm_pos", "lm_neg", "lm_polarity"]


def lexicon_features(S, lex, setting="strict"):
    L = lex[LEX_COLS].values
    pos = (lex["vader_compound"].values > 0.05).astype(float)
    neg = (lex["vader_compound"].values < -0.05).astype(float)
    rows = []
    for tids, last in zip(S[f"tids_{setting}"], S[f"tids_{setting}_last"]):
        tids = np.asarray(tids, dtype=int)
        last = np.asarray(last, dtype=int)
        if len(tids) == 0:  # possible only under the calendar cutoff
            rows.append([0.0] * (2 * len(LEX_COLS) + 6))
            continue
        f = list(L[tids].mean(0)) + [pos[tids].mean(), neg[tids].mean(),
                                     pos[tids].mean() - neg[tids].mean(),
                                     np.log1p(len(tids))]
        if len(last):
            f += list(L[last].mean(0)) + [pos[last].mean() - neg[last].mean(),
                                          np.log1p(len(last))]
        else:
            f += [0.0] * (len(LEX_COLS) + 2)
        rows.append(f)
    cols = ([f"win_{c}" for c in LEX_COLS] + ["win_frac_pos", "win_frac_neg", "win_net",
                                               "win_logn"]
            + [f"last_{c}" for c in LEX_COLS] + ["last_net", "last_logn"])
    return pd.DataFrame(rows, columns=cols)


def window_texts(S, tw, setting="strict"):
    text = tw["text"].values
    return [" ".join(text[np.asarray(t, dtype=int)]) for t in S[f"tids_{setting}"]]


def roberta_features(S, emb, setting="strict"):
    """Window mean and most-recent-day mean of frozen RoBERTa tweet embeddings."""
    E = emb.astype(np.float32)
    win = np.stack([E[np.asarray(t, dtype=int)].mean(0) if len(t) else
                    np.zeros(E.shape[1], np.float32) for t in S[f"tids_{setting}"]])
    last = np.stack([E[np.asarray(d, dtype=int)].mean(0) if len(d) else
                     np.zeros(E.shape[1], np.float32) for d in S[f"tids_{setting}_last"]])
    return np.hstack([win, last])


def sequence_inputs(S, emb, lex):
    """[n, 5 days, feat] tensors for the LSTM (strict per-day tweet groups)."""
    E = emb.astype(np.float32)
    L = lex[LEX_COLS].values.astype(np.float32)
    R = np.stack(S["past_rets"].values)[:, -5:]
    G = np.stack(S["past_range"].values)
    V = np.stack(S["past_logvol"].values)
    Vs = V[:, -5:] - V[:, :-5].mean(1, keepdims=True)
    n = len(S)
    text = np.zeros((n, 5, E.shape[1]), np.float32)
    lexd = np.zeros((n, 5, L.shape[1] + 2), np.float32)
    for i, days in enumerate(S["tids_by_day"]):
        for j, ids in enumerate(days):
            ids = np.asarray(ids, dtype=int)
            if len(ids):
                text[i, j] = E[ids].mean(0)
                lexd[i, j, :-2] = L[ids].mean(0)
                lexd[i, j, -2] = np.log1p(len(ids))
                lexd[i, j, -1] = 1.0  # has-tweets flag
    price = np.stack([R, G, Vs], -1).astype(np.float32)
    price = np.nan_to_num(price)
    return text, lexd, price
