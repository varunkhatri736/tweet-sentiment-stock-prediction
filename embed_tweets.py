"""Per-tweet features, computed once and reused by every model.

  data/roberta_emb_L<N>.npy  float16 [n_tweets, 768]  mean-pooled layer N of a
                         frozen RoBERTa-base encoder (see convert_roberta.py)
  data/lexicon.parquet   VADER compound/pos/neg and Loughran-McDonald
                         positive/negative/polarity per tweet

Usage: python embed_tweets.py <roberta_dir> [--layer N] [--skip-lexicon] [--limit N]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

DATA = Path(__file__).parent / "data"
model_dir = sys.argv[1]
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
# --layer N keeps only the first N transformer layers (12 = full model). A probe
# of each layer against VADER polarity (see README) showed layer 4 carries the
# most sentiment information in this encoder, the last layer the least.
layer = int(sys.argv[sys.argv.index("--layer") + 1]) if "--layer" in sys.argv else 12
skip_lex = "--skip-lexicon" in sys.argv

tw = pd.read_parquet(DATA / "tweets.parquet")
texts = tw["text"].tolist()[:limit]

# ---- lexicon sentiment (fast) ----------------------------------------------
if limit is None and not skip_lex:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import pysentiment2 as ps
    vader, lm = SentimentIntensityAnalyzer(), ps.LM()
    lex = []
    for t in texts:
        v = vader.polarity_scores(t)
        s = lm.get_score(lm.tokenize(t))
        lex.append((v["compound"], v["pos"], v["neg"], s["Positive"], s["Negative"],
                    s["Polarity"]))
    pd.DataFrame(lex, columns=["vader_compound", "vader_pos", "vader_neg", "lm_pos",
                               "lm_neg", "lm_polarity"]).to_parquet(DATA / "lexicon.parquet")
    print("lexicon done")

# ---- RoBERTa embeddings ------------------------------------------------------
torch.set_num_threads(max(1, torch.get_num_threads()))
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir).eval()
model.encoder.layer = model.encoder.layer[:layer]
# sort by length so batches have little padding
order = np.argsort([len(t) for t in texts])
emb = np.zeros((len(texts), 768), dtype=np.float16)
bs, t0 = 64, time.time()
with torch.inference_mode():
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        b = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                truncation=True, max_length=64)
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        emb[idx] = ((h * m).sum(1) / m.sum(1)).numpy().astype(np.float16)
        if (k // bs) % 100 == 0:
            done = k + len(idx)
            print(f"{done:,}/{len(order):,}  {done / (time.time() - t0):.1f} tweets/s",
                  flush=True)
if limit is None:
    np.save(DATA / f"roberta_emb_L{layer}.npy", emb)
print(f"done in {time.time() - t0:.0f}s")
