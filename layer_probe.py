"""Pick the RoBERTa layer without touching movement labels.

On 3,000 tweets with clear VADER polarity (|compound| > 0.3), fit a linear probe
from each layer's mean-pooled embedding to the polarity sign (3-fold CV). The
layer that carries the most sentiment information is used for the main models.
Writes results/layer_probe.csv.

Usage: python layer_probe.py <roberta_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

import features as F

_, tw, lex = F.load()
c = lex["vader_compound"].values
idx = np.random.default_rng(0).choice(np.where(np.abs(c) > 0.3)[0], 3000, replace=False)
y = (c[idx] > 0).astype(int)
texts = tw["text"].values[idx].tolist()
rows = [{"features": "majority class", "cv_accuracy": max(y.mean(), 1 - y.mean())},
        {"features": "TF-IDF", "cv_accuracy": cross_val_score(
            make_pipeline(TfidfVectorizer(min_df=2), LogisticRegression(C=10, max_iter=2000)),
            texts, y, cv=3).mean()}]

tok = AutoTokenizer.from_pretrained(sys.argv[1])
model = AutoModel.from_pretrained(sys.argv[1]).eval()
layers = [[] for _ in range(model.config.num_hidden_layers + 1)]
with torch.inference_mode():
    for k in range(0, len(texts), 64):
        b = tok(texts[k:k + 64], return_tensors="pt", padding=True, truncation=True,
                max_length=64)
        mask = b["attention_mask"].unsqueeze(-1).float()
        for l, h in enumerate(model(**b, output_hidden_states=True).hidden_states):
            layers[l].append(((h * mask).sum(1) / mask.sum(1)).numpy())
for l, chunks in enumerate(layers):
    acc = cross_val_score(make_pipeline(StandardScaler(),
                                        LogisticRegression(C=0.01, max_iter=3000)),
                          np.vstack(chunks), y, cv=3).mean()
    rows.append({"features": f"RoBERTa layer {l}", "cv_accuracy": acc})
out = pd.DataFrame(rows)
res = Path(__file__).parent / "results"
res.mkdir(exist_ok=True)
out.to_csv(res / "layer_probe.csv", index=False)
print(out.round(3).to_string(index=False))
