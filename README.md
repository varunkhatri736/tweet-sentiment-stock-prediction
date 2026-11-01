# Do Tweets Predict Next-Day Stock Moves?

Does the sentiment of tweets about a stock help predict whether that stock closes
up or down the next trading day? This project tests that on the public
[StockNet benchmark](https://github.com/yumoxu/stocknet-dataset). It compares
sentiment lexicons, TF-IDF, a RoBERTa encoder and LSTMs against simple baselines,
and it is careful never to let a model see information from after the moment it
is predicting.

**Short answer: no, not measurably.** The best tweet-based model was right 52.5%
of the time. Always guessing "up" was right 52.6% of the time.

## Project development

I started this project in summer 2025 to explore financial tweet sentiment
analysis with logistic regression, LSTMs and BERT. I later replaced BERT with RoBERTa to test whether it could improve prediction accuracy.
The current version expands
that question to next-day stock movement prediction, comparing tweet-based
models with price-only and simple prediction baselines.

The evaluation focuses on leakage-controlled tweet windows, chronological data
splits and day-level confidence intervals. These checks help assess whether tweet
sentiment adds predictive value beyond historical prices and baseline guesses.

## The question

For each stock *s* and trading day *d*, predict whether the close-to-close return
from day *d*−1 to day *d* is:

- **up**: +0.55% or more
- **down**: −0.50% or less

Smaller moves are dropped, as in the original StockNet paper. A model may use
only tweets posted **before 20:00 UTC on day *d*−1** (at or before the 4 pm New
York close) and prices up to day *d*−1.

## Data

[StockNet](https://github.com/yumoxu/stocknet-dataset) (Xu & Cohen, ACL 2018):
tweets and daily prices for 87 large US-listed stocks, 2014–2015. The dataset is
**not included** here. Clone it yourself (see "How to run").

After removing duplicate copy-paste posts, 92,289 tweets remain.

| Split | Dates | Samples | Share labelled up |
| --- | --- | --- | --- |
| Train | 2014-01-01 to 2015-07-31 | 15,453 | 51.3% |
| Validation | 2015-08-01 to 2015-09-30 | 2,033 | 45.7% |
| Test | 2015-10-01 to 2015-12-31 | 2,845 | 52.6% |

## Models

| Model | Input |
| --- | --- |
| Always predict up / random guess | nothing (baselines) |
| Price-only logistic regression | past returns, momentum, volatility, volume |
| Lexicon sentiment + logistic regression | VADER and Loughran–McDonald scores |
| TF-IDF + logistic regression | word and word-pair counts |
| Frozen RoBERTa + logistic regression | 768-dim tweet embeddings |
| LSTM | 5 days of prices, optionally plus RoBERTa and lexicon features |

## Results (test set, 2,845 predictions)

| Model | Accuracy | 95% interval |
| --- | --- | --- |
| Always predict up | 52.6% | 42.8–62.2% |
| Random guess | 48.6% | 47.0–50.3% |
| Price-only logistic regression | 51.1% | 45.3–56.9% |
| Lexicon sentiment | 52.3% | 46.7–57.7% |
| Lexicon sentiment + price | 52.0% | 47.2–56.9% |
| TF-IDF | 49.0% | 47.1–50.6% |
| Frozen RoBERTa | 50.5% | 48.2–52.7% |
| Frozen RoBERTa + price | 50.8% | 47.9–53.3% |
| LSTM: price | 51.6% | 47.0–56.3% |
| LSTM: price + lexicon | 49.2% | 44.8–53.7% |
| LSTM: price + RoBERTa | 51.8% | 48.0–55.8% |
| LSTM: price + RoBERTa + lexicon | 52.5% | 48.6–56.3% |

Intervals come from a bootstrap that resamples whole trading days, because all
stocks tend to move together on the same day.

**Why the tweets don't help:**

- 48% of tweets mention three or more tickers, and the ones inspected were mostly
  automated feeds.
- Even the **same day's** tweet sentiment barely tracks that day's return
  (Spearman ρ = 0.029). The previous day's is essentially zero (ρ = 0.005).
- As a deliberate "cheating" check, models given tweets from during the price move
  itself still only reached 49–54%.

Full numbers are in `results/L4/` (main run) and `results/L12/` (comparison run
using RoBERTa's last layer).

## How to run

Tested with Python 3.13 on CPU. The full run takes about 1 hour on 2 cores.

```bash
# 1. Install
pip install -r requirements.txt

# 2. Get the dataset (into this folder)
git clone https://github.com/yumoxu/stocknet-dataset

# 3. Get RoBERTa weights.
#    Option A (internet access to Hugging Face): use "roberta-base" as MODEL below.
#    Option B: download the spaCy en_core_web_trf-3.8.0 wheel from
#    https://github.com/explosion/spacy-models/releases/tag/en_core_web_trf-3.8.0
#    unzip it, then convert it:
python convert_roberta.py en_core_web_trf/en_core_web_trf-3.8.0/transformer/model models/roberta
#    and use models/roberta as MODEL below.

# 4. Build the dataset and check for leakage
python build_dataset.py
python test_pipeline.py

# 5. Tweet features, train, evaluate
python embed_tweets.py MODEL --layer 4
python layer_probe.py MODEL
python data_quality.py
EMB_LAYER=4 python models.py
EMB_LAYER=4 python analyze.py
EMB_LAYER=4 python positive_control.py
```

The published results used Option B. Results with vanilla `roberta-base` may
differ slightly.

## Files

| File | What it does |
| --- | --- |
| `build_dataset.py` | Cleans tweets, builds labels and leakage-safe tweet windows |
| `test_pipeline.py` | Automated checks that no future information leaks in |
| `convert_roberta.py` | Converts spaCy's RoBERTa weights to Hugging Face format |
| `embed_tweets.py` | RoBERTa embeddings and lexicon scores for every tweet |
| `layer_probe.py` | Picks the RoBERTa layer using sentiment only, never price labels |
| `features.py` | Turns tweet features into per-sample model inputs |
| `models.py` | Trains and evaluates all models |
| `analyze.py` | Confidence intervals and paired comparisons |
| `data_quality.py` | Measures how much signal the tweets carry |
| `positive_control.py` | The deliberate "cheating" check |

## References

- Xu, Y. and Cohen, S. B. (2018). [Stock Movement Prediction from Tweets and Historical Prices](https://aclanthology.org/P18-1183). ACL.
- Hutto, C. J. and Gilbert, E. (2014). [VADER](https://ojs.aaai.org/index.php/icwsm/article/view/14550). ICWSM.
- Loughran, T. and McDonald, B. (2011). [When Is a Liability Not a Liability?](https://papers.ssrn.com/abstract=1331573) *Journal of Finance*.
- Liu, Y. et al. (2019). [RoBERTa](https://arxiv.org/abs/1907.11692). arXiv.
