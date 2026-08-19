# synfire

Forward-Forward + Hebbian competitive learning for time series anomaly detection, clustering, and representation learning.

## Install

```bash
poetry install
```

## Usage

```python
from synfire import SynfirePipeline

pipeline = SynfirePipeline()
pipeline.fit(normal_time_series)

scores = pipeline.anomaly_scores(test_series)
clusters = pipeline.cluster(test_series)
representations = pipeline.transform(test_series)
```

## Score alignment

Scores are computed over sliding windows, so the output is shorter than the input
and offset from it. With the default `window_size=25, stride=1`, a 300-sample
series yields 275 scores.

A score index is therefore not a sample index. Map it back before reading the
series:

```python
scores = pipeline.anomaly_scores(series)   # 275 scores for 300 samples
worst = int(scores.argmax())

sample = pipeline.score_index_to_sample(worst)   # index into `series`
start, end = pipeline.score_window_bounds(worst) # window the score covers
```

## Development

```bash
poetry run pytest -v
poetry run ruff check synfire/ tests/
```
