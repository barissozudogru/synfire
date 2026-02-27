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

## Development

```bash
poetry run pytest -v
poetry run ruff check synfire/ tests/
```
