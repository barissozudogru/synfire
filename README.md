# synfire

Forward-Forward + Hebbian competitive learning for time series anomaly detection, clustering, and representation learning.

## Status and requirements

A research implementation of Forward-Forward and Hebbian competitive learning.
Use the tests and benchmark code to evaluate it for your own time series; this
repository does not establish production suitability for a specific dataset.

Requires Python 3.12 and Poetry.

## Install from source

```bash
git clone https://github.com/barissozudogru/synfire.git
cd synfire
poetry install
```

Install this project from its source repository. The
[package named `synfire` on PyPI](https://pypi.org/project/synfire/) is a separate
model registry SDK.

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
poetry run ruff check synfire/ tests/ benchmarks/
poetry run pyright synfire/
```

## Support and license

Report reproducible problems through [GitHub issues](https://github.com/barissozudogru/synfire/issues).
Include your Python version, input shapes, pipeline settings, and a traceback.
Avoid attaching private datasets.

Licensed under [MIT](./LICENSE).
