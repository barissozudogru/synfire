"""Profile vectorized vs loop-based implementations of hot paths.

Compares current (vectorized) implementations against reference
loop-based versions to measure speedup factors.

Usage:
    poetry run python -m benchmarks.profile_vectorization
"""

from __future__ import annotations

import time

import numpy as np

# ---- Reference (old loop-based) implementations ----

def _old_make_random_pairs(windows, rng, min_gap=5):
    """Original per-sample loop version."""
    n = len(windows)
    left_indices = np.arange(n)
    right_indices = np.empty(n, dtype=np.intp)
    for i in range(n):
        candidates = np.where(np.abs(np.arange(n) - i) >= min_gap)[0]
        if len(candidates) == 0:
            candidates = np.array([j for j in range(n) if j != i])
        right_indices[i] = rng.choice(candidates)
    return windows[left_indices], windows[right_indices]


def _old_transition_surprise(labels, n_prototypes, eps=1e-12):
    """Original double-loop version."""
    trans = np.zeros((n_prototypes, n_prototypes))
    for i in range(len(labels) - 1):
        trans[labels[i], labels[i + 1]] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    trans_prob = trans / (row_sums + eps)
    surprise = np.zeros(len(labels))
    for i in range(1, len(labels)):
        p = trans_prob[labels[i - 1], labels[i]]
        surprise[i] = -np.log(p + eps)
    return surprise


def _old_hebbian_update(state, x):
    """Original double-nested loop version."""
    from synfire.layers.hebbian import HebbianState, assign
    winners = assign(state, x)
    new_prototypes = state.prototypes.copy()
    lr = state.config.lr
    inhibition = state.config.inhibition_strength
    for i in range(len(x)):
        w = winners[i]
        diff = x[i] - new_prototypes[w]
        new_prototypes[w] += lr * diff
        for j in range(state.config.n_prototypes):
            if j != w:
                repel = new_prototypes[j] - x[i]
                new_prototypes[j] += (
                    inhibition * lr * repel / (np.linalg.norm(repel) + 1e-12)
                )
    return HebbianState(prototypes=new_prototypes, config=state.config)


# ---- Benchmark runner ----

def timeit(fn, *args, n_runs=5, **kwargs):
    """Time a function, return (mean_seconds, std_seconds)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times)


def profile_random_pairs():
    """Compare make_random_pairs: old loop vs vectorized."""
    from synfire.preprocessing.windows import make_random_pairs

    rng = np.random.default_rng(42)
    windows = rng.standard_normal((1000, 20))

    rng1 = np.random.default_rng(0)
    old_time, old_std = timeit(_old_make_random_pairs, windows, rng1, 5, n_runs=5)

    rng2 = np.random.default_rng(0)
    new_time, new_std = timeit(make_random_pairs, windows, rng2, 5, n_runs=5)

    return "make_random_pairs", old_time, new_time


def profile_transition_surprise():
    """Compare _transition_surprise: old loops vs vectorized."""
    from synfire.pipeline.anomaly import _transition_surprise

    rng = np.random.default_rng(42)
    labels = rng.integers(0, 8, size=2000)

    old_time, _ = timeit(_old_transition_surprise, labels, 8, n_runs=10)
    new_time, _ = timeit(_transition_surprise, labels, 8, n_runs=10)

    return "_transition_surprise", old_time, new_time


def profile_hebbian_update():
    """Compare update_step: old double-loop vs vectorized."""
    from synfire.core.config import HebbianConfig
    from synfire.layers.hebbian import HebbianState, update_step

    rng = np.random.default_rng(42)
    prototypes = rng.standard_normal((8, 32))
    config = HebbianConfig(n_prototypes=8, lr=0.01, inhibition_strength=0.1)
    state = HebbianState(prototypes=prototypes, config=config)
    x = rng.standard_normal((64, 32))

    old_time, _ = timeit(_old_hebbian_update, state, x, n_runs=10)
    new_time, _ = timeit(update_step, state, x, n_runs=10)

    return "hebbian_update_step", old_time, new_time


def profile_anomaly_scoring():
    """Profile full anomaly scoring pipeline."""
    from synfire import SynfireConfig, SynfirePipeline
    from synfire.core.config import FFStackConfig, HebbianConfig, WindowConfig

    config = SynfireConfig(
        window=WindowConfig(window_size=20, stride=1),
        ff_stack=FFStackConfig(layer_dims=(64, 32), lr=0.01, epochs_per_layer=50),
        hebbian=HebbianConfig(n_prototypes=8, lr=0.01, epochs=20),
    )

    t = np.arange(2000, dtype=np.float64)
    series = np.sin(2 * np.pi * t / 50)

    pipeline = SynfirePipeline(config)
    pipeline.fit(series[:1000])

    fit_time, _ = timeit(lambda: SynfirePipeline(config).fit(series[:1000]), n_runs=3)
    score_time, _ = timeit(pipeline.anomaly_scores, series, n_runs=5)

    return fit_time, score_time


def run_profiling():
    """Run all profiling benchmarks."""
    print("=" * 65)
    print("Vectorization Profiling Results")
    print("=" * 65)
    print(f"{'Operation':<25} | {'Old (s)':>10} | {'New (s)':>10} | {'Speedup':>8}")
    print("-" * 65)

    for profile_fn in [profile_random_pairs, profile_transition_surprise, profile_hebbian_update]:
        name, old_t, new_t = profile_fn()
        speedup = old_t / new_t if new_t > 0 else float("inf")
        print(f"{name:<25} | {old_t:>10.5f} | {new_t:>10.5f} | {speedup:>7.1f}x")

    print("-" * 65)

    fit_time, score_time = profile_anomaly_scoring()
    print("\nPipeline timing (2000-point sine):")
    print(f"  fit():             {fit_time:.3f}s")
    print(f"  anomaly_scores():  {score_time:.4f}s")


if __name__ == "__main__":
    run_profiling()
