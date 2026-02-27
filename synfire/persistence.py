"""Model persistence: save and load fitted SynfirePipeline to/from .npz files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from synfire.core.config import (
    AnomalyConfig,
    FFStackConfig,
    HebbianConfig,
    NormConfig,
    SynfireConfig,
    WindowConfig,
)
from synfire.layers.ff_layer import FFLayerState
from synfire.layers.ff_stack import FFStackState
from synfire.layers.hebbian import HebbianState
from synfire.pipeline.anomaly import AnomalyScaler


def _config_to_dict(config: SynfireConfig) -> dict:
    """Serialize SynfireConfig to a JSON-compatible dict."""
    return {
        "window": {"window_size": config.window.window_size, "stride": config.window.stride},
        "norm": {"method": config.norm.method, "eps": config.norm.eps},
        "ff_stack": {
            "layer_dims": list(config.ff_stack.layer_dims),
            "lr": config.ff_stack.lr,
            "threshold": config.ff_stack.threshold,
            "epochs_per_layer": config.ff_stack.epochs_per_layer,
            "seed": config.ff_stack.seed,
        },
        "hebbian": {
            "n_prototypes": config.hebbian.n_prototypes,
            "lr": config.hebbian.lr,
            "inhibition_strength": config.hebbian.inhibition_strength,
            "epochs": config.hebbian.epochs,
            "seed": config.hebbian.seed,
        },
        "anomaly": {
            "weight_goodness": config.anomaly.weight_goodness,
            "weight_distance": config.anomaly.weight_distance,
            "weight_transition": config.anomaly.weight_transition,
            "use_goodness": config.anomaly.use_goodness,
            "use_distance": config.anomaly.use_distance,
            "use_transition": config.anomaly.use_transition,
        },
    }


def _config_from_dict(d: dict) -> SynfireConfig:
    """Deserialize SynfireConfig from a dict."""
    return SynfireConfig(
        window=WindowConfig(**d["window"]),
        norm=NormConfig(**d["norm"]),
        ff_stack=FFStackConfig(
            layer_dims=tuple(d["ff_stack"]["layer_dims"]),
            lr=d["ff_stack"]["lr"],
            threshold=d["ff_stack"]["threshold"],
            epochs_per_layer=d["ff_stack"]["epochs_per_layer"],
            seed=d["ff_stack"]["seed"],
        ),
        hebbian=HebbianConfig(**d["hebbian"]),
        anomaly=AnomalyConfig(**d["anomaly"]),
    )


def save_pipeline(pipeline, path: str | Path) -> None:
    """Serialize a fitted SynfirePipeline to an .npz file.

    The file contains:
    - config: JSON-encoded SynfireConfig
    - layer weights and biases (W_0, b_0, W_1, b_1, ...)
    - layer configs as JSON (layer_configs)
    - hebbian prototypes
    - anomaly scaler scalars and transition matrix

    Args:
        pipeline: A fitted SynfirePipeline instance.
        path: Destination file path.

    Raises:
        RuntimeError: If the pipeline is not fitted.
    """
    from synfire.api import SynfirePipeline

    if not isinstance(pipeline, SynfirePipeline):
        raise TypeError(f"Expected SynfirePipeline, got {type(pipeline).__name__}")

    pipeline._check_fitted()
    assert pipeline._stack is not None
    assert pipeline._hebbian is not None
    assert pipeline._anomaly_scaler is not None

    arrays: dict[str, NDArray] = {}

    # Config as JSON bytes
    config_json = json.dumps(_config_to_dict(pipeline.config))
    arrays["config_json"] = np.frombuffer(config_json.encode("utf-8"), dtype=np.uint8)

    # FF stack layers
    stack = pipeline._stack
    layer_configs = []
    for i, layer in enumerate(stack.layers):
        arrays[f"W_{i}"] = layer.W
        arrays[f"b_{i}"] = layer.b
        layer_configs.append({
            "input_dim": layer.config.input_dim,
            "hidden_dim": layer.config.hidden_dim,
            "lr": layer.config.lr,
            "threshold": layer.config.threshold,
            "epochs": layer.config.epochs,
            "seed": layer.config.seed,
        })
    lc_json = json.dumps(layer_configs)
    arrays["layer_configs_json"] = np.frombuffer(lc_json.encode("utf-8"), dtype=np.uint8)
    arrays["n_layers"] = np.array([len(stack.layers)])

    # Hebbian state
    arrays["prototypes"] = pipeline._hebbian.prototypes

    # Anomaly scaler
    scaler = pipeline._anomaly_scaler
    arrays["scaler_scalars"] = np.array([
        scaler.goodness_min, scaler.goodness_range,
        scaler.distance_min, scaler.distance_range,
        scaler.surprise_min, scaler.surprise_range,
    ])
    arrays["scaler_trans_prob"] = scaler.trans_prob

    np.savez(str(path), **arrays)  # type: ignore[arg-type]


def load_pipeline(path: str | Path):
    """Deserialize a SynfirePipeline from an .npz file.

    Args:
        path: Path to the .npz file.

    Returns:
        A fitted SynfirePipeline instance.
    """
    from synfire.api import SynfirePipeline
    from synfire.core.config import FFLayerConfig

    data = np.load(path, allow_pickle=False)

    # Config
    config_json = data["config_json"].tobytes().decode("utf-8")
    config = _config_from_dict(json.loads(config_json))

    # FF stack layers
    n_layers = int(data["n_layers"][0])
    lc_json = data["layer_configs_json"].tobytes().decode("utf-8")
    layer_configs = json.loads(lc_json)

    layers = []
    for i in range(n_layers):
        lc = layer_configs[i]
        layer_cfg = FFLayerConfig(**lc)
        layers.append(FFLayerState(
            W=data[f"W_{i}"],
            b=data[f"b_{i}"],
            config=layer_cfg,
        ))

    stack = FFStackState(layers=layers, config=config.ff_stack)

    # Hebbian state
    hebbian = HebbianState(prototypes=data["prototypes"], config=config.hebbian)

    # Anomaly scaler
    s = data["scaler_scalars"]
    scaler = AnomalyScaler(
        goodness_min=float(s[0]),
        goodness_range=float(s[1]),
        distance_min=float(s[2]),
        distance_range=float(s[3]),
        surprise_min=float(s[4]),
        surprise_range=float(s[5]),
        trans_prob=data["scaler_trans_prob"],
    )

    # Reconstruct pipeline
    pipeline = SynfirePipeline(config)
    pipeline._stack = stack
    pipeline._hebbian = hebbian
    pipeline._anomaly_scaler = scaler
    pipeline._fitted = True

    return pipeline
