"""Model persistence: save and load fitted SynfirePipeline to/from .npz files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from synfire.api import SynfirePipeline

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
            "batch_size": config.ff_stack.batch_size,
            "early_stopping_patience": config.ff_stack.early_stopping_patience,
            "early_stopping_min_delta": config.ff_stack.early_stopping_min_delta,
            "lr_schedule": config.ff_stack.lr_schedule,
            "lr_warmup_fraction": config.ff_stack.lr_warmup_fraction,
            "grad_clip_norm": config.ff_stack.grad_clip_norm,
            "optimizer": config.ff_stack.optimizer,
            "weight_decay": config.ff_stack.weight_decay,
            "layer_norm": config.ff_stack.layer_norm,
            "negative_strategy": config.ff_stack.negative_strategy,
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
            "ensemble_goodness": config.anomaly.ensemble_goodness,
        },
        "adaptive_threshold": config.adaptive_threshold,
    }


def _config_from_dict(d: dict) -> SynfireConfig:
    """Deserialize SynfireConfig from a dict."""
    ff_stack_d = d["ff_stack"]
    anomaly_d = d["anomaly"]
    return SynfireConfig(
        window=WindowConfig(**d["window"]),
        norm=NormConfig(**d["norm"]),
        ff_stack=FFStackConfig(
            layer_dims=tuple(ff_stack_d["layer_dims"]),
            lr=ff_stack_d["lr"],
            threshold=ff_stack_d["threshold"],
            epochs_per_layer=ff_stack_d["epochs_per_layer"],
            seed=ff_stack_d["seed"],
            batch_size=ff_stack_d.get("batch_size", 256),
            early_stopping_patience=ff_stack_d.get("early_stopping_patience", 0),
            early_stopping_min_delta=ff_stack_d.get("early_stopping_min_delta", 1e-4),
            lr_schedule=ff_stack_d.get("lr_schedule", "none"),
            lr_warmup_fraction=ff_stack_d.get("lr_warmup_fraction", 0.1),
            grad_clip_norm=ff_stack_d.get("grad_clip_norm", 0.0),
            optimizer=ff_stack_d.get("optimizer", "sgd"),
            weight_decay=ff_stack_d.get("weight_decay", 0.0),
            layer_norm=ff_stack_d.get("layer_norm", False),
            negative_strategy=ff_stack_d.get("negative_strategy", "random"),
        ),
        hebbian=HebbianConfig(**d["hebbian"]),
        anomaly=AnomalyConfig(
            weight_goodness=anomaly_d["weight_goodness"],
            weight_distance=anomaly_d["weight_distance"],
            weight_transition=anomaly_d["weight_transition"],
            use_goodness=anomaly_d["use_goodness"],
            use_distance=anomaly_d["use_distance"],
            use_transition=anomaly_d["use_transition"],
            ensemble_goodness=anomaly_d.get("ensemble_goodness", False),
        ),
        adaptive_threshold=d.get("adaptive_threshold", False),
    )


def save_pipeline(pipeline: SynfirePipeline, path: str | Path) -> None:
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
    if pipeline._stack is None or pipeline._hebbian is None or pipeline._anomaly_scaler is None:
        raise RuntimeError("Pipeline state is corrupted: missing stack, hebbian, or scaler.")

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
        # Layer norm parameters (only present when layer_norm=True)
        if layer.ln_gain is not None:
            arrays[f"ln_gain_{i}"] = layer.ln_gain
        if layer.ln_bias is not None:
            arrays[f"ln_bias_{i}"] = layer.ln_bias
        layer_configs.append({
            "input_dim": layer.config.input_dim,
            "hidden_dim": layer.config.hidden_dim,
            "lr": layer.config.lr,
            "threshold": layer.config.threshold,
            "epochs": layer.config.epochs,
            "seed": layer.config.seed,
            "batch_size": layer.config.batch_size,
            "early_stopping_patience": layer.config.early_stopping_patience,
            "early_stopping_min_delta": layer.config.early_stopping_min_delta,
            "lr_schedule": layer.config.lr_schedule,
            "lr_warmup_fraction": layer.config.lr_warmup_fraction,
            "grad_clip_norm": layer.config.grad_clip_norm,
            "optimizer": layer.config.optimizer,
            "weight_decay": layer.config.weight_decay,
            "layer_norm": layer.config.layer_norm,
            "negative_strategy": layer.config.negative_strategy,
        })
    lc_json = json.dumps(layer_configs)
    arrays["layer_configs_json"] = np.frombuffer(lc_json.encode("utf-8"), dtype=np.uint8)
    arrays["n_layers"] = np.array([len(stack.layers)])

    # Effective threshold (may differ from config when adaptive_threshold=True)
    arrays["effective_threshold"] = np.array([pipeline._effective_threshold])

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


def load_pipeline(path: str | Path) -> SynfirePipeline:
    """Deserialize a SynfirePipeline from an .npz file.

    Args:
        path: Path to the .npz file.

    Returns:
        A fitted SynfirePipeline instance.
    """
    from synfire.api import SynfirePipeline
    from synfire.core.config import FFLayerConfig

    with np.load(path, allow_pickle=False) as data:
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
            # Use .get() for fields added after initial release for backward compat
            layer_cfg = FFLayerConfig(
                input_dim=lc["input_dim"],
                hidden_dim=lc["hidden_dim"],
                lr=lc["lr"],
                threshold=lc["threshold"],
                epochs=lc["epochs"],
                seed=lc["seed"],
                batch_size=lc.get("batch_size", 256),
                early_stopping_patience=lc.get("early_stopping_patience", 0),
                early_stopping_min_delta=lc.get("early_stopping_min_delta", 1e-4),
                lr_schedule=lc.get("lr_schedule", "none"),
                lr_warmup_fraction=lc.get("lr_warmup_fraction", 0.1),
                grad_clip_norm=lc.get("grad_clip_norm", 0.0),
                optimizer=lc.get("optimizer", "sgd"),
                weight_decay=lc.get("weight_decay", 0.0),
                layer_norm=lc.get("layer_norm", False),
                negative_strategy=lc.get("negative_strategy", "random"),
            )
            ln_gain = data.get(f"ln_gain_{i}", None)
            ln_bias = data.get(f"ln_bias_{i}", None)
            layers.append(FFLayerState(
                W=data[f"W_{i}"],
                b=data[f"b_{i}"],
                config=layer_cfg,
                ln_gain=ln_gain,
                ln_bias=ln_bias,
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

        # Effective threshold (fall back to config threshold for files saved before this field)
        if "effective_threshold" in data:
            effective_threshold = float(data["effective_threshold"][0])
        else:
            effective_threshold = config.ff_stack.threshold

    # Reconstruct pipeline
    pipeline = SynfirePipeline(config)
    pipeline._stack = stack
    pipeline._hebbian = hebbian
    pipeline._anomaly_scaler = scaler
    pipeline._effective_threshold = effective_threshold
    pipeline._fitted = True

    return pipeline
