"""Safe creation rules for weights-only training experiments."""
from __future__ import annotations

from pathlib import Path


def new_experiment_root(training_config: dict, drive_dir: str) -> Path:
    configured = training_config.get("experiment_dir")
    if configured:
        return Path(configured)
    return Path(drive_dir) / "experiments" / training_config.get(
        "experiment_name", "neural_alignment_v1"
    )


def prepare_new_adapter_dir(experiment_root: Path, style: str) -> Path:
    """Create an empty adapter output directory; never imply weights-only resume."""
    path = experiment_root / f"adapter_{style}"
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite an existing experiment at {path}; "
            "choose a new training.experiment_name. Incomplete optimizer "
            "state is not supported as an exact resume."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path
