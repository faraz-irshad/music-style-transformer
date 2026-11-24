"""Select new experiment adapters without hiding historical baseline adapters."""
from pathlib import Path


def resolve_adapter_path(drive_dir: str | Path, experiment_root: str | Path, style: str) -> Path:
    new_path = Path(experiment_root) / f"adapter_{style}" / "final"
    legacy_path = Path(drive_dir) / f"adapter_{style}" / "final"
    return new_path if new_path.exists() else legacy_path
