"""Detect the latest checkpoint in an output directory."""
from pathlib import Path
from typing import Optional


def find_latest(output_dir: Path) -> Optional[Path]:
    """Return path to the latest checkpoint in output_dir, or None if none exists.

    Prefers the explicit `latest.txt` pointer; falls back to the highest-step
    ckpt_step_*.pt file if the pointer is missing or stale.
    """
    output_dir = Path(output_dir)
    pointer = output_dir / "latest.txt"
    if pointer.exists():
        name = pointer.read_text().strip()
        candidate = output_dir / name
        if candidate.exists():
            return candidate

    ckpts = sorted(output_dir.glob("ckpt_step_*.pt"))
    return ckpts[-1] if ckpts else None
