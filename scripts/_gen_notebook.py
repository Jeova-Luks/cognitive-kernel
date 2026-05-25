"""One-time generator: emits scripts/train_colab.ipynb from cell definitions below.

Re-run this script whenever you want to regenerate the notebook from source.
The reason we generate from a .py script: .ipynb JSON files are painful to
review in git diffs; the cell definitions below are the source of truth."""
import json
from pathlib import Path

CELLS = [
    ("md", [
        "# Cognitive Kernel v0.1 — Training Notebook\n",
        "\n",
        "Run this in Google Colab. Resumes automatically if a checkpoint exists in `OUTPUT_DIR`.\n"
    ]),
    ("code", [
        "# Cell 1: mount Drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n"
    ]),
    ("code", [
        "# Cell 2: clone repo (or pull if already cloned)\n",
        "import os, subprocess\n",
        "REPO_DIR = '/content/cognitive-kernel'\n",
        "if not os.path.exists(REPO_DIR):\n",
        "    subprocess.run(['git', 'clone', 'https://github.com/Jeova-Luks/cognitive-kernel.git', REPO_DIR], check=True)\n",
        "else:\n",
        "    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)\n",
        "os.chdir(REPO_DIR)\n"
    ]),
    ("code", [
        "# Cell 3: install dependencies\n",
        "!pip install -q -r requirements.txt\n"
    ]),
    ("code", [
        "# Cell 4: wandb login (skip with empty key for offline mode)\n",
        "import wandb\n",
        "# wandb.login(key='YOUR_KEY')  # uncomment and paste, or run `wandb login` interactively\n"
    ]),
    ("code", [
        "# Cell 5: pick config and output dir\n",
        "from pathlib import Path\n",
        "from config import load_config\n",
        "from trainer import Trainer\n",
        "\n",
        "CONFIG = 'configs/base_100m.yaml'\n",
        "OUTPUT_DIR = Path('/content/drive/MyDrive/cognitive-kernel/checkpoints/base_100m')\n",
        "\n",
        "cfg = load_config(Path(CONFIG))\n",
        "print(cfg)\n"
    ]),
    ("code", [
        "# Cell 6: train (resumes automatically if checkpoint exists in OUTPUT_DIR)\n",
        "trainer = Trainer(cfg, output_dir=OUTPUT_DIR)\n",
        "print(f'Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}')\n",
        "trainer.fit()\n"
    ]),
]


def make_cell(kind, source):
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


nb = {
    "cells": [make_cell(k, s) for k, s in CELLS],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "train_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out}")
