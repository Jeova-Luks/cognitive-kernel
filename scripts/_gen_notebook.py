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
    ("md", [
        "## Loss curves\n",
        "\n",
        "Reads `metrics.jsonl` from OUTPUT_DIR (written by the trainer) and plots\n",
        "train_loss + val_loss vs step. Run this cell after Cell 6 finishes.\n"
    ]),
    ("code", [
        "# Cell 7: plot loss curves from metrics.jsonl\n",
        "import json\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "metrics_path = OUTPUT_DIR / 'metrics.jsonl'\n",
        "if not metrics_path.exists():\n",
        "    print(f'No metrics.jsonl found at {metrics_path}; run Cell 6 first.')\n",
        "else:\n",
        "    records = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]\n",
        "    train_pts = [(r['step'], r['train_loss']) for r in records if 'train_loss' in r]\n",
        "    val_pts   = [(r['step'], r['val_loss'])   for r in records if 'val_loss'   in r]\n",
        "    if not train_pts:\n",
        "        print('No train_loss records yet.')\n",
        "    else:\n",
        "        fig, ax = plt.subplots(figsize=(11, 5))\n",
        "        ts, ls = zip(*train_pts)\n",
        "        ax.plot(ts, ls, label='train_loss', color='tab:blue', alpha=0.7, linewidth=1)\n",
        "        if val_pts:\n",
        "            vs, vl = zip(*val_pts)\n",
        "            ax.plot(vs, vl, label='val_loss', color='tab:red', marker='o', linestyle='--')\n",
        "        ax.set_xlabel('step')\n",
        "        ax.set_ylabel('loss')\n",
        "        ax.set_title(f'Training curve ({len(records)} log points, final step {trainer.step})')\n",
        "        ax.legend()\n",
        "        ax.grid(alpha=0.3)\n",
        "        plt.show()\n",
        "        print(f'\\\\n  First train_loss: {ls[0]:.4f}')\n",
        "        print(f'  Last  train_loss: {ls[-1]:.4f}')\n",
        "        print(f'  Delta: {ls[-1] - ls[0]:+.4f}  ({\"falling \\u2713\" if ls[-1] < ls[0] else \"rising \\u2717\"})')\n"
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
