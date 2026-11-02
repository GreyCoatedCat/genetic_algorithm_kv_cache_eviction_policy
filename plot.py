"""Buat grafik dari history.csv dan results.json tanpa menjalankan model."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from search import pyramid_budgets, uniform_budgets

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
METHODS = {
    "evolkv_ga": ("EvolKV-GA", "#2a78d6"),
    "snapkv": ("SnapKV", "#eb6834"),
    "pyramidkv": ("PyramidKV", "#1baf7a"),
}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(path)


def plot_convergence(history, path):
    """Fitness tiap generasi; garis pemisah menandai pergantian grup layer."""
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    steps = range(len(history))
    ax.plot(steps, [float(row["mean_fitness"]) for row in history], color=MUTED, linewidth=1.5, label="Rata-rata populasi")
    ax.plot(steps, [float(row["best_fitness"]) for row in history], color=METHODS["evolkv_ga"][1], linewidth=2, label="Terbaik")

    groups = {}
    for step, row in enumerate(history):
        groups.setdefault((int(row["first_layer"]), int(row["last_layer"])), []).append(step)
    top = max(float(row["best_fitness"]) for row in history)
    for (first, last), members in groups.items():
        if members[0] > 0:
            ax.axvline(members[0] - 0.5, color=GRID, linewidth=1)
        center = (members[0] + members[-1]) / 2
        ax.text(center, top * 1.1, f"Layer {first}–{last}", ha="center", color=MUTED, fontsize=9)

    ax.set_ylim(0, top * 1.2 or 1)
    ax.set_xlabel("Generasi kumulatif")
    ax.set_ylabel("Fitness")
    ax.set_title("Konvergensi Genetic Algorithm per grup layer", loc="left")
    ax.legend(frameon=False, labelcolor=MUTED, loc="center right")
    style(ax)
    save(fig, path)


def plot_budgets(results, path):
    """Alokasi per layer pada target utama, seperti Gambar 2a di paper."""
    target = results["args"]["target"]
    evolved = results["targets"][str(target)]["budgets"]
    layers = len(evolved)
    allocations = {
        "evolkv_ga": evolved,
        "snapkv": uniform_budgets(layers, target),
        "pyramidkv": pyramid_budgets(layers, target, results["args"]["window"]),
    }

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    for method, budgets in allocations.items():
        label, color = METHODS[method]
        ax.plot(range(layers), budgets, color=color, linewidth=2, marker="o", markersize=4, label=label)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Budget KV cache (token)")
    ax.set_title(f"Alokasi budget per layer, target rata-rata c = {target}", loc="left")
    ax.legend(frameon=False, labelcolor=MUTED, loc="upper left")
    style(ax)
    save(fig, path)


def plot_recall(results, path):
    """Recall data uji per target budget, dengan Full KV sebagai batas acuan."""
    targets = list(results["targets"])
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    for offset, method in enumerate(["snapkv", "pyramidkv", "evolkv_ga"]):
        label, color = METHODS[method]
        values = [results["targets"][target][method] for target in targets]
        positions = [i + (offset - 1) * width for i in range(len(targets))]
        bars = ax.bar(positions, values, width * 0.9, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=2, fontsize=8, color=MUTED)

    full = results["full"]
    ax.axhline(full, color=TEXT, linewidth=1.5, linestyle="--")
    ax.text(len(targets) - 0.5, full + 0.015, f"Full KV {full:.2f}", ha="right", color=MUTED, fontsize=9)
    ax.set_xticks(range(len(targets)), [f"c = {target}" for target in targets])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Recall data uji per target budget", loc="left")
    ax.legend(frameon=False, labelcolor=MUTED, loc="upper left", ncols=3)
    style(ax)
    ax.grid(axis="x", visible=False)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default="history.csv")
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    with open(args.history, newline="") as file:
        history = list(csv.DictReader(file))
    with open(args.results) as file:
        results = json.load(file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    plot_convergence(history, output_dir / "convergence.png")
    plot_budgets(results, output_dir / "budgets.png")
    plot_recall(results, output_dir / "recall.png")


if __name__ == "__main__":
    main()
