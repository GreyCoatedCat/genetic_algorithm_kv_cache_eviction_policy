"""Rangkum semua seed di folder hasil menjadi grafik, tanpa menjalankan model."""

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


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def summarize(columns):
    """Dari daftar baris per seed, hasilkan rata-rata, minimum, dan maksimum per posisi."""
    rows = list(zip(*columns))
    return [mean(row) for row in rows], [min(row) for row in rows], [max(row) for row in rows]


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


def plot_series(ax, histories, column, color, label):
    average, low, high = summarize([[float(row[column]) for row in history] for history in histories])
    steps = range(len(average))
    ax.fill_between(steps, low, high, color=color, alpha=0.15, linewidth=0)
    ax.plot(steps, average, color=color, linewidth=2, label=label)


def plot_convergence(histories, path):
    """Fitness dan recall latih per generasi; fitness memakai skor parsial, recall skor ketat."""
    blue = METHODS["evolkv_ga"][1]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, facecolor=SURFACE)
    plot_series(top, histories, "mean_fitness", MUTED, "Rata-rata populasi")
    plot_series(top, histories, "best_fitness", blue, "Terbaik")
    plot_series(bottom, histories, "train_recall", blue, "Kandidat terbaik")

    groups = {}
    for step, row in enumerate(histories[0]):
        groups.setdefault((int(row["first_layer"]), int(row["last_layer"])), []).append(step)
    for ax in (top, bottom):
        ax.set_ylim(bottom=0)
        for members in groups.values():
            if members[0] > 0:
                ax.axvline(members[0] - 0.5, color=GRID, linewidth=1)
    y_top = top.get_ylim()[1]
    top.set_ylim(0, y_top * 1.12)
    for (first, last), members in groups.items():
        top.text((members[0] + members[-1]) / 2, y_top * 1.04, f"Layer {first}–{last}", ha="center", color=MUTED, fontsize=9)

    top.set_ylabel("Fitness (skor parsial)")
    top.set_title(f"Konvergensi Genetic Algorithm, rata-rata dan rentang {len(histories)} seed", loc="left")
    top.legend(frameon=False, labelcolor=MUTED, loc="lower right")
    bottom.set_xlabel("Generasi kumulatif")
    bottom.set_ylabel("Recall latih (skor ketat)")
    style(top)
    style(bottom)
    save(fig, path)


def plot_budgets(all_results, path):
    """Alokasi per layer pada target utama, seperti Gambar 2a di paper."""
    args = all_results[0]["args"]
    target = args["target"]
    average, low, high = summarize([results["targets"][str(target)]["budgets"] for results in all_results])
    layers = len(average)

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    blue = METHODS["evolkv_ga"][1]
    ax.fill_between(range(layers), low, high, color=blue, alpha=0.15, linewidth=0)
    ax.plot(range(layers), average, color=blue, linewidth=2, marker="o", markersize=4, label="EvolKV-GA")
    for method, budgets in (
        ("snapkv", uniform_budgets(layers, target)),
        ("pyramidkv", pyramid_budgets(layers, target, args["window"])),
    ):
        label, color = METHODS[method]
        ax.plot(range(layers), budgets, color=color, linewidth=2, label=label)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Budget KV cache (token)")
    ax.set_title(f"Alokasi budget per layer pada c = {target}, rentang {len(all_results)} seed", loc="left")
    ax.legend(frameon=False, labelcolor=MUTED, loc="upper left")
    style(ax)
    save(fig, path)


def plot_recall(all_results, path):
    """Recall data uji per target budget; garis kecil menunjukkan rentang antar seed."""
    targets = list(all_results[0]["targets"])
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    for offset, method in enumerate(["snapkv", "pyramidkv", "evolkv_ga"]):
        label, color = METHODS[method]
        values, low, high = summarize(
            [[results["targets"][target][method] for target in targets] for results in all_results]
        )
        positions = [i + (offset - 1) * width for i in range(len(targets))]
        errors = [[v - a for v, a in zip(values, low)], [b - v for v, b in zip(values, high)]]
        ax.bar(positions, values, width * 0.9, color=color, label=label)
        ax.errorbar(positions, values, yerr=errors, fmt="none", ecolor=TEXT, elinewidth=1, capsize=3)
        for x, value, top in zip(positions, values, high):
            ax.text(x, top + 0.02, f"{value:.2f}", ha="center", fontsize=8, color=MUTED)

    full = mean(results["full"] for results in all_results)
    ax.axhline(full, color=TEXT, linewidth=1.5, linestyle="--")
    ax.text(-0.45, full + 0.015, f"Full KV {full:.2f}", ha="left", color=MUTED, fontsize=9)
    ax.set_xticks(range(len(targets)), [f"c = {target}" for target in targets])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title(f"Recall data uji, rata-rata dan rentang {len(all_results)} seed", loc="left")
    ax.legend(frameon=False, labelcolor=MUTED, loc="upper left", ncols=3)
    style(ax)
    ax.grid(axis="x", visible=False)
    save(fig, path)


def print_summary(all_results):
    print(f"Full KV: {mean(results['full'] for results in all_results):.3f}")
    print(f"{'c':>6} {'SnapKV':>8} {'PyramidKV':>10} {'EvolKV-GA':>10} {'rentang EvolKV-GA':>18}")
    for target in all_results[0]["targets"]:
        values = {m: [results["targets"][target][m] for results in all_results] for m in METHODS}
        evolved = values["evolkv_ga"]
        print(
            f"{target:>6} {mean(values['snapkv']):>8.3f} {mean(values['pyramidkv']):>10.3f} "
            f"{mean(evolved):>10.3f} {f'{min(evolved):.3f}-{max(evolved):.3f}':>18}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs", help="folder berisi seed_*/history.csv dan seed_*/results.json")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    run_dirs = sorted(Path(args.runs).glob("seed_*"))
    if not run_dirs:
        raise SystemExit(f"Tidak ada folder seed_* di {args.runs}; jalankan run.py dulu")

    histories, all_results = [], []
    for run_dir in run_dirs:
        with open(run_dir / "history.csv", newline="") as file:
            histories.append(list(csv.DictReader(file)))
        with open(run_dir / "results.json") as file:
            all_results.append(json.load(file))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print_summary(all_results)
    plot_convergence(histories, output_dir / "convergence.png")
    plot_budgets(all_results, output_dir / "budgets.png")
    plot_recall(all_results, output_dir / "recall.png")


if __name__ == "__main__":
    main()
