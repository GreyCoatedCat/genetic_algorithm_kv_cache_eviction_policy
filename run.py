"""Cari budget KV cache per layer dengan GA, lalu bandingkan dengan baseline pada data uji."""

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ga import GAConfig
from kv_cache import KVCompressor
from niah import make_samples
from search import complete_budgets, evolkv_search, population_size, pyramid_budgets, uniform_budgets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target", type=int, default=128, help="target rata-rata budget per layer (c)")
    parser.add_argument("--expand", type=int, nargs="*", default=[256, 512], help="target tambahan untuk uji hasil")
    parser.add_argument("--group-size", type=int, default=6, help="n_g; paper memakai 8 untuk model 32 layer")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--kernel", type=int, default=7)
    parser.add_argument("--context-tokens", type=int, default=3000)
    parser.add_argument("--needles", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=10)
    parser.add_argument("--test-samples", type=int, default=30)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, help="default mengikuti rumus paper")
    parser.add_argument("--high", type=int, help="batas atas gen, default 4 * target")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", default="runs", help="hasil tiap seed ditulis ke <folder>/seed_<n>")
    return parser.parse_args()


def run_seed(args, seed, tokenizer, compressor, num_layers):
    def prepare(count, sample_seed):
        samples = make_samples(tokenizer, count, args.context_tokens, args.needles, sample_seed)
        return [(sample, compressor.prefill(sample.input_ids)) for sample in samples]

    def generate(prepared, budgets):
        return [(sample, compressor.generate(prefilled, budgets, args.max_new_tokens)) for sample, prefilled in prepared]

    def recall(prepared, budgets):
        scores = [sample.score(text) for sample, text in generate(prepared, budgets)]
        return sum(scores) / len(scores)

    print(f"seed {seed}: menyiapkan {args.train_samples} sampel latih dan {args.test_samples} sampel uji")
    train = prepare(args.train_samples, seed)
    test = prepare(args.test_samples, seed + 1)

    train_recall = {}

    def train_performance(candidate):
        outputs = generate(train, candidate)
        train_recall[tuple(candidate)] = sum(sample.score(text) for sample, text in outputs) / len(outputs)
        return sum(sample.partial_score(text) for sample, text in outputs) / len(outputs)

    config = GAConfig(
        population_size=args.population or population_size(args.group_size),
        generations=args.generations,
    )
    history = []
    budgets, best_fitness = evolkv_search(
        train_performance,
        num_layers,
        args.target,
        low=args.window,
        high=args.high or 4 * args.target,
        group_size=args.group_size,
        config=config,
        seed=seed,
        history=history,
        describe=lambda candidate: {"train_recall": train_recall[tuple(candidate)]},
    )
    print(f"fitness terbaik {best_fitness:.4f}, budget {budgets}")

    results = {"full": recall(test, [10**9] * num_layers), "targets": {}}
    print(f"\nRecall uji seed {seed}\nFull KV: {results['full']:.3f}")
    print(f"{'c':>6} {'SnapKV':>8} {'PyramidKV':>10} {'EvolKV-GA':>10}")
    for target in [args.target, *args.expand]:
        evolved = complete_budgets(budgets, target)
        row = {
            "snapkv": recall(test, uniform_budgets(num_layers, target)),
            "pyramidkv": recall(test, pyramid_budgets(num_layers, target, args.window)),
            "evolkv_ga": recall(test, evolved),
            "budgets": evolved,
        }
        results["targets"][target] = row
        print(f"{target:>6} {row['snapkv']:>8.3f} {row['pyramidkv']:>10.3f} {row['evolkv_ga']:>10.3f}")

    results["args"] = {**vars(args), "seed": seed}
    run_dir = Path(args.output_dir) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "results.json", "w") as file:
        json.dump(results, file, indent=2)
    with open(run_dir / "history.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"\nHasil disimpan di {run_dir}\n")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="eager",
    ).to(device).eval()
    compressor = KVCompressor(model, tokenizer, args.window, args.kernel)
    num_layers = len(model.model.layers)
    print(f"{num_layers} layer")

    for seed in args.seeds:
        run_seed(args, seed, tokenizer, compressor, num_layers)


if __name__ == "__main__":
    main()
