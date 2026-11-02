"""Cari budget KV cache per layer dengan GA, lalu bandingkan dengan baseline pada data uji."""

import argparse
import csv
import json

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
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, help="default mengikuti rumus paper")
    parser.add_argument("--high", type=int, help="batas atas gen, default 4 * target")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--history", default="history.csv", help="riwayat GA per generasi untuk plot.py")
    return parser.parse_args()


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

    def prepare(count, seed):
        samples = make_samples(tokenizer, count, args.context_tokens, args.needles, seed)
        return [(sample, compressor.prefill(sample.input_ids)) for sample in samples]

    def evaluate(prepared, budgets):
        scores = [
            sample.score(compressor.generate(prefilled, budgets, args.max_new_tokens))
            for sample, prefilled in prepared
        ]
        return sum(scores) / len(scores)

    print(f"{num_layers} layer, menyiapkan {args.train_samples} sampel latih dan {args.test_samples} sampel uji")
    train = prepare(args.train_samples, args.seed)
    test = prepare(args.test_samples, args.seed + 1)

    config = GAConfig(
        population_size=args.population or population_size(args.group_size),
        generations=args.generations,
    )
    history = []
    budgets, best_fitness = evolkv_search(
        lambda candidate: evaluate(train, candidate),
        num_layers,
        args.target,
        low=args.window,
        high=args.high or 4 * args.target,
        group_size=args.group_size,
        config=config,
        seed=args.seed,
        history=history,
    )
    with open(args.history, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"fitness terbaik {best_fitness:.4f}, budget mentah {budgets}")

    results = {"full": evaluate(test, [10**9] * num_layers), "targets": {}}
    print(f"\nRecall uji\nFull KV: {results['full']:.3f}")
    print(f"{'c':>6} {'SnapKV':>8} {'PyramidKV':>10} {'EvolKV-GA':>10}")
    for target in [args.target, *args.expand]:
        evolved = complete_budgets(budgets, target)
        row = {
            "snapkv": evaluate(test, uniform_budgets(num_layers, target)),
            "pyramidkv": evaluate(test, pyramid_budgets(num_layers, target, args.window)),
            "evolkv_ga": evaluate(test, evolved),
            "budgets": evolved,
        }
        results["targets"][target] = row
        print(f"{target:>6} {row['snapkv']:>8.3f} {row['pyramidkv']:>10.3f} {row['evolkv_ga']:>10.3f}")

    results["args"] = vars(args)
    with open(args.output, "w") as file:
        json.dump(results, file, indent=2)
    print(f"\nHasil disimpan di {args.output} dan {args.history}")


if __name__ == "__main__":
    main()
