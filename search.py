"""Bagian EvolKV yang tidak bergantung pada model: fitness dan Algoritma 1."""

import math
import random
from typing import Callable, Sequence

from ga import GAConfig, evolve


def cache_score(budgets: Sequence[int], target: float, gamma: float) -> float:
    """Persamaan 2: efisiensi cache, bernilai lebih rendah bila melewati target."""
    mean_budget = sum(budgets) / len(budgets)
    if mean_budget > target:
        return max(0.0, 1 - (mean_budget - target) / target)
    return 1 - gamma * (1 - mean_budget / target)


def fitness_score(performance: float, budgets: Sequence[int], target: float, lam: float, gamma: float) -> float:
    """Persamaan 1: f(S) * (1 + lambda * CacheScore(S, c))."""
    return performance * (1 + lam * cache_score(budgets, target, gamma))


def complete_budgets(budgets: Sequence[int], target: float) -> list[int]:
    """Budget completion (bagian 3.2): sebarkan selisih ke target sesuai porsi layer.

    Dipakai juga untuk memperluas hasil c = 128 ke target yang lebih besar.
    """
    achieved = sum(budgets)
    delta = target * len(budgets) - achieved
    return [math.ceil(k + k / achieved * delta) for k in budgets]


def fit_budgets(budgets: Sequence[int], target: int, low: int) -> list[int]:
    """Skalakan bagian di atas `low` agar rata-rata budget tepat `target`.

    Semua kandidat GA melewati fungsi ini, jadi gen berarti proporsi antar-layer:
    mengurangi satu layer langsung memberi budget ke layer lain, dan yang
    dievaluasi saat pencarian sama dengan yang dipakai saat pengujian.
    """
    excess = [k - low for k in budgets]
    spare = (target - low) * len(budgets)
    total = sum(excess)
    if total == 0:
        return [target] * len(budgets)
    scaled = [e * spare / total for e in excess]
    fitted = [math.floor(s) for s in scaled]
    by_remainder = sorted(range(len(scaled)), key=lambda i: fitted[i] - scaled[i])
    for i in by_remainder[: spare - sum(fitted)]:
        fitted[i] += 1
    return [low + k for k in fitted]


def uniform_budgets(num_layers: int, target: int) -> list[int]:
    """Alokasi SnapKV: budget sama di semua layer."""
    return [target] * num_layers


def pyramid_budgets(num_layers: int, target: int, window: int, beta: int = 20) -> list[int]:
    """Alokasi PyramidKV sesuai KVCache-Factory: token terpilih menurun linear, ditambah window."""
    capacity = target - window
    min_selected = capacity // beta
    max_selected = capacity * 2 - min_selected
    step = (max_selected - min_selected) // (num_layers - 1)
    return [max_selected - i * step + window for i in range(num_layers)]


def population_size(group_size: int) -> int:
    """Rumus empiris paper: 4 + floor(3 * ln(n_g))."""
    return 4 + math.floor(3 * math.log(group_size))


def evolkv_search(
    evaluate: Callable[[list[int]], float],
    num_layers: int,
    target: int,
    low: int,
    high: int,
    group_size: int,
    config: GAConfig,
    lam: float = 0.3,
    gamma: float = 0.2,
    seed: int = 0,
    log: Callable[[str], None] = print,
    history: list[dict] | None = None,
    describe: Callable[[list[int]], dict] | None = None,
) -> tuple[list[int], float]:
    """Algoritma 1 dengan GA sebagai optimizer: optimasi grup demi grup dari layer bawah.

    `evaluate` menerima budget semua layer dan mengembalikan performa tugas f(S).
    Bila `history` diberikan, satu baris per generasi ditambahkan ke sana, dengan
    kolom tambahan dari `describe` untuk kandidat terbaik generasi itu.
    """
    rng = random.Random(seed)
    best_budgets = [target] * num_layers
    best_fitness = -math.inf

    for start in range(0, num_layers, group_size):
        end = min(start + group_size, num_layers)
        frozen = list(best_budgets)

        def fitted(chromosome: tuple[int, ...]) -> list[int]:
            return fit_budgets(frozen[:start] + list(chromosome) + frozen[end:], target, low)

        def group_fitness(chromosome: tuple[int, ...]) -> float:
            candidate = fitted(chromosome)
            return fitness_score(evaluate(candidate), candidate, target, lam, gamma)

        def report(generation: int, best: tuple[int, ...], value: float, mean_value: float) -> None:
            log(f"layer {start}-{end - 1} | generasi {generation} | fitness {value:.4f} | {list(best)}")
            if history is not None:
                history.append(
                    {
                        "first_layer": start,
                        "last_layer": end - 1,
                        "generation": generation,
                        "best_fitness": value,
                        "mean_fitness": mean_value,
                        "best_chromosome": " ".join(map(str, best)),
                        **(describe(fitted(best)) if describe else {}),
                    }
                )

        chromosome, value = evolve(group_fitness, frozen[start:end], low, high, config, rng, report)
        if value > best_fitness:
            best_fitness = value
            best_budgets = frozen[:start] + list(chromosome) + frozen[end:]

    return fit_budgets(best_budgets, target, low), best_fitness
