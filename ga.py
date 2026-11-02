"""Genetic Algorithm yang menggantikan CMA-ES sebagai optimizer EvolKV.

Satu kromosom adalah budget KV cache untuk satu grup layer: daftar n_g bilangan
bulat, satu gen per layer. GA hanya melihat kromosom dan fungsi fitness, jadi
modul ini tidak tahu apa pun soal model.
"""

import random
from dataclasses import dataclass
from typing import Callable, Sequence

Chromosome = tuple[int, ...]


@dataclass
class GAConfig:
    population_size: int = 10
    generations: int = 8
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.25  # peluang mutasi per gen
    mutation_sigma: float = 0.3  # simpangan mutasi, relatif terhadap rentang gen
    elite_count: int = 1


def tournament_select(scored: list[tuple[Chromosome, float]], size: int, rng: random.Random) -> Chromosome:
    contestants = rng.sample(scored, min(size, len(scored)))
    return max(contestants, key=lambda item: item[1])[0]


def uniform_crossover(a: Chromosome, b: Chromosome, rate: float, rng: random.Random) -> tuple[Chromosome, Chromosome]:
    if rng.random() >= rate:
        return a, b
    mask = [rng.random() < 0.5 for _ in a]
    child_a = tuple(x if keep else y for x, y, keep in zip(a, b, mask))
    child_b = tuple(y if keep else x for x, y, keep in zip(a, b, mask))
    return child_a, child_b


def gaussian_mutation(
    chromosome: Chromosome, low: int, high: int, rate: float, sigma: float, rng: random.Random
) -> Chromosome:
    genes = []
    for gene in chromosome:
        if rng.random() < rate:
            gene = round(gene + rng.gauss(0, sigma * (high - low)))
        genes.append(min(high, max(low, gene)))
    return tuple(genes)


def initial_population(seed: Sequence[int], low: int, high: int, size: int, rng: random.Random) -> list[Chromosome]:
    """Individu pertama adalah solusi saat ini, sisanya acak di rentang gen."""
    population = [tuple(seed)]
    while len(population) < size:
        population.append(tuple(rng.randint(low, high) for _ in seed))
    return population


def evolve(
    fitness: Callable[[Chromosome], float],
    seed: Sequence[int],
    low: int,
    high: int,
    config: GAConfig,
    rng: random.Random,
    on_generation: Callable[[int, Chromosome, float, float], None] | None = None,
) -> tuple[Chromosome, float]:
    """Jalankan GA dan kembalikan kromosom terbaik beserta fitness-nya.

    Evaluasi fitness itu mahal (inferensi model), jadi hasilnya disimpan per
    kromosom dan individu yang muncul lagi tidak dievaluasi ulang.
    """
    memo: dict[Chromosome, float] = {}

    def score(population: list[Chromosome]) -> list[tuple[Chromosome, float]]:
        for chromosome in population:
            if chromosome not in memo:
                memo[chromosome] = fitness(chromosome)
        return sorted(((c, memo[c]) for c in population), key=lambda item: item[1], reverse=True)

    scored = score(initial_population(seed, low, high, config.population_size, rng))
    best, best_fitness = scored[0]

    def report(generation: int) -> None:
        if on_generation:
            mean_fitness = sum(value for _, value in scored) / len(scored)
            on_generation(generation, best, best_fitness, mean_fitness)

    report(0)

    for generation in range(1, config.generations + 1):
        next_population = [c for c, _ in scored[: config.elite_count]]
        while len(next_population) < config.population_size:
            parent_a = tournament_select(scored, config.tournament_size, rng)
            parent_b = tournament_select(scored, config.tournament_size, rng)
            for child in uniform_crossover(parent_a, parent_b, config.crossover_rate, rng):
                mutated = gaussian_mutation(child, low, high, config.mutation_rate, config.mutation_sigma, rng)
                if len(next_population) < config.population_size:
                    next_population.append(mutated)

        scored = score(next_population)
        if scored[0][1] > best_fitness:
            best, best_fitness = scored[0]
        report(generation)

    return best, best_fitness
