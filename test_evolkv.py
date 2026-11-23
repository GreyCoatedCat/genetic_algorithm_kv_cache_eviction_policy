import random
import unittest

from ga import GAConfig, evolve, gaussian_mutation, uniform_crossover
from niah import Sample
from search import (
    cache_score,
    complete_budgets,
    evolkv_search,
    fit_budgets,
    fitness_score,
    population_size,
    pyramid_budgets,
)


class GeneticOperators(unittest.TestCase):
    def test_crossover_only_swaps_parent_genes(self):
        rng = random.Random(1)
        a, b = (1, 2, 3, 4), (5, 6, 7, 8)
        child_a, child_b = uniform_crossover(a, b, 1.0, rng)
        for x, y, p, q in zip(child_a, child_b, a, b):
            self.assertEqual({x, y}, {p, q})

    def test_mutation_stays_in_range(self):
        rng = random.Random(2)
        for _ in range(200):
            child = gaussian_mutation((32, 500), 32, 500, 1.0, 0.5, rng)
            self.assertTrue(all(32 <= gene <= 500 for gene in child))

    def test_evolve_finds_optimum(self):
        optimum = (40, 300, 120, 480)
        fitness = lambda c: -sum(abs(x - y) for x, y in zip(c, optimum))
        config = GAConfig(population_size=30, generations=120)
        best, value = evolve(fitness, (250, 250, 250, 250), 0, 500, config, random.Random(3))
        self.assertGreater(value, -30, best)

    def test_evolve_never_worse_than_seed(self):
        fitness = lambda c: -abs(c[0] - 7)
        best, value = evolve(fitness, (7,), 0, 100, GAConfig(generations=3), random.Random(4))
        self.assertEqual(best, (7,))
        self.assertEqual(value, 0)

    def test_fitness_is_not_recomputed_for_repeats(self):
        seen = []

        def fitness(chromosome):
            seen.append(chromosome)
            return 0.0

        evolve(fitness, (1, 1), 0, 3, GAConfig(generations=10), random.Random(5))
        self.assertEqual(len(seen), len(set(seen)))


class Formulas(unittest.TestCase):
    def test_cache_score_on_target_is_one(self):
        self.assertEqual(cache_score([128, 128], 128, 0.2), 1)

    def test_cache_score_over_budget_is_penalized(self):
        self.assertAlmostEqual(cache_score([192], 128, 0.2), 0.5)
        self.assertEqual(cache_score([400], 128, 0.2), 0)

    def test_cache_score_under_budget_is_discounted(self):
        self.assertAlmostEqual(cache_score([64], 128, 0.2), 0.9)

    def test_fitness_rewards_same_performance_with_better_budget(self):
        self.assertGreater(fitness_score(0.5, [128], 128, 0.3, 0.2), fitness_score(0.5, [256], 128, 0.3, 0.2))

    def test_complete_budgets_reaches_target(self):
        completed = complete_budgets([64, 192, 64, 400], 512)
        self.assertAlmostEqual(sum(completed) / 4, 512, delta=1)

    def test_pyramid_matches_kvcache_factory(self):
        for target, first, last in [(128, 220, 36), (256, 469, 55), (512, 968, 71)]:
            budgets = pyramid_budgets(24, target, 32)
            self.assertEqual((budgets[0], budgets[-1]), (first, last))
            self.assertEqual(budgets, sorted(budgets, reverse=True))

    def test_fit_budgets_hits_target_exactly(self):
        for budgets in ([512, 32, 32, 400], [32, 40, 33, 32], [128, 128, 128, 128]):
            fitted = fit_budgets(budgets, 128, 32)
            self.assertEqual(sum(fitted), 128 * 4)
            self.assertGreaterEqual(min(fitted), 32)

    def test_fit_budgets_keeps_proportions_above_floor(self):
        self.assertEqual(fit_budgets([132, 232], 82, 32), [65, 99])

    def test_fit_budgets_all_at_floor_becomes_uniform(self):
        self.assertEqual(fit_budgets([32, 32], 128, 32), [128, 128])

    def test_recall_needs_full_answer_but_partial_score_does_not(self):
        sample = Sample([], ["5279348", "2978347"])
        self.assertEqual(sample.score("52799 2978347"), 0.5)
        self.assertAlmostEqual(sample.partial_score("52799 2978347"), (4 / 7 + 1) / 2)
        self.assertEqual(sample.partial_score("tidak tahu"), 0)
        self.assertEqual(sample.partial_score("5279348 2978347"), 1)

    def test_population_size_matches_paper(self):
        self.assertEqual([population_size(n) for n in (2, 4, 8, 16, 32)], [6, 8, 10, 12, 14])


class GroupSearch(unittest.TestCase):
    def test_search_learns_which_layers_need_budget(self):
        # Hanya layer 2 dan 5 yang berpengaruh; layer lain seharusnya tetap kecil.
        evaluate = lambda budgets: min(budgets[2], 300) + min(budgets[5], 300)
        config = GAConfig(population_size=16, generations=25)
        budgets, _ = evolkv_search(evaluate, 8, 128, 32, 512, 4, config, seed=6, log=lambda _: None)
        others = [b for i, b in enumerate(budgets) if i not in (2, 5)]
        self.assertGreater(min(budgets[2], budgets[5]), max(others))

    def test_every_candidate_respects_the_target(self):
        means = []

        def evaluate(budgets):
            means.append(sum(budgets) / len(budgets))
            return budgets[0]

        budgets, _ = evolkv_search(evaluate, 8, 128, 32, 512, 4, GAConfig(generations=3), log=lambda _: None)
        self.assertEqual(set(means), {128})
        self.assertEqual(sum(budgets), 128 * 8)

    def test_history_records_extra_columns_for_best_candidate(self):
        history = []
        evolkv_search(
            lambda b: b[0], 4, 128, 32, 512, 4, GAConfig(population_size=4, generations=1),
            log=lambda _: None, history=history, describe=lambda b: {"first_budget": b[0]},
        )
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]["first_budget"], history[-1]["best_fitness"] / 1.3)

    def test_groups_cover_layers_not_divisible_by_group_size(self):
        budgets, _ = evolkv_search(
            lambda b: 1.0, 21, 128, 32, 512, 8, GAConfig(population_size=4, generations=1), log=lambda _: None
        )
        self.assertEqual(len(budgets), 21)


if __name__ == "__main__":
    unittest.main()
