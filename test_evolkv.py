import random
import unittest

from ga import GAConfig, evolve, gaussian_mutation, uniform_crossover
from search import cache_score, complete_budgets, evolkv_search, fitness_score, population_size, pyramid_budgets


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

    def test_pyramid_decreases_with_target_mean(self):
        budgets = pyramid_budgets(8, 128, 32)
        self.assertEqual(budgets, sorted(budgets, reverse=True))
        self.assertAlmostEqual(sum(budgets) / 8, 128, delta=1)

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

    def test_groups_cover_layers_not_divisible_by_group_size(self):
        budgets, _ = evolkv_search(
            lambda b: 1.0, 21, 128, 32, 512, 8, GAConfig(population_size=4, generations=1), log=lambda _: None
        )
        self.assertEqual(len(budgets), 21)


if __name__ == "__main__":
    unittest.main()
