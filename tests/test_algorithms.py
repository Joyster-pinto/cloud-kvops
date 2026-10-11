"""
Unit Tests for LLM-KVOps Algorithmic Engine & Data Structures
Verifies correctness, boundary conditions, and asymptotic properties.
"""

import unittest
import numpy as np

from src.algorithms.rabin_karp import RabinKarpHasher, PrefixCacheTrie
from src.algorithms.sweep_line import BentleyOttmannLineSweep, TokenInterval
from src.algorithms.convex_hull import GrahamsScanConvexHull, GeometricEntityMerger, Point2D
from src.algorithms.knapsack_dp import KnapsackTokenEvictor
from src.algorithms.layer_allocator import LayerBudgetAllocator
from src.engine.kv_cache_manager import KVCacheManager


class TestRabinKarpDeduplication(unittest.TestCase):
    def setUp(self):
        self.hasher = RabinKarpHasher()
        self.trie = PrefixCacheTrie(self.hasher)

    def test_rolling_hash_matches_static_hash(self):
        tokens = [101, 2054, 2003, 1037, 2742, 6251, 102, 345, 876, 12]
        window_size = 4
        rolling_results = self.hasher.compute_window_hashes(tokens, window_size)

        for start_idx, r_hash in rolling_results:
            window_slice = tokens[start_idx : start_idx + window_size]
            static_hash = self.hasher.compute_hash(window_slice)
            self.assertEqual(r_hash, static_hash, f"Rolling hash mismatch at index {start_idx}")

    def test_prefix_trie_zero_copy_sharing(self):
        system_prompt_tokens = [12, 45, 78, 99, 105, 332]
        self.trie.insert_prefix(system_prompt_tokens, cache_ref_id="system_prompt_v1")

        query_1 = [12, 45, 78, 99, 105, 332, 500, 600, 700]
        matched_len, ref_id, saved_slots = self.trie.match_longest_prefix(query_1)
        self.assertEqual(matched_len, 6)
        self.assertEqual(ref_id, "system_prompt_v1")
        self.assertEqual(saved_slots, 6)


class TestBentleyOttmannLineSweep(unittest.TestCase):
    def setUp(self):
        self.sweep = BentleyOttmannLineSweep()

    def test_overlapping_intervals_detection(self):
        intervals = [
            TokenInterval(token_idx=0, t_start=1.0, t_end=4.0, layer_id=1, text="National"),
            TokenInterval(token_idx=1, t_start=2.0, t_end=5.0, layer_id=1, text="Aeronautics"),
            TokenInterval(token_idx=2, t_start=3.0, t_end=6.0, layer_id=1, text="Space"),
            TokenInterval(token_idx=3, t_start=10.0, t_end=12.0, layer_id=1, text="FarAway")
        ]
        intersections = self.sweep.find_intersections(intervals)
        self.assertIn((0, 1), intersections)
        self.assertIn((1, 2), intersections)
        self.assertIn((0, 2), intersections)
        self.assertNotIn((0, 3), intersections)

    def test_coactive_clustering(self):
        intervals = [
            TokenInterval(token_idx=0, t_start=1.0, t_end=3.0, layer_id=1),
            TokenInterval(token_idx=1, t_start=2.5, t_end=4.5, layer_id=1),
            TokenInterval(token_idx=2, t_start=4.0, t_end=5.5, layer_id=1),
            TokenInterval(token_idx=5, t_start=20.0, t_end=22.0, layer_id=1),
            TokenInterval(token_idx=6, t_start=21.0, t_end=23.0, layer_id=1)
        ]
        clusters = self.sweep.cluster_coactive_tokens(intervals)
        self.assertIn([0, 1, 2], clusters)
        self.assertIn([5, 6], clusters)


class TestGrahamsScanConvexHull(unittest.TestCase):
    def setUp(self):
        self.hull_solver = GrahamsScanConvexHull()
        self.merger = GeometricEntityMerger()

    def test_convex_hull_square_with_interior_points(self):
        points = [
            Point2D(0, 0), Point2D(4, 0), Point2D(4, 4), Point2D(0, 4),
            Point2D(2, 2), Point2D(1, 1), Point2D(3, 3)  # Interior points
        ]
        hull = self.hull_solver.compute_hull(points)
        self.assertEqual(len(hull), 4)

    def test_entity_cluster_centroid_merging(self):
        cluster_indices = [10, 11, 12, 13]
        fake_kvs = np.zeros((20, 4), dtype=np.float32)
        fake_kvs[10] = [1.0, 2.0, 3.0, 4.0]
        fake_kvs[11] = [3.0, 4.0, 5.0, 6.0]
        fake_kvs[12] = [5.0, 6.0, 7.0, 8.0]
        fake_kvs[13] = [7.0, 8.0, 9.0, 10.0]
        texts = [f"w_{i}" for i in range(20)]

        res = self.merger.merge_token_cluster(cluster_indices, fake_kvs, texts)
        self.assertEqual(res["original_count"], 4)
        self.assertEqual(res["compressed_count"], 1)
        self.assertEqual(res["memory_saved_slots"], 3)
        expected_centroid = np.array([[4.0, 5.0, 6.0, 7.0]], dtype=np.float32)
        np.testing.assert_allclose(res["centroid_vector"], expected_centroid)


class TestKnapsackDPEviction(unittest.TestCase):
    def setUp(self):
        self.evictor = KnapsackTokenEvictor(num_sink_tokens=2)

    def test_attention_sinks_guaranteed_retention(self):
        n = 10
        att_weights = np.array([0.01, 0.01, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        token_positions = np.arange(n)
        scores = self.evictor.compute_utility_scores(att_weights, token_positions, current_time_step=10)

        # Retain budget of 4 tokens
        retained, total_util, _ = self.evictor.solve_01_knapsack(scores, token_weights=None, budget_capacity=4)
        self.assertIn(0, retained, "Sink token 0 must be retained")
        self.assertIn(1, retained, "Sink token 1 must be retained")
        self.assertEqual(len(retained), 4)

    def test_dp_optimality_over_greedy(self):
        # Specific knapsack test case
        values = np.array([10.0, 10.0, 6.0, 10.0, 12.0])  # token 0 and 1 are sinks
        retained, total_util, _ = self.evictor.solve_01_knapsack(values, token_weights=None, budget_capacity=3)
        self.assertEqual(len(retained), 3)
        self.assertIn(0, retained)
        self.assertIn(1, retained)
        self.assertIn(4, retained)  # Max value among candidates is 12.0 at index 4


class TestLayerAllocator(unittest.TestCase):
    def setUp(self):
        self.allocator = LayerBudgetAllocator(num_layers=8)

    def test_budget_conservation(self):
        total_budget = 320
        budgets = self.allocator.allocate_greedy_budgets(total_token_budget=total_budget)
        self.assertEqual(sum(budgets), total_budget, "Sum of layer budgets must equal total budget")
        self.assertEqual(len(budgets), 8)
        # Middle layers should receive more budget than outer layers
        self.assertGreater(budgets[3], budgets[0])


class TestKVCacheManagerIntegration(unittest.TestCase):
    def setUp(self):
        self.manager = KVCacheManager(num_layers=16, num_heads=4, head_dim=64, target_budget_ratio=0.5)

    def test_end_to_end_context_step(self):
        n = 100
        token_ids = list(range(n))
        token_texts = [f"word_{i}" for i in range(n)]
        att_matrix = np.random.uniform(0.1, 1.0, size=n).astype(np.float32)

        res = self.manager.process_context_step(token_ids, token_texts, att_matrix, current_step=100)
        self.assertEqual(res["total_tokens"], 100)
        self.assertLessEqual(res["compressed_slots"], 55)
        self.assertGreater(res["memory_saved_pct"], 40.0)
        self.assertIn("uncompressed_mb", res)
        self.assertIn("compressed_mb", res)


if __name__ == "__main__":
    unittest.main()
