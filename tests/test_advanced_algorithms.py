"""
Unit Tests for Advanced KV Cache Optimization Algorithms
Verifies:
1. H2O Attention Accumulator (EMA tracking, Heavy Hitters, Attention Sinks)
2. KIVI Dynamic Per-Channel INT8 / INT4 Quantizer (Cosine fidelity > 0.99, byte savings)
3. PagedAttention Virtual Memory Block Manager (O(1) translation, block recycling, fragmentation)
"""

import unittest
import numpy as np

from src.algorithms.h2o_accumulator import H2OAttentionAccumulator
from src.algorithms.kv_quantizer import KVIQuantizer, QuantizedTensor
from src.algorithms.paged_cache import PagedKVCacheManager, PhysicalBlock


class TestH2OAttentionAccumulator(unittest.TestCase):
    def setUp(self):
        self.accumulator = H2OAttentionAccumulator(
            decay_factor=0.6,
            num_sink_tokens=3,
            recent_window_tokens=2
        )

    def test_ema_accumulation(self):
        # Step 1
        att1 = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        scores1 = self.accumulator.update(att1)
        np.testing.assert_allclose(scores1, att1, rtol=1e-5)

        # Step 2: token at index 3 gets a spike
        att2 = np.array([0.1, 0.1, 0.1, 0.7], dtype=np.float32)
        scores2 = self.accumulator.update(att2)
        expected = 0.6 * att1 + 0.4 * att2
        np.testing.assert_allclose(scores2, expected, rtol=1e-5)

    def test_heavy_hitters_and_sink_preservation(self):
        # 10 tokens
        # Sinks: indices 0, 1, 2
        # Recent: indices 8, 9
        # Intermediate: 3, 4, 5, 6, 7
        att = np.array([0.05, 0.05, 0.05, 0.01, 0.40, 0.02, 0.30, 0.02, 0.05, 0.05], dtype=np.float32)
        self.accumulator.update(att)

        # Budget of 6 tokens
        # Expect: 3 sinks (0,1,2) + 2 recent (8,9) + 1 top heavy hitter from intermediate (index 4 with score 0.40)
        retained, sinks, heavy = self.accumulator.get_heavy_hitters(budget_capacity=6)
        self.assertEqual(len(retained), 6)
        self.assertIn(0, sinks)
        self.assertIn(1, sinks)
        self.assertIn(2, sinks)
        self.assertIn(4, heavy)
        self.assertIn(8, retained)
        self.assertIn(9, retained)


class TestKVIQuantizer(unittest.TestCase):
    def setUp(self):
        self.quantizer_8bit = KVIQuantizer(bits=8)
        self.quantizer_4bit = KVIQuantizer(bits=4)

    def test_int8_quantization_fidelity(self):
        np.random.seed(42)
        # Random realistic Key tensor of shape (32 tokens, 128 head_dim)
        original_k = np.random.randn(32, 128).astype(np.float32) * 0.5 + 0.1

        q_container = self.quantizer_8bit.quantize(original_k)
        reconstructed_k = self.quantizer_8bit.dequantize(q_container)

        self.assertEqual(q_container.quantized_data.dtype, np.int8)
        self.assertEqual(reconstructed_k.shape, original_k.shape)

        metrics = KVIQuantizer.compute_fidelity_metrics(original_k, reconstructed_k)
        # Cosine similarity should exceed 0.99 for 8-bit quantization
        self.assertGreater(metrics["cosine_similarity"], 0.99)
        self.assertLess(metrics["mean_squared_error"], 0.01)

    def test_int4_quantization_fidelity(self):
        np.random.seed(42)
        original_v = np.random.randn(16, 64).astype(np.float32) * 0.8

        q_container = self.quantizer_4bit.quantize(original_v)
        reconstructed_v = self.quantizer_4bit.dequantize(q_container)

        self.assertEqual(q_container.bits, 4)
        # Packed array should take half the number of elements
        self.assertEqual(len(q_container.quantized_data), (16 * 64) // 2)

        metrics = KVIQuantizer.compute_fidelity_metrics(original_v, reconstructed_v)
        # Cosine similarity should comfortably exceed 0.95 for 4-bit quantization
        self.assertGreater(metrics["cosine_similarity"], 0.95)

    def test_memory_savings_calculation(self):
        shape = (100, 128)  # 12,800 elements
        # FP16 = 25,600 bytes
        comp_8bit = KVIQuantizer.compute_compression_ratio(shape, bits=8, fp_precision_bytes=2)
        self.assertEqual(comp_8bit["uncompressed_bytes"], 25600)
        self.assertEqual(comp_8bit["compressed_bytes"], 12800)
        self.assertEqual(comp_8bit["savings_pct"], 50.0)

        comp_4bit = KVIQuantizer.compute_compression_ratio(shape, bits=4, fp_precision_bytes=2)
        self.assertEqual(comp_4bit["compressed_bytes"], 6400)
        self.assertEqual(comp_4bit["savings_pct"], 75.0)


class TestPagedKVCacheManager(unittest.TestCase):
    def setUp(self):
        # 10 blocks, 4 tokens per block, 2 heads, 8 dim
        self.paged_manager = PagedKVCacheManager(
            total_blocks=10,
            block_size=4,
            num_heads=2,
            head_dim=8
        )

    def test_dynamic_allocation_and_translation(self):
        session_id = "user_req_1"
        key_sample = np.ones((2, 8), dtype=np.float16)
        val_sample = np.ones((2, 8), dtype=np.float16)

        # Append 9 tokens (should require 3 blocks since block_size = 4: 4 + 4 + 1)
        for i in range(9):
            self.paged_manager.append_token_to_session(session_id, key_sample * (i + 1), val_sample * (i + 1))

        # Check block table length
        allocated_blocks = self.paged_manager.block_tables[session_id]
        self.assertEqual(len(allocated_blocks), 3)

        # Test O(1) logical translation
        # Token 0 -> Block 0, offset 0
        b_id_0, off_0 = self.paged_manager.translate_logical_index(session_id, 0)
        self.assertEqual(b_id_0, allocated_blocks[0])
        self.assertEqual(off_0, 0)

        # Token 7 -> Block 1, offset 3
        b_id_7, off_7 = self.paged_manager.translate_logical_index(session_id, 7)
        self.assertEqual(b_id_7, allocated_blocks[1])
        self.assertEqual(off_7, 3)

        # Token 8 -> Block 2, offset 0
        b_id_8, off_8 = self.paged_manager.translate_logical_index(session_id, 8)
        self.assertEqual(b_id_8, allocated_blocks[2])
        self.assertEqual(off_8, 0)

    def test_session_free_and_block_recycling(self):
        session_id = "user_req_2"
        dummy = np.zeros((2, 8), dtype=np.float16)
        for _ in range(8):  # takes 2 blocks
            self.paged_manager.append_token_to_session(session_id, dummy, dummy)

        self.assertEqual(len(self.paged_manager.free_block_ids), 8)  # 10 - 2 = 8 free

        # Release session
        self.paged_manager.free_session(session_id)
        self.assertEqual(len(self.paged_manager.free_block_ids), 10)  # All 10 returned to pool

    def test_fragmentation_metrics(self):
        session_id = "user_req_3"
        dummy = np.zeros((2, 8), dtype=np.float16)
        for _ in range(5):  # 5 tokens in 4-token blocks -> 2 blocks (8 slots allocated, 3 wasted)
            self.paged_manager.append_token_to_session(session_id, dummy, dummy)

        metrics = self.paged_manager.compute_fragmentation_metrics(session_id)
        self.assertEqual(metrics["total_tokens"], 5)
        self.assertEqual(metrics["paged_allocated_slots"], 8)
        self.assertEqual(metrics["paged_wasted_slots"], 3)
        self.assertEqual(metrics["paged_fragmentation_pct"], 37.5)


if __name__ == "__main__":
    unittest.main()
