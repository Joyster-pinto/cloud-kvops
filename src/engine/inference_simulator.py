"""
LLM-KVOps: Long-Context LLM Inference Simulator & Hardware Profiler
Simulates autoregressive decoding and tracks real-time RAM/VRAM resource usage across methods.
"""

from typing import List, Dict, Tuple, Any, Optional
import time
import os
import psutil
import numpy as np

from src.engine.kv_cache_manager import KVCacheManager

# Standard LLM Architecture Presets
MODEL_PRESETS = {
    "LLaMA-3-70B (80L, 8H, 128D)": {
        "num_layers": 80, "num_heads": 8, "head_dim": 128, "bytes_per_elem": 2, "name": "LLaMA-3-70B"
    },
    "LLaMA-3-8B (32L, 8H, 128D)": {
        "num_layers": 32, "num_heads": 8, "head_dim": 128, "bytes_per_elem": 2, "name": "LLaMA-3-8B"
    },
    "Mistral-7B (32L, 8H, 128D)": {
        "num_layers": 32, "num_heads": 8, "head_dim": 128, "bytes_per_elem": 2, "name": "Mistral-7B"
    },
    "GPT-2 Medium (24L, 16H, 64D)": {
        "num_layers": 24, "num_heads": 16, "head_dim": 64, "bytes_per_elem": 2, "name": "GPT-2-Medium"
    }
}

class InferenceSimulator:
    """
    Simulates sequential token generation and benchmarks hardware metrics.
    """
    def __init__(self, model_key: str = "LLaMA-3-70B (80L, 8H, 128D)", target_budget_ratio: float = 0.40):
        self.preset = MODEL_PRESETS.get(model_key, MODEL_PRESETS["LLaMA-3-70B (80L, 8H, 128D)"])
        self.manager = KVCacheManager(
            num_layers=self.preset["num_layers"],
            num_heads=self.preset["num_heads"],
            head_dim=self.preset["head_dim"],
            bytes_per_elem=self.preset["bytes_per_elem"],
            target_budget_ratio=target_budget_ratio
        )
        self.process = psutil.Process(os.getpid())

    def get_system_ram_mb(self) -> float:
        """Returns current process RAM usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    def generate_synthetic_attention(self, seq_len: int) -> np.ndarray:
        """
        Synthesizes a realistic Transformer attention distribution:
        - Massive spike on first 4 tokens (Attention Sinks).
        - Random spikes on heavy-hitter keywords / entities.
        - Higher concentration on recent tokens (Locality).
        """
        att = np.random.exponential(scale=0.2, size=seq_len).astype(np.float32)
        
        # 1. Attention Sinks (First 4 tokens absorb ~20-30% of attention)
        sink_len = min(4, seq_len)
        att[:sink_len] += np.array([2.5, 1.8, 1.2, 0.9][:sink_len])

        # 2. Heavy Hitters (Entities spaced across the document)
        num_hitters = max(1, seq_len // 40)
        hitter_indices = np.random.choice(range(sink_len, seq_len), size=min(num_hitters, seq_len - sink_len), replace=False)
        att[hitter_indices] += np.random.uniform(1.2, 2.0, size=len(hitter_indices))

        # 3. Recency bias (last 15% of tokens)
        recent_cutoff = int(seq_len * 0.85)
        att[recent_cutoff:] += np.linspace(0.2, 1.0, seq_len - recent_cutoff)

        # Normalize to sum to 1.0
        att_norm = att / np.sum(att)
        return att_norm

    def run_side_by_side_simulation(
        self,
        token_count: int = 4096,
        sample_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs a side-by-side simulation comparing:
        1. Baseline (Without Eviction / Full Uncompressed KV-Cache)
        2. Sliding Window Baseline (Window = Budget)
        3. LLM-KVOps (Combinatorial Eviction with Knapsack + Geometry + Suffix Hash)
        """
        # Generate token representations
        if sample_text:
            words = sample_text.split()
            if len(words) < token_count:
                # Loop sample text to reach token_count
                reps = (token_count // len(words)) + 1
                words = (words * reps)[:token_count]
            else:
                words = words[:token_count]
            token_texts = words
        else:
            token_texts = [f"tok_{i}" for i in range(token_count)]

        token_ids = list(range(token_count))
        bytes_per_slot = self.manager.bytes_per_token_slot()

        # Attention distribution
        attention_weights = self.generate_synthetic_attention(token_count)

        # 1. Baseline Full KV-Cache
        baseline_vram_bytes = token_count * bytes_per_slot
        baseline_vram_mb = baseline_vram_bytes / (1024 * 1024)
        baseline_vram_gb = baseline_vram_bytes / (1024 * 1024 * 1024)

        # 2. Sliding Window Baseline (Naive sliding window of size = Budget)
        budget_capacity = max(self.manager.min_sink_tokens + 4, int(token_count * self.manager.target_budget_ratio))
        sliding_vram_bytes = budget_capacity * bytes_per_slot
        sliding_vram_mb = sliding_vram_bytes / (1024 * 1024)

        # 3. Run LLM-KVOps
        kvops_result = self.manager.process_context_step(
            token_ids=token_ids,
            token_texts=token_texts,
            attention_matrix=attention_weights,
            current_step=token_count
        )

        # Time Series Simulation (Progressive memory growth during generation)
        step_intervals = np.linspace(100, token_count, num=min(20, max(5, token_count // 200 + 2)), dtype=int)
        time_series = {
            "steps": [],
            "baseline_vram_mb": [],
            "sliding_window_vram_mb": [],
            "kvops_vram_mb": [],
            "kvops_saved_pct": [],
            "kvops_latency_ms": []
        }

        for step_val in step_intervals:
            step = int(step_val)
            b_mb = (float(step) * float(bytes_per_slot)) / (1024.0 * 1024.0)
            b_cap = max(self.manager.min_sink_tokens + 4, int(step * self.manager.target_budget_ratio))
            sw_mb = (float(b_cap) * float(bytes_per_slot)) / (1024.0 * 1024.0)

            # Step sub-slice
            sub_ids = token_ids[:step]
            sub_texts = token_texts[:step]
            sub_att = self.generate_synthetic_attention(step)
            step_res = self.manager.process_context_step(sub_ids, sub_texts, sub_att, current_step=step)

            time_series["steps"].append(int(step))
            time_series["baseline_vram_mb"].append(float(b_mb))
            time_series["sliding_window_vram_mb"].append(float(sw_mb))
            time_series["kvops_vram_mb"].append(float(step_res["compressed_mb"]))
            time_series["kvops_saved_pct"].append(float(step_res["memory_saved_pct"]))
            time_series["kvops_latency_ms"].append(float(step_res["latency_overhead_ms"]))


        # Calculate Quality / Context Retention Score (Cumulative attention mass preserved)
        retained_set = set(kvops_result["retained_indices"])
        kvops_attention_retained = float(np.sum(attention_weights[list(retained_set)])) * 100.0
        
        sliding_retained = set(range(max(0, token_count - budget_capacity), token_count))
        sliding_attention_retained = float(np.sum(attention_weights[list(sliding_retained)])) * 100.0

        return {
            "model_name": self.preset["name"],
            "token_count": token_count,
            "bytes_per_slot": bytes_per_slot,
            "baseline": {
                "vram_bytes": baseline_vram_bytes,
                "vram_mb": baseline_vram_mb,
                "vram_gb": baseline_vram_gb,
                "attention_retained_pct": 100.0
            },
            "sliding_window": {
                "vram_mb": sliding_vram_mb,
                "attention_retained_pct": sliding_attention_retained
            },
            "kvops": {
                "vram_mb": kvops_result["compressed_mb"],
                "vram_gb": kvops_result["compressed_gb"],
                "memory_saved_mb": kvops_result["memory_saved_mb"],
                "memory_saved_pct": kvops_result["memory_saved_pct"],
                "attention_retained_pct": kvops_attention_retained,
                "latency_overhead_ms": kvops_result["latency_overhead_ms"],
                "tier_latencies_ms": kvops_result["tier_latencies_ms"],
                "retained_count": kvops_result["retained_count"],
                "merged_clusters_count": kvops_result["merged_clusters_count"],
                "token_status": kvops_result["token_status"],
                "layer_budgets": kvops_result["layer_budgets"]
            },
            "time_series": time_series,
            "token_texts": token_texts[:min(500, len(token_texts))],
            "token_status_sample": kvops_result["token_status"][:min(500, len(kvops_result["token_status"]))]
        }
