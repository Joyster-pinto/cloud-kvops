"""
LLM-KVOps: Advanced Algorithm 1 - Heavy-Hitter Oracle (H2O) Attention Accumulator
Paper Reference: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models" (NeurIPS)

Maintains an Exponential Moving Average (EMA) of attention scores across autoregressive
generation steps. Prevents dropping critical entity/reasoning tokens whose instantaneous
single-step attention may momentarily dip.
"""

from typing import List, Tuple, Optional
import numpy as np


class H2OAttentionAccumulator:
    """
    Accumulates attention scores over decoding steps using Exponential Moving Average (EMA).
    Identifies persistent 'Heavy Hitter' (H2) tokens and protects initial 'Attention Sinks'.
    """
    def __init__(
        self,
        decay_factor: float = 0.70,
        num_sink_tokens: int = 4,
        recent_window_tokens: int = 32
    ):
        """
        Args:
            decay_factor: Momentum beta in A^{(t)} = beta * A^{(t-1)} + (1 - beta) * alpha^{(t)}.
            num_sink_tokens: Initial tokens with infinite attention retention (Attention Sinks).
            recent_window_tokens: Most recent tokens guaranteed retention.
        """
        self.decay_factor = decay_factor
        self.num_sink_tokens = num_sink_tokens
        self.recent_window_tokens = recent_window_tokens
        self.cumulative_scores: Optional[np.ndarray] = None
        self.step_count: int = 0

    def reset(self):
        """Resets accumulator state for a new request."""
        self.cumulative_scores = None
        self.step_count = 0

    def update(self, step_attention: np.ndarray) -> np.ndarray:
        """
        Updates cumulative attention scores with the attention distribution from the current step.

        Args:
            step_attention: 1D array of shape (N,) containing raw or normalized attention weights.

        Returns:
            Updated 1D cumulative attention array of length N.
        """
        n = len(step_attention)
        if n == 0:
            return np.array([])

        step_att = np.asarray(step_attention, dtype=np.float32)
        norm_sum = np.sum(step_att)
        if norm_sum > 0:
            step_att = step_att / norm_sum

        if self.cumulative_scores is None:
            self.cumulative_scores = step_att.copy()
        else:
            prev_len = len(self.cumulative_scores)
            if n > prev_len:
                # Sequence expanded by new token(s)
                expanded = np.zeros(n, dtype=np.float32)
                expanded[:prev_len] = self.cumulative_scores
                # Newly added tokens initialize with their first attention score
                expanded[prev_len:] = step_att[prev_len:]
                self.cumulative_scores = expanded

            # Apply Exponential Moving Average (EMA)
            self.cumulative_scores = (
                self.decay_factor * self.cumulative_scores[:n] +
                (1.0 - self.decay_factor) * step_att
            )

        self.step_count += 1
        return self.cumulative_scores.copy()

    def get_heavy_hitters(
        self,
        budget_capacity: int,
        current_seq_len: Optional[int] = None
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Extracts top tokens partitioned into:
        1. Attention Sinks (first N tokens)
        2. Recent Window Tokens (last W tokens)
        3. Heavy Hitters (highest cumulative attention tokens in the intermediate span)

        Returns:
            (retained_indices, sink_indices, heavy_hitter_indices)
        """
        if self.cumulative_scores is None:
            n = current_seq_len or 0
            return list(range(min(n, budget_capacity))), [], []

        n = len(self.cumulative_scores)
        if budget_capacity >= n:
            return list(range(n)), list(range(min(self.num_sink_tokens, n))), []

        sink_count = min(self.num_sink_tokens, budget_capacity)
        sink_indices = list(range(sink_count))

        avail_for_recent = max(0, budget_capacity - sink_count)
        recent_count = min(self.recent_window_tokens, avail_for_recent, max(0, n - sink_count))
        recent_start = n - recent_count
        recent_indices = list(range(recent_start, n)) if recent_count > 0 else []

        # Candidate pool: intermediate tokens between sinks and recent window
        cand_indices = [i for i in range(sink_count, recent_start)]
        remaining_budget = max(0, budget_capacity - (len(sink_indices) + len(recent_indices)))

        heavy_hitters = []
        if remaining_budget > 0 and len(cand_indices) > 0:
            cand_scores = self.cumulative_scores[cand_indices]
            # Select top-k based on cumulative attention
            top_indices_local = np.argsort(cand_scores)[-remaining_budget:]
            heavy_hitters = [cand_indices[idx] for idx in sorted(top_indices_local)]

        retained = sorted(list(set(sink_indices + heavy_hitters + recent_indices)))
        return retained, sink_indices, heavy_hitters

    def get_normalized_scores(self) -> np.ndarray:
        """Returns cumulative scores normalized to [0, 1] range for knapsack utility."""
        if self.cumulative_scores is None or len(self.cumulative_scores) == 0:
            return np.array([])
        max_val = np.max(self.cumulative_scores)
        return (self.cumulative_scores / max_val) if max_val > 0 else self.cumulative_scores
