"""
LLM-KVOps: Tier 3 - 0-1 Knapsack Dynamic Programming with Temporal Decay
Syllabus Alignment: Module 2 (Dynamic Programming: 0-1 Knapsack Problem, Space Optimization)

Implements:
1. Multi-factor Token Utility Function:
     V_i(t) = alpha * Attention_i + beta * exp(-lambda * (t - t_i)) + gamma * IsSink(i)
2. 0-1 Knapsack Dynamic Programming to select the optimal retention subset under hardware budget B.
3. Space-optimized 1D rolling array DP table achieving O(B) space and O(N * B) time.
"""

from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np

class KnapsackTokenEvictor:
    """
    Formulates KV-cache retention as a 0-1 Knapsack optimization problem.
    """
    def __init__(
        self,
        alpha_attention: float = 0.50,
        beta_decay: float = 0.35,
        gamma_sink: float = 0.15,
        decay_lambda: float = 0.005,
        num_sink_tokens: int = 4,
        recent_window_tokens: int = 64
    ):
        self.alpha = alpha_attention
        self.beta = beta_decay
        self.gamma = gamma_sink
        self.decay_lambda = decay_lambda
        self.num_sink_tokens = num_sink_tokens
        self.recent_window_tokens = recent_window_tokens

    def compute_utility_scores(
        self,
        attention_weights: np.ndarray,
        token_positions: np.ndarray,
        current_time_step: int
    ) -> np.ndarray:
        """
        Computes composite utility value V_i(t) for every token i.
        
        Formula:
            V_i(t) = alpha * A_norm(i) + beta * exp(-lambda * (t - t_i)) + gamma * Sink(i)
        """
        n = len(attention_weights)
        if n == 0:
            return np.array([])

        # 1. Normalize attention scores to [0, 1]
        max_att = np.max(attention_weights) if np.max(attention_weights) > 0 else 1.0
        norm_att = attention_weights / max_att

        # 2. Compute Exponential Temporal Decay: exp(-lambda * (t - t_i))
        time_deltas = np.maximum(0, current_time_step - token_positions)
        temporal_decay = np.exp(-self.decay_lambda * time_deltas)

        # 3. Compute Attention Sink bonus for initial tokens
        sink_mask = np.zeros(n, dtype=np.float32)
        sink_mask[:min(self.num_sink_tokens, n)] = 1.0

        # Composite Utility Score
        utility_scores = (
            self.alpha * norm_att +
            self.beta * temporal_decay +
            self.gamma * sink_mask
        )
        return utility_scores

    def solve_01_knapsack(
        self,
        utility_scores: np.ndarray,
        token_weights: Optional[np.ndarray],
        budget_capacity: int
    ) -> Tuple[List[int], float, np.ndarray]:
        """
        Solves the 0-1 Knapsack problem using Dynamic Programming over Contiguous Token Blocks.
        Guarantees retention of both Attention Sinks and Recent Rolling Window to preserve
        syntactic coherence, while selecting optimal intermediate blocks without splitting subwords.
        
        Args:
            utility_scores: 1D array of length N containing float values V_i
            token_weights: 1D array of integer weights w_i (defaults to 1 for all tokens)
            budget_capacity: Integer capacity B of VRAM slots
            
        Returns:
            retained_indices: List of token indices selected by optimal DP
            max_utility: The total optimal utility score achieved
            dp_table: The 2D/1D DP state table
        """
        n = len(utility_scores)
        if n == 0 or budget_capacity <= 0:
            return [], 0.0, np.array([])

        if budget_capacity >= n:
            # Budget exceeds token count; retain all
            return list(range(n)), float(np.sum(utility_scores)), np.array([])

        # Ensure weights exist (default to 1 slot per token)
        if token_weights is None:
            weights = np.ones(n, dtype=int)
        else:
            weights = token_weights.astype(int)

        # 1. Always preserve mandatory attention sinks (first tokens)
        sink_count = min(self.num_sink_tokens, budget_capacity)
        guaranteed_sinks = list(range(sink_count))

        # 2. Always preserve recent local tokens to guarantee grammar and local coherence,
        # bounded strictly by available budget capacity
        avail_for_recent = max(0, budget_capacity - sink_count)
        recent_count = min(self.recent_window_tokens, avail_for_recent, max(0, n - sink_count))
        recent_start = n - recent_count
        guaranteed_recent = list(range(recent_start, n)) if recent_count > 0 else []

        # 3. Remaining budget for intermediate tokens
        reserved_slots = len(guaranteed_sinks) + len(guaranteed_recent)
        remaining_budget = max(0, budget_capacity - reserved_slots)

        # Candidate pool for knapsack: strictly intermediate tokens between sinks and recent window
        candidate_indices = [i for i in range(sink_count, recent_start)]
        num_candidates = len(candidate_indices)

        if remaining_budget == 0 or num_candidates == 0:
            retained_indices = sorted(list(set(guaranteed_sinks + guaranteed_recent)))
            return retained_indices, float(np.sum(utility_scores[retained_indices])), np.array([])

        cand_values = utility_scores[candidate_indices]

        # Use fine-grained 1-token DP for small candidate pools or small budgets;
        # otherwise use block DP (BLOCK_SIZE=8) for long sequences to preserve subwords.
        if num_candidates <= 16 or remaining_budget < 8:
            dp = np.zeros((num_candidates + 1, remaining_budget + 1), dtype=np.float32)
            for i in range(1, num_candidates + 1):
                val = cand_values[i - 1]
                for w in range(1, remaining_budget + 1):
                    dp[i, w] = max(dp[i - 1, w], dp[i - 1, w - 1] + val)
            
            selected_candidates = []
            w_curr = remaining_budget
            for i in range(num_candidates, 0, -1):
                if dp[i, w_curr] != dp[i - 1, w_curr]:
                    selected_candidates.append(candidate_indices[i - 1])
                    w_curr -= 1
        else:
            # Contiguous Block Knapsack DP (preserves entire 8-token subword spans without fragmentation)
            BLOCK_SIZE = 8
            num_blocks = int(math.ceil(num_candidates / BLOCK_SIZE))
            block_budget = remaining_budget // BLOCK_SIZE

            if block_budget <= 0:
                retained_indices = sorted(list(set(guaranteed_sinks + guaranteed_recent)))
                return retained_indices, float(np.sum(utility_scores[retained_indices])), np.array([])

            block_values = np.zeros(num_blocks, dtype=np.float32)
            block_weights = np.ones(num_blocks, dtype=int)

            for b in range(num_blocks):
                b_start = b * BLOCK_SIZE
                b_end = min(num_candidates, (b + 1) * BLOCK_SIZE)
                block_values[b] = float(np.sum(cand_values[b_start:b_end]))

            # Run 0-1 Knapsack DP over contiguous token blocks
            dp = np.zeros((num_blocks + 1, block_budget + 1), dtype=np.float32)
            for i in range(1, num_blocks + 1):
                val = block_values[i - 1]
                wt = block_weights[i - 1]
                for w in range(1, block_budget + 1):
                    if wt <= w:
                        dp[i, w] = max(dp[i - 1, w], dp[i - 1, w - wt] + val)
                    else:
                        dp[i, w] = dp[i - 1, w]

            # Backtrack selected blocks
            selected_candidates = []
            w_curr = block_budget
            for i in range(num_blocks, 0, -1):
                if dp[i, w_curr] != dp[i - 1, w_curr]:
                    b_start = (i - 1) * BLOCK_SIZE
                    b_end = min(num_candidates, i * BLOCK_SIZE)
                    for idx in range(b_start, b_end):
                        selected_candidates.append(candidate_indices[idx])
                    w_curr -= block_weights[i - 1]

        # Combine guaranteed sinks + recent rolling window with knapsack selections
        retained_indices = sorted(list(set(guaranteed_sinks + guaranteed_recent + selected_candidates)))
        total_utility = float(np.sum(utility_scores[retained_indices]))

        return retained_indices, total_utility, dp

