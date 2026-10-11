"""
LLM-KVOps: Tier 4 - Greedy & Branch-and-Bound Layer-Wise Budget Allocation
Syllabus Alignment: Module 1 (Greedy Techniques: Fractional Knapsack) & Module 2 (Branch & Bound)

Implements:
1. Greedy Fractional Knapsack allocation based on layer-wise attention entropy density:
     rho_l = Delta_Attention(l) / Memory_Cost(l)
2. Pyramidal information funneling ensuring middle reasoning layers receive higher capacity.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

class LayerBudgetAllocator:
    """
    Distributes total VRAM token capacity across L transformer layers non-uniformly.
    """
    def __init__(self, num_layers: int = 32):
        self.num_layers = num_layers

    def compute_layer_importance_profile(self, custom_entropies: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes layer-wise importance weights based on Transformer Pyramidal Funneling.
        - Early layers (0 to ~25%): Syntactic / local attention -> Moderate weight
        - Middle layers (~25% to ~75%): Complex semantic reasoning / recall -> Peak weight
        - Late layers (~75% to 100%): Output token generation -> Moderate-low weight
        """
        if custom_entropies is not None and len(custom_entropies) == self.num_layers:
            # Normalize user-supplied attention entropies
            return custom_entropies / np.sum(custom_entropies)

        # Generate Gaussian-Pyramidal bell curve centered at layer ~45% depth
        layer_indices = np.arange(self.num_layers)
        center = 0.45 * (self.num_layers - 1)
        sigma = 0.25 * self.num_layers
        profile = np.exp(-((layer_indices - center) ** 2) / (2 * (sigma ** 2))) + 0.25
        normalized_profile = profile / np.sum(profile)
        return normalized_profile

    def allocate_greedy_budgets(
        self,
        total_token_budget: int,
        layer_importance: Optional[np.ndarray] = None,
        min_budget_per_layer: int = 16
    ) -> List[int]:
        """
        Solves layer budget allocation using the Greedy Fractional Knapsack paradigm.
        
        Args:
            total_token_budget: Global KV slot capacity across all layers (e.g., L * B_avg)
            layer_importance: 1D array of layer utility densities
            min_budget_per_layer: Minimum safety token floor per layer
            
        Returns:
            layer_budgets: List of integer token capacities allocated to each layer
        """
        L = self.num_layers
        if layer_importance is None:
            importance = self.compute_layer_importance_profile()
        else:
            importance = layer_importance / np.sum(layer_importance)

        # 1. Guarantee minimum baseline budget to all layers
        guaranteed_total = min_budget_per_layer * L
        if total_token_budget <= guaranteed_total:
            # Equal distribution if budget is extremely tight
            equal_quota = max(1, total_token_budget // L)
            return [equal_quota] * L

        remaining_budget = total_token_budget - guaranteed_total

        # 2. Greedy allocation: allocate remaining budget proportional to utility density rho_l
        additional_allocations = np.floor(importance * remaining_budget).astype(int)
        
        # Distribute remaining fractional rounding remainder to highest density layers
        remainder = remaining_budget - np.sum(additional_allocations)
        if remainder > 0:
            sorted_by_density = np.argsort(-importance)
            for i in range(remainder):
                layer_idx = sorted_by_density[i % L]
                additional_allocations[layer_idx] += 1

        final_budgets = (min_budget_per_layer + additional_allocations).tolist()
        return final_budgets
