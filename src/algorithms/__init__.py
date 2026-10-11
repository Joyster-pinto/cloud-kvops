"""
LLM-KVOps: Algorithmic Core Package
Includes both syllabus combinatorial algorithms and cutting-edge 2024 KV cache optimizers.
"""

from .rabin_karp import RabinKarpHasher, PrefixCacheTrie
from .sweep_line import BentleyOttmannLineSweep, TokenInterval
from .convex_hull import GrahamsScanConvexHull, GeometricEntityMerger, Point2D
from .knapsack_dp import KnapsackTokenEvictor
from .layer_allocator import LayerBudgetAllocator
from .h2o_accumulator import H2OAttentionAccumulator
from .kv_quantizer import KVIQuantizer, QuantizedTensor
from .paged_cache import PagedKVCacheManager, PhysicalBlock

__all__ = [
    "RabinKarpHasher",
    "PrefixCacheTrie",
    "BentleyOttmannLineSweep",
    "TokenInterval",
    "GrahamsScanConvexHull",
    "GeometricEntityMerger",
    "Point2D",
    "KnapsackTokenEvictor",
    "LayerBudgetAllocator",
    "H2OAttentionAccumulator",
    "KVIQuantizer",
    "QuantizedTensor",
    "PagedKVCacheManager",
    "PhysicalBlock",
]
