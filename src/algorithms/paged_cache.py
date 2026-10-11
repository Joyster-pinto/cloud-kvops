"""
LLM-KVOps: Advanced Algorithm 3 - PagedAttention Virtual Memory Block Manager
Paper Reference: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP / vLLM)

Partitions KV-cache tensors into fixed-size physical blocks (e.g., 16 tokens per block).
Uses a page/block table to map logical token indices to physical blocks, eliminating
both external memory fragmentation (100%) and internal fragmentation (near 0%).
"""

from typing import List, Dict, Tuple, Optional, Set
import math
import numpy as np


class PhysicalBlock:
    """Represents one contiguous physical memory page holding block_size token slots."""
    def __init__(self, block_id: int, block_size: int = 16, num_heads: int = 8, head_dim: int = 128):
        self.block_id = block_id
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ref_count = 0  # Support Copy-on-Write / multi-tenant sharing
        self.occupied_slots = 0
        # Keys and Values pre-allocated as (block_size, num_heads, head_dim)
        self.key_buffer = np.zeros((block_size, num_heads, head_dim), dtype=np.float16)
        self.value_buffer = np.zeros((block_size, num_heads, head_dim), dtype=np.float16)

    def is_full(self) -> bool:
        return self.occupied_slots >= self.block_size

    def append_token(self, key_vector: np.ndarray, value_vector: np.ndarray) -> int:
        """Appends one token KV vector into the block. Returns local slot offset."""
        if self.is_full():
            raise IndexError("Cannot append to a full PhysicalBlock.")
        offset = self.occupied_slots
        self.key_buffer[offset] = key_vector
        self.value_buffer[offset] = value_vector
        self.occupied_slots += 1
        return offset

    def reset(self):
        self.occupied_slots = 0
        self.ref_count = 0


class PagedKVCacheManager:
    """
    Manages non-contiguous virtual memory allocation for multi-tenant LLM inference.
    Maps logical token sequences to physical memory blocks via Block Tables.
    """
    def __init__(
        self,
        total_blocks: int = 128,
        block_size: int = 16,
        num_heads: int = 8,
        head_dim: int = 128
    ):
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_blocks = total_blocks

        # Pre-allocate physical block pool
        self.physical_blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i, block_size=block_size, num_heads=num_heads, head_dim=head_dim)
            for i in range(total_blocks)
        ]
        self.free_block_ids: Set[int] = set(range(total_blocks))

        # Logical session block tables: session_id -> list of physical_block_ids
        self.block_tables: Dict[str, List[int]] = {}

    def allocate_block(self) -> int:
        """Allocates one physical block from the free pool."""
        if not self.free_block_ids:
            raise MemoryError("PagedKVCacheManager: Out of Physical Blocks (GPU VRAM Full).")
        block_id = self.free_block_ids.pop()
        self.physical_blocks[block_id].ref_count = 1
        return block_id

    def free_block(self, block_id: int):
        """Frees a block or decrements its reference counter."""
        block = self.physical_blocks[block_id]
        block.ref_count -= 1
        if block.ref_count <= 0:
            block.reset()
            self.free_block_ids.add(block_id)

    def append_token_to_session(
        self,
        session_id: str,
        key_vector: np.ndarray,
        value_vector: np.ndarray
    ) -> Tuple[int, int]:
        """
        Appends a newly decoded token to a session's logical sequence.
        Allocates a new physical block on-demand if the current block is full.

        Returns:
            (physical_block_id, slot_offset)
        """
        if session_id not in self.block_tables:
            self.block_tables[session_id] = []

        table = self.block_tables[session_id]

        if not table or self.physical_blocks[table[-1]].is_full():
            new_block_id = self.allocate_block()
            table.append(new_block_id)

        target_block = self.physical_blocks[table[-1]]
        slot_offset = target_block.append_token(key_vector, value_vector)
        return target_block.block_id, slot_offset

    def translate_logical_index(self, session_id: str, logical_idx: int) -> Tuple[int, int]:
        """
        Translates logical token index (0, 1, 2, ...) to (physical_block_id, block_offset).
        Complexity: O(1).
        """
        table = self.block_tables.get(session_id, [])
        block_idx = logical_idx // self.block_size
        slot_offset = logical_idx % self.block_size

        if block_idx >= len(table):
            raise IndexError(f"Logical index {logical_idx} out of allocated bounds.")

        return table[block_idx], slot_offset

    def free_session(self, session_id: str):
        """Releases all physical memory blocks allocated to a session."""
        if session_id in self.block_tables:
            for b_id in self.block_tables[session_id]:
                self.free_block(b_id)
            del self.block_tables[session_id]

    def compute_fragmentation_metrics(self, session_id: str) -> Dict[str, float]:
        """
        Compares Paged memory allocation vs standard contiguous over-allocation:
        Standard contiguous allocators typically allocate memory in powers of 2 (512, 1024, 2048),
        causing 30% to 60% internal fragmentation. Paged memory restricts waste strictly
        to the unused slots in the final block.
        """
        table = self.block_tables.get(session_id, [])
        if not table:
            return {"total_tokens": 0, "allocated_slots": 0, "wasted_slots": 0, "fragmentation_pct": 0.0}

        num_blocks = len(table)
        allocated_slots = num_blocks * self.block_size

        # Total tokens stored
        total_tokens = sum(self.physical_blocks[b_id].occupied_slots for b_id in table)
        wasted_slots = allocated_slots - total_tokens
        paged_waste_pct = (wasted_slots / allocated_slots) * 100.0 if allocated_slots > 0 else 0.0

        # Contiguous allocator worst-case (next power-of-2 allocation)
        next_pow2 = 2 ** int(math.ceil(math.log2(max(1, total_tokens))))
        contiguous_waste_slots = max(0, next_pow2 - total_tokens)
        contiguous_waste_pct = (contiguous_waste_slots / next_pow2) * 100.0 if next_pow2 > 0 else 0.0

        return {
            "total_tokens": total_tokens,
            "allocated_blocks": num_blocks,
            "paged_allocated_slots": allocated_slots,
            "paged_wasted_slots": wasted_slots,
            "paged_fragmentation_pct": paged_waste_pct,
            "contiguous_hypothetical_waste_pct": contiguous_waste_pct,
            "fragmentation_eliminated_pct": max(0.0, contiguous_waste_pct - paged_waste_pct)
        }
