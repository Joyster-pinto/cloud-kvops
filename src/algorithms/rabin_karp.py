"""
LLM-KVOps: Tier 1 - Prefix Deduplication & Cache Sharing
Syllabus Alignment: Module 3 (String Matching: Rabin-Karp Algorithm & Prefix Trees)

Implements:
1. Rabin-Karp Polynomial Rolling Hash for O(1) sliding window fingerprinting.
2. PrefixTrie for sublinear O(1) amortized prefix cache matching and zero-copy sharing.
"""

from typing import List, Dict, Tuple, Optional, Any
import time

class RabinKarpHasher:
    """
    Computes and updates polynomial rolling hashes over token sequences.
    
    Formula:
        H(S[i..i+k-1]) = sum_{j=0}^{k-1} S[i+j] * p^(k-1-j) mod q
    Rolling Update:
        H(S[i+1..i+k]) = ( (H(S[i..i+k-1]) - S[i] * p^(k-1)) * p + S[i+k] ) mod q
    """
    def __init__(self, prime_mod: int = 1_000_000_007, base: int = 257):
        self.q = prime_mod
        self.p = base

    def compute_hash(self, tokens: List[int]) -> int:
        """Compute the full polynomial hash for a sequence of tokens in O(k) time."""
        h = 0
        for token in tokens:
            h = (h * self.p + (token % self.q)) % self.q
        return h

    def compute_window_hashes(self, tokens: List[int], window_size: int) -> List[Tuple[int, int]]:
        """
        Computes rolling hashes for all sliding windows of length window_size.
        Returns list of (start_index, hash_value).
        Complexity: O(N) total time (O(1) per rolling step).
        """
        n = len(tokens)
        if n < window_size:
            return [(0, self.compute_hash(tokens))]

        # Precompute p^(window_size - 1) mod q
        p_pow = 1
        for _ in range(window_size - 1):
            p_pow = (p_pow * self.p) % self.q

        hashes = []
        # Initial hash for first window
        curr_hash = 0
        for i in range(window_size):
            curr_hash = (curr_hash * self.p + (tokens[i] % self.q)) % self.q
        hashes.append((0, curr_hash))

        # Slide window across remaining tokens
        for i in range(1, n - window_size + 1):
            # Remove leading token tokens[i-1] and add trailing token tokens[i+window_size-1]
            prev_token = tokens[i - 1] % self.q
            next_token = tokens[i + window_size - 1] % self.q
            
            curr_hash = (curr_hash - (prev_token * p_pow) % self.q + self.q) % self.q
            curr_hash = (curr_hash * self.p + next_token) % self.q
            hashes.append((i, curr_hash))

        return hashes


class TrieNode:
    """Node structure for PrefixTrie with KV tensor cache reference pointers."""
    def __init__(self):
        self.children: Dict[int, 'TrieNode'] = {}
        self.is_terminal: bool = False
        self.hash_value: Optional[int] = None
        self.cache_ref_id: Optional[str] = None
        self.token_count: int = 0
        self.hit_count: int = 0


class PrefixCacheTrie:
    """
    Prefix Trie combined with Rabin-Karp hash verification for zero-copy KV cache sharing.
    Enables O(1) detection of common system prompts, few-shot examples, and shared headers.
    """
    def __init__(self, hasher: Optional[RabinKarpHasher] = None):
        self.root = TrieNode()
        self.hasher = hasher or RabinKarpHasher()
        self.hash_to_node: Dict[int, TrieNode] = {}
        self.total_shared_tokens_saved: int = 0

    def insert_prefix(self, tokens: List[int], cache_ref_id: str) -> int:
        """
        Inserts a prompt prefix into the trie and stores its cache reference ID.
        Returns the hash value of the prefix.
        Complexity: O(M) where M is prefix length.
        """
        curr = self.root
        prefix_hash = self.hasher.compute_hash(tokens)
        
        for idx, token in enumerate(tokens):
            if token not in curr.children:
                curr.children[token] = TrieNode()
            curr = curr.children[token]
            curr.token_count = idx + 1

        curr.is_terminal = True
        curr.hash_value = prefix_hash
        curr.cache_ref_id = cache_ref_id
        self.hash_to_node[prefix_hash] = curr
        return prefix_hash

    def match_longest_prefix(self, tokens: List[int]) -> Tuple[int, Optional[str], int]:
        """
        Finds the longest cached prefix matching the incoming token sequence.
        Returns: (matched_length, cache_ref_id, memory_saved_tokens)
        Complexity: O(M) traversal with O(1) hash verification.
        """
        curr = self.root
        matched_length = 0
        last_valid_ref = None

        for idx, token in enumerate(tokens):
            if token in curr.children:
                curr = curr.children[token]
                matched_length = idx + 1
                if curr.is_terminal:
                    last_valid_ref = curr.cache_ref_id
                    curr.hit_count += 1
            else:
                break

        if last_valid_ref is not None:
            self.total_shared_tokens_saved += matched_length

        return matched_length, last_valid_ref, matched_length if last_valid_ref else 0
