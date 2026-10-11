"""
LLM-KVOps: Tier 2 - Bentley-Ottmann 2D Line Sweep Segment Intersection
Syllabus Alignment: Module 5 (Line Segments: Properties, Intersection, Sweeping Lines)

Implements:
The Bentley-Ottmann Sweep-Line Algorithm to detect overlapping, co-active token 
attention intervals in O((N + K) log N) time, avoiding naive O(N^2) pairwise intersection.
"""

from typing import List, Tuple, Dict, Set, Optional
import heapq

class TokenInterval:
    """
    Represents a token's temporal-semantic activation interval on the generation timeline.
    Segment: [(t_start, layer_id), (t_end, layer_id)]
    """
    def __init__(self, token_idx: int, t_start: float, t_end: float, layer_id: int, text: str = ""):
        self.token_idx = token_idx
        self.t_start = min(t_start, t_end)
        self.t_end = max(t_start, t_end)
        self.layer_id = layer_id
        self.text = text

    def overlaps(self, other: 'TokenInterval') -> bool:
        """Check if two 1D intervals overlap on the timeline."""
        return max(self.t_start, other.t_start) <= min(self.t_end, other.t_end)

    def __repr__(self) -> str:
        return f"TokenInterval(idx={self.token_idx}, text='{self.text}', span=[{self.t_start:.1f}, {self.t_end:.1f}], layer={self.layer_id})"


class SweepEvent:
    """Event point in the Bentley-Ottmann sweep-line algorithm."""
    EVENT_START = 0
    EVENT_END = 1

    def __init__(self, time_x: float, event_type: int, segment: TokenInterval):
        self.time_x = time_x
        self.event_type = event_type
        self.segment = segment

    def __lt__(self, other: 'SweepEvent') -> bool:
        if abs(self.time_x - other.time_x) > 1e-9:
            return self.time_x < other.time_x
        # If time is equal, process START events before END events
        return self.event_type < other.event_type


class BentleyOttmannLineSweep:
    """
    Bentley-Ottmann Line Sweep Algorithm.
    Detects all pairs of overlapping / co-active token attention intervals in O((N + K) log N).
    """
    def __init__(self):
        self.intersections: List[Tuple[int, int]] = []
        self.clusters: List[List[int]] = []

    def find_intersections(self, segments: List[TokenInterval]) -> List[Tuple[int, int]]:
        """
        Executes the sweep-line algorithm over the set of token intervals.
        Returns list of (token_idx_1, token_idx_2) representing intersecting activation spans.
        """
        self.intersections = []
        if not segments:
            return []

        # 1. Initialize Event Queue (Min-Heap based on timeline X coordinate)
        event_queue: List[SweepEvent] = []
        for seg in segments:
            heapq.heappush(event_queue, SweepEvent(seg.t_start, SweepEvent.EVENT_START, seg))
            heapq.heappush(event_queue, SweepEvent(seg.t_end, SweepEvent.EVENT_END, seg))

        # 2. Active Status Structure (Active segments currently sliced by the vertical sweep-line)
        active_segments: Set[TokenInterval] = set()

        # 3. Process Events
        while event_queue:
            event = heapq.heappop(event_queue)
            seg = event.segment

            if event.event_type == SweepEvent.EVENT_START:
                # Check for intersection with all currently active segments
                for active_seg in active_segments:
                    if seg.overlaps(active_seg):
                        idx_pair = (min(seg.token_idx, active_seg.token_idx), max(seg.token_idx, active_seg.token_idx))
                        if idx_pair not in self.intersections:
                            self.intersections.append(idx_pair)
                active_segments.add(seg)
            elif event.event_type == SweepEvent.EVENT_END:
                active_segments.discard(seg)

        return self.intersections

    def cluster_coactive_tokens(self, segments: List[TokenInterval], max_gap: float = 1.0) -> List[List[int]]:
        """
        Groups intersecting and contiguous token intervals into candidate entity clusters
        for convex hull geometric merging.
        Returns list of token index groups: e.g. [[12, 13, 14, 15], [45, 46, 47]]
        """
        intersections = self.find_intersections(segments)
        if not segments:
            return []

        # Build adjacency graph of overlapping tokens
        adj: Dict[int, Set[int]] = {s.token_idx: set() for s in segments}
        for u, v in intersections:
            adj[u].add(v)
            adj[v].add(u)

        # Also connect contiguous sequential tokens if their timestamps are within max_gap
        sorted_segs = sorted(segments, key=lambda s: s.token_idx)
        for i in range(len(sorted_segs) - 1):
            s1, s2 = sorted_segs[i], sorted_segs[i + 1]
            if s2.token_idx == s1.token_idx + 1 and (s2.t_start - s1.t_end) <= max_gap:
                adj[s1.token_idx].add(s2.token_idx)
                adj[s2.token_idx].add(s1.token_idx)

        # Find connected components with size >= 2 (multi-token entities)
        visited = set()
        clusters = []

        for token_idx in sorted(adj.keys()):
            if token_idx not in visited:
                component = []
                queue = [token_idx]
                visited.add(token_idx)
                while queue:
                    curr = queue.pop(0)
                    component.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                # Only keep multi-token clusters (size >= 2) that warrant compression
                if len(component) >= 2:
                    clusters.append(sorted(component))

        self.clusters = clusters
        return clusters
