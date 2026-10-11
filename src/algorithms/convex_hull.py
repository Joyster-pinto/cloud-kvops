"""
LLM-KVOps: Tier 2 - Graham's Scan Convex Hull & Centroid Vector Merging
Syllabus Alignment: Module 5 (Convex Hull Finding Algorithms: Graham's Scan)

Implements:
1. Graham's Scan Algorithm to compute the 2D Convex Hull of multi-token semantic clusters in O(M log M).
2. Centroid Vector Summarizer that compresses M contiguous/co-active token KV slots into 1 summary slot.
"""

from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np

class Point2D:
    """Represents a 2D point in the token attention/embedding projection space."""
    def __init__(self, x: float, y: float, token_id: Optional[int] = None):
        self.x = x
        self.y = y
        self.token_id = token_id

    def distance_to(self, other: 'Point2D') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self) -> str:
        return f"Point2D({self.x:.3f}, {self.y:.3f}, id={self.token_id})"


class GrahamsScanConvexHull:
    """
    Graham's Scan Algorithm for finding the 2D Convex Hull.
    Time Complexity: O(M log M) where M is the number of points in the cluster.
    """
    @staticmethod
    def cross_product_orientation(p: Point2D, q: Point2D, r: Point2D) -> float:
        """
        Returns orientation of triplet (p, q, r):
         > 0 -> Counter-Clockwise (Left turn)
         < 0 -> Clockwise (Right turn)
         = 0 -> Collinear
        """
        return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)

    def compute_hull(self, points: List[Point2D]) -> List[Point2D]:
        """
        Computes the Convex Hull of a set of 2D points using Graham's Scan.
        Returns ordered list of vertices on the convex boundary.
        """
        n = len(points)
        if n <= 2:
            return list(points)

        # 1. Find the bottom-most point (or leftmost if tie) -> Pivot P0
        pivot = min(points, key=lambda p: (p.y, p.x))

        # 2. Sort points by polar angle with pivot P0. If angles match, sort by distance.
        def polar_angle_and_dist(p: Point2D):
            angle = math.atan2(p.y - pivot.y, p.x - pivot.x)
            dist = pivot.distance_to(p)
            return (angle, dist)

        other_points = [p for p in points if p is not pivot]
        sorted_points = sorted(other_points, key=polar_angle_and_dist)

        # Filter out collinear points with same angle, keeping only the farthest
        unique_points = []
        for i, p in enumerate(sorted_points):
            if i < len(sorted_points) - 1:
                next_p = sorted_points[i + 1]
                if abs(math.atan2(p.y - pivot.y, p.x - pivot.x) - 
                       math.atan2(next_p.y - pivot.y, next_p.x - pivot.x)) < 1e-9:
                    continue  # skip shorter distance
            unique_points.append(p)

        if len(unique_points) < 2:
            return [pivot] + unique_points

        # 3. Process points using a Stack
        hull_stack: List[Point2D] = [pivot, unique_points[0], unique_points[1]]

        for i in range(2, len(unique_points)):
            curr = unique_points[i]
            # While stack has at least 2 elements and the turn is not strictly counter-clockwise
            while len(hull_stack) >= 2 and self.cross_product_orientation(hull_stack[-2], hull_stack[-1], curr) <= 0:
                hull_stack.pop()
            hull_stack.append(curr)

        return hull_stack


class GeometricEntityMerger:
    """
    Compresses multi-token entity clusters into a single Centroid KV vector,
    computing the convex boundary and memory savings.
    """
    def __init__(self):
        self.convex_hull_solver = GrahamsScanConvexHull()

    def merge_token_cluster(
        self,
        token_indices: List[int],
        kv_tensors: np.ndarray,
        token_texts: List[str]
    ) -> Dict[str, Any]:
        """
        Takes a cluster of M token indices and their corresponding KV embeddings (M x D).
        1. Projects high-dimensional KV embeddings to 2D for geometric boundary analysis.
        2. Computes the Graham's Scan Convex Hull.
        3. Computes the geometric centroid vector (mean KV vector).
        
        Returns a dictionary with merged summary vector and memory savings metadata.
        """
        m = len(token_indices)
        if m == 0:
            return {}

        cluster_kvs = kv_tensors[token_indices]  # Shape: (M, D)
        d = cluster_kvs.shape[1]

        # Compute 2D projection for visualization (using first 2 principal directions or normalized slices)
        if d >= 2:
            points_2d = [
                Point2D(float(cluster_kvs[i, 0]), float(cluster_kvs[i, 1]), token_id=token_indices[i])
                for i in range(m)
            ]
        else:
            points_2d = [Point2D(float(i), float(cluster_kvs[i, 0]), token_id=token_indices[i]) for i in range(m)]

        # Compute Graham's Scan Convex Hull
        hull_vertices = self.convex_hull_solver.compute_hull(points_2d)

        # Compute Centroid Vector across all D dimensions
        centroid_vector = np.mean(cluster_kvs, axis=0, keepdims=True)  # Shape: (1, D)

        cluster_text = " ".join([token_texts[idx] for idx in token_indices if idx < len(token_texts)])

        return {
            "original_token_indices": token_indices,
            "cluster_text": cluster_text,
            "original_count": m,
            "compressed_count": 1,
            "memory_saved_slots": m - 1,
            "compression_ratio": (m - 1) / m,
            "centroid_vector": centroid_vector,
            "hull_vertex_count": len(hull_vertices),
            "hull_vertices": [(p.x, p.y) for p in hull_vertices]
        }
