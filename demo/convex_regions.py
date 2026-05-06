"""
convex_regions.py - Convex region representation and operations.

Handles:
- Polytope representation (V-rep and H-rep)
- Intersection computation
- Membership checking with safety margin
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import linprog
from shapely.geometry import Polygon

from graph_types import region_node_label
from problem_data import DEFAULT_PROBLEM


@dataclass
class ConvexRegion:
    """
    Represents a convex polytope in 2D.
    
    Supports both vertex representation (V-rep) and halfspace representation (H-rep).
    H-rep: {x : A @ x <= b}
    
    Attributes:
        vertices: Array of shape (n_vertices, 2) in CCW order
        A: Halfspace matrix of shape (n_constraints, 2)
        b: Halfspace vector of shape (n_constraints,)
        index: Region index in the graph
        label: Optional string label
    """
    vertices: np.ndarray
    A: np.ndarray = field(default=None, repr=False)
    b: np.ndarray = field(default=None, repr=False)
    index: int = -1
    label: str = ""
    
    def __post_init__(self):
        """Convert V-rep to H-rep if not provided."""
        self.vertices = np.array(self.vertices, dtype=np.float64)
        
        # Ensure vertices are in CCW order
        self._ensure_ccw()
        
        # Compute H-representation if not provided
        if self.A is None or self.b is None:
            self._compute_halfspaces()
    
    def _ensure_ccw(self):
        """Ensure vertices are in counter-clockwise order."""
        # Compute signed area
        n = len(self.vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i, 0] * self.vertices[j, 1]
            area -= self.vertices[j, 0] * self.vertices[i, 1]
        
        # If clockwise (negative area), reverse
        if area < 0:
            self.vertices = self.vertices[::-1]
    
    def _compute_halfspaces(self):
        """
        Compute H-representation from vertices.
        
        For a polygon with CCW vertices, each edge defines a halfspace
        with the interior on the left side.
        """
        n = len(self.vertices)
        A_list = []
        b_list = []
        
        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]
            
            # Edge direction
            edge = p2 - p1
            
            # Outward normal (perpendicular to edge, pointing right for CCW)
            # For interior on left: normal points outward
            normal = np.array([edge[1], -edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            
            # Halfspace: normal * (x - p1) <= 0
            # => normal * x <= normal * p1
            A_list.append(normal)
            b_list.append(np.dot(normal, p1))
        
        self.A = np.array(A_list)
        self.b = np.array(b_list)
    
    def contains(self, point: np.ndarray,
                 margin: float = 0.0,
                 tol: float = 1e-7) -> bool:
        """
        Check if point is inside the region.
        
        Args:
            point: 2D point [x, y]
            margin: Safety margin (positive = shrink region)
            tol: Numerical slack for points that lie on the boundary up to
                floating-point error
            
        Returns:
            True if A @ point <= b - margin + tol
        """
        point = np.asarray(point).flatten()[:2]
        return np.all(self.A @ point <= self.b - margin + tol)
    
    def contains_batch(self, points: np.ndarray,
                       margin: float = 0.0,
                       tol: float = 1e-7) -> np.ndarray:
        """
        Check containment for multiple points.
        
        Args:
            points: Array of shape (N, 2)
            margin: Safety margin
            tol: Numerical slack for points near the boundary
            
        Returns:
            Boolean array of shape (N,)
        """
        # A @ points.T has shape (n_constraints, N)
        violations = self.A @ points.T - (self.b - margin).reshape(-1, 1)
        return np.all(violations <= tol, axis=0)
    
    def get_centroid(self) -> np.ndarray:
        """Compute centroid of the polygon."""
        return np.mean(self.vertices, axis=0)
    
    def get_shapely_polygon(self) -> Polygon:
        """Convert to Shapely polygon."""
        return Polygon(self.vertices)
    
    def get_interior_point(self) -> np.ndarray:
        """
        Find a strictly interior point using LP.
        
        Solves: max s s.t. A @ x + s * 1 <= b
        """
        n_constraints = len(self.b)
        
        # Variables: [x, y, s]
        c = np.array([0, 0, -1])  # maximize s
        
        # Constraints: A @ [x,y] + s <= b
        A_ub = np.hstack([self.A, np.ones((n_constraints, 1))])
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=self.b, bounds=[(-100, 100), (-100, 100), (None, None)])
            if result.success and result.x[2] > 0:
                return result.x[:2]
        except Exception:
            pass
        
        # Fallback to centroid
        return self.get_centroid()


def _supports_region_transition(geometry,
                                min_area: float = 1e-10,
                                min_shared_length: float = 1e-10) -> bool:
    """
    Return True when an intersection has a traversable interface.

    We accept:
    - positive-area overlap
    - shared boundary segment with positive length

    We reject:
    - isolated point contacts
    """
    if geometry.is_empty:
        return False

    geom_type = geometry.geom_type

    if geom_type == 'Polygon':
        return geometry.area >= min_area

    if geom_type == 'LineString':
        return geometry.length >= min_shared_length

    if hasattr(geometry, 'geoms'):
        return any(
            _supports_region_transition(part, min_area, min_shared_length)
            for part in geometry.geoms
        )

    return False


def compute_intersection(region1: ConvexRegion, region2: ConvexRegion) -> Optional[ConvexRegion]:
    """
    Compute intersection of two convex regions.
    
    For adjacent regions that share a boundary segment, returns a thin
    "virtual" intersection region around the shared edge.
    
    Args:
        region1, region2: ConvexRegion objects
        
    Returns:
        New ConvexRegion representing intersection, or None if disjoint or
        only touching at a single point
    """
    poly1 = region1.get_shapely_polygon()
    poly2 = region2.get_shapely_polygon()
    
    intersection = poly1.intersection(poly2)
    
    if not _supports_region_transition(intersection):
        return None
    
    # For proper area intersection
    if intersection.geom_type == 'Polygon' and intersection.area > 1e-10:
        vertices = np.array(intersection.exterior.coords[:-1])
        A = np.vstack([region1.A, region2.A])
        b = np.hstack([region1.b, region2.b])
        return ConvexRegion(vertices=vertices, A=A, b=b)
    
    # For shared edges, create a thin interface region around the boundary.
    if intersection.geom_type in ['LineString', 'MultiLineString', 'GeometryCollection']:
        # Buffer the intersection slightly to create a valid region
        buffer_eps = 0.01
        buffered = intersection.buffer(buffer_eps, cap_style='square')
        
        # Clip to small neighborhoods of both regions so the interface stays
        # localized near the shared boundary segment.
        clipped = (
            buffered
            .intersection(poly1.buffer(buffer_eps))
            .intersection(poly2.buffer(buffer_eps))
        )
        
        if clipped.is_empty or clipped.area < 1e-10:
            # Fall back to a tiny square around the shared-edge centroid.
            centroid = intersection.centroid
            eps = buffer_eps
            vertices = np.array([
                [centroid.x - eps, centroid.y - eps],
                [centroid.x + eps, centroid.y - eps],
                [centroid.x + eps, centroid.y + eps],
                [centroid.x - eps, centroid.y + eps]
            ])
            # Use combined H-rep
            A = np.vstack([region1.A, region2.A])
            b = np.hstack([region1.b, region2.b])
            return ConvexRegion(vertices=vertices, A=A, b=b)
        
        if clipped.geom_type == 'Polygon':
            vertices = np.array(clipped.exterior.coords[:-1])
            A = np.vstack([region1.A, region2.A])
            b = np.hstack([region1.b, region2.b])
            return ConvexRegion(vertices=vertices, A=A, b=b)
    
    return None


def regions_intersect(region1: ConvexRegion, region2: ConvexRegion, 
                      min_area: float = 1e-6,
                      min_shared_length: float = 1e-6) -> bool:
    """
    Check if two regions overlap or share a boundary segment.
    
    Regions that only touch at a single point are NOT considered adjacent.
    
    Args:
        region1, region2: Regions to check
        min_area: Minimum overlap area to consider non-empty
        min_shared_length: Minimum shared boundary length to consider adjacent
        
    Returns:
        True if regions have area overlap or share a boundary segment
    """
    poly1 = region1.get_shapely_polygon()
    poly2 = region2.get_shapely_polygon()
    
    if not poly1.intersects(poly2):
        return False

    intersection = poly1.intersection(poly2)
    return _supports_region_transition(
        intersection,
        min_area=min_area,
        min_shared_length=min_shared_length,
    )


def get_intersection_halfspaces(region1: ConvexRegion, region2: ConvexRegion) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get combined H-representation for intersection Q1 intersect Q2.
    
    Returns:
        (A, b) where intersection = {x : A @ x <= b}
    """
    A = np.vstack([region1.A, region2.A])
    b = np.hstack([region1.b, region2.b])
    return A, b


def create_regions_from_vertices_list(vertices_list: List[np.ndarray]) -> List[ConvexRegion]:
    """
    Create ConvexRegion objects from list of vertex arrays.
    
    Args:
        vertices_list: List of arrays, each of shape (n_v, 2)
        
    Returns:
        List of ConvexRegion objects
    """
    regions = []
    for i, vertices in enumerate(vertices_list):
        region = ConvexRegion(vertices=vertices, index=i, label=region_node_label(i))
        regions.append(region)
    return regions


def create_buffered_regions_from_vertices_list(
    vertices_list: List[np.ndarray],
    workspace_vertices: np.ndarray,
    buffer_size: float
) -> List[ConvexRegion]:
    """
    Expand region polygons by a small buffer and clip them to the workspace.
    """
    workspace = Polygon(np.asarray(workspace_vertices, dtype=np.float64))
    expanded_list = []
    
    for verts in vertices_list:
        poly = Polygon(np.asarray(verts, dtype=np.float64))
        buffered = poly.buffer(buffer_size, join_style='mitre')
        clipped = buffered.intersection(workspace)
        if clipped.geom_type == 'Polygon':
            expanded_list.append(np.array(clipped.exterior.coords[:-1]))
        else:
            expanded_list.append(np.asarray(verts, dtype=np.float64))
    
    return create_regions_from_vertices_list(expanded_list)


def get_default_regions() -> List[ConvexRegion]:
    """
    Get the default safe convex regions from problem specification.
    
    Note: Original regions only touch at boundaries. For unicycle dynamics
    to be feasible, we expand each region slightly to create overlap.
    This is a common practice in motion planning to ensure dynamic feasibility.
    """
    original_vertices_list = [
        np.asarray(vertices, dtype=np.float64)
        for vertices in DEFAULT_PROBLEM.region_vertices
    ]
    return create_buffered_regions_from_vertices_list(
        original_vertices_list,
        np.asarray(DEFAULT_PROBLEM.workspace_vertices, dtype=np.float64),
        DEFAULT_PROBLEM.region_buffer
    )


def get_original_regions() -> List[ConvexRegion]:
    """
    Get the original (touching only) safe convex regions.
    These may not be dynamically feasible for unicycle model.
    """
    vertices_list = [
        np.asarray(vertices, dtype=np.float64)
        for vertices in DEFAULT_PROBLEM.region_vertices
    ]
    return create_regions_from_vertices_list(vertices_list)
