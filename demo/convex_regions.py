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


def _chebyshev_center_radius(A: np.ndarray, b: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """Return (center, radius) of the max inscribed ball in {x: A@x <= b}.

    Returns None when the LP is infeasible or unbounded.
    """
    n = A.shape[1]
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    c_obj = np.zeros(n + 1)
    c_obj[-1] = -1.0
    A_ub = np.hstack([A, norms])
    bounds = [(None, None)] * n + [(0.0, None)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b, bounds=bounds, method='highs')
    if not res.success or res.x is None:
        return None
    return res.x[:n].copy(), float(res.x[n])


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

    if _supports_region_transition(intersection):
        # For proper area intersection
        if intersection.geom_type == 'Polygon' and intersection.area > 1e-10:
            vertices = np.array(intersection.exterior.coords[:-1])
            A = np.vstack([region1.A, region2.A])
            b = np.hstack([region1.b, region2.b])
            return ConvexRegion(vertices=vertices, A=A, b=b)

        # For shared edges, create a thin interface region around the boundary.
        if intersection.geom_type in ['LineString', 'MultiLineString', 'GeometryCollection']:
            buffer_eps = 0.01
            buffered = intersection.buffer(buffer_eps, cap_style='square')
            clipped = (
                buffered
                .intersection(poly1.buffer(buffer_eps))
                .intersection(poly2.buffer(buffer_eps))
            )

            if clipped.is_empty or clipped.area < 1e-10:
                centroid = intersection.centroid
                eps = buffer_eps
                vertices = np.array([
                    [centroid.x - eps, centroid.y - eps],
                    [centroid.x + eps, centroid.y - eps],
                    [centroid.x + eps, centroid.y + eps],
                    [centroid.x - eps, centroid.y + eps]
                ])
                A = np.vstack([region1.A, region2.A])
                b = np.hstack([region1.b, region2.b])
                return ConvexRegion(vertices=vertices, A=A, b=b)

            if clipped.geom_type == 'Polygon':
                vertices = np.array(clipped.exterior.coords[:-1])
                A = np.vstack([region1.A, region2.A])
                b = np.hstack([region1.b, region2.b])
                return ConvexRegion(vertices=vertices, A=A, b=b)

    # V-rep Shapely shows no traversable intersection. Fall back to H-rep LP:
    # _vrep_from_hrep can be conservative for tightened regions (complex obstacle
    # H-reps cause near-singular constraint pairs to be skipped), so the Shapely
    # polygon may under-represent the true feasible region. The H-rep Chebyshev
    # check is the authoritative feasibility test.
    A = np.vstack([region1.A, region2.A])
    b = np.hstack([region1.b, region2.b])
    result = _chebyshev_center_radius(A, b)
    if result is not None:
        center, rho = result
        if rho > 1e-6:
            vertices = _vrep_from_hrep(A, b)
            if vertices is None:
                eps = min(rho, 0.01)
                vertices = np.array([
                    [center[0] - eps, center[1] - eps],
                    [center[0] + eps, center[1] - eps],
                    [center[0] + eps, center[1] + eps],
                    [center[0] - eps, center[1] + eps],
                ])
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


def _vrep_from_hrep(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Reconstruct a bounded 2D polygon from normalized halfplanes."""
    points = []
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            matrix = np.vstack([A[i], A[j]])
            det = float(np.linalg.det(matrix))
            if abs(det) <= 1e-10:
                continue
            point = np.linalg.solve(matrix, np.array([b[i], b[j]]))
            if np.all(A @ point <= b + 1e-8):
                points.append(point)

    if len(points) < 3:
        return None

    hull = Polygon(np.asarray(points, dtype=np.float64)).convex_hull
    if hull.geom_type != 'Polygon' or hull.area <= 1e-10:
        return None
    return np.asarray(hull.exterior.coords[:-1], dtype=np.float64)


def _tighten_hrep_against_obstacles(
    region: ConvexRegion,
    original_polygon: Polygon,
    obstacle_polygons: List,
) -> ConvexRegion:
    """Clip a buffered convex region to the original cell's side of obstacle faces."""
    A_rows = [row.copy() for row in region.A]
    b_rows = [float(value) for value in region.b]
    original_vertices = np.asarray(
        original_polygon.exterior.coords[:-1], dtype=np.float64
    )

    for obstacle in obstacle_polygons:
        if not region.get_shapely_polygon().intersects(obstacle):
            continue
        coords = np.asarray(obstacle.exterior.coords[:-1], dtype=np.float64)
        if not obstacle.exterior.is_ccw:
            coords = coords[::-1]
        for i, p1 in enumerate(coords):
            p2 = coords[(i + 1) % len(coords)]
            edge = p2 - p1
            norm = float(np.linalg.norm(edge))
            if norm <= 1e-12:
                continue

            normal = np.array([edge[1], -edge[0]], dtype=np.float64) / norm
            offset = float(normal @ p1)

            # For a CCW obstacle, normal points outward and the obstacle lies
            # in normal @ x <= offset. Keep only a face whose exterior side
            # contains the complete original ACD2D cell.
            if np.min(original_vertices @ normal - offset) < -1e-8:
                continue

            free_normal = -normal
            free_offset = -offset
            if np.max(region.vertices @ free_normal - free_offset) > 1e-9:
                A_rows.append(free_normal)
                b_rows.append(free_offset)

    A = np.asarray(A_rows, dtype=np.float64)
    b = np.asarray(b_rows, dtype=np.float64)
    vertices = _vrep_from_hrep(A, b)
    if vertices is None:
        raise ValueError("Obstacle tightening produced an empty convex region")
    return ConvexRegion(vertices=vertices, A=A, b=b)


def create_buffered_regions_from_vertices_list(
    vertices_list: List[np.ndarray],
    workspace_vertices: np.ndarray,
    buffer_size: float,
    obstacle_polygons: Optional[List] = None,
) -> List[ConvexRegion]:
    """
    Expand each ACD2D region by buffer_size and tighten against obstacles.

    The outward mitre buffer and workspace clip follow the decomposition spec.
    Obstacle faces are then appended to the H-rep, preserving one convex graph
    region per ACD2D cell without taking a hull across obstacle space.
    """
    workspace = Polygon(np.asarray(workspace_vertices, dtype=np.float64))

    expanded_regions = []
    for index, verts in enumerate(vertices_list):
        poly = Polygon(np.asarray(verts, dtype=np.float64))
        buffered = poly.buffer(buffer_size, join_style='mitre')
        clipped = buffered.intersection(workspace)
        if clipped.geom_type != 'Polygon' or clipped.area <= 1e-10:
            expanded_regions.append(
                ConvexRegion(
                    vertices=np.asarray(verts, dtype=np.float64),
                    index=index,
                    label=region_node_label(index),
                )
            )
            continue

        region = ConvexRegion(
            vertices=np.asarray(clipped.exterior.coords[:-1], dtype=np.float64),
            index=index,
            label=region_node_label(index),
        )
        if obstacle_polygons:
            region = _tighten_hrep_against_obstacles(
                region, poly, obstacle_polygons
            )
            region.index = index
            region.label = region_node_label(index)
        expanded_regions.append(region)

    return expanded_regions


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
