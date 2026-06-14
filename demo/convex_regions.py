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
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

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


def compute_intersection(
    region1: ConvexRegion,
    region2: ConvexRegion,
    obs_polys: Optional[List[Polygon]] = None,
) -> Optional[ConvexRegion]:
    """
    Compute intersection of two convex regions.

    For adjacent regions that share a boundary segment, returns a thin
    "virtual" intersection region around the shared edge.

    Args:
        region1, region2: ConvexRegion objects
        obs_polys: Obstacle polygons used to tighten the resulting H-rep so
            that NLP transition constraints cannot place trajectory nodes
            inside obstacles.  Pass the same obstacle list used when building
            the buffered regions.

    Returns:
        New ConvexRegion representing intersection, or None if disjoint or
        only touching at a single point
    """
    poly1 = region1.get_shapely_polygon()
    poly2 = region2.get_shapely_polygon()

    intersection = poly1.intersection(poly2)

    if not _supports_region_transition(intersection):
        return None

    result: Optional[ConvexRegion] = None

    # For proper area intersection
    if intersection.geom_type == 'Polygon' and intersection.area > 1e-10:
        vertices = np.array(intersection.exterior.coords[:-1])
        result = ConvexRegion(vertices=vertices)

    # For shared edges, create a thin interface region around the boundary.
    elif intersection.geom_type in ['LineString', 'MultiLineString', 'GeometryCollection']:
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
            result = ConvexRegion(vertices=vertices)
        elif clipped.geom_type == 'Polygon':
            vertices = np.array(clipped.exterior.coords[:-1])
            result = ConvexRegion(vertices=vertices)

    if result is not None and obs_polys:
        result = _tighten_intersection_hrep_against_obstacles(result, obs_polys)

    return result


def regions_intersect(region1: ConvexRegion, region2: ConvexRegion,
                      min_area: float = 1e-6,
                      min_shared_length: float = 1e-6,
                      tolerance: float = 0.0) -> bool:
    """
    Check if two regions overlap or share a boundary segment.

    When tolerance > 0, each polygon is dilated by that amount before the
    intersection test.  This converts vertex-only (point) contacts into small
    area overlaps so they are detected as adjacent.  Use a value strictly less
    than half the minimum wall thickness to avoid bridging across walls.

    Args:
        region1, region2: Regions to check
        min_area: Minimum overlap area to consider non-empty
        min_shared_length: Minimum shared boundary length to consider adjacent
        tolerance: Outward dilation applied to each polygon before checking.

    Returns:
        True if regions have area overlap or share a boundary segment
    """
    poly1 = region1.get_shapely_polygon()
    poly2 = region2.get_shapely_polygon()

    if tolerance > 0:
        poly1 = poly1.buffer(tolerance)
        poly2 = poly2.buffer(tolerance)

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
    buffer_size: float,
    obstacle_vertices: Optional[List[np.ndarray]] = None,
) -> List[ConvexRegion]:
    """
    Expand region polygons by a small buffer and clip them to the workspace.

    Each ACD2D polygon is buffered outward by buffer_size (mitre join keeps
    corners sharp → polygon stays convex) then clipped to the workspace
    boundary.  Adjacent regions gain an overlap strip of 2*buffer_size at
    each shared edge, which is required by the CRD interface QP.

    obstacle_vertices is accepted for API compatibility but the clip is
    against the workspace only: clipping against the full free-space
    (workspace minus obstacles) can produce non-convex polygons or
    MultiPolygon fragments, breaking the Chebyshev LP that needs a convex
    H-rep.  Obstacle safety is enforced downstream via the H-rep tightening
    added in _tighten_hrep_against_obstacles.
    """
    workspace = Polygon(np.asarray(workspace_vertices, dtype=np.float64))
    obs_polys: List[Polygon] = []
    if obstacle_vertices:
        obs_polys = [Polygon(np.asarray(v, dtype=np.float64)) for v in obstacle_vertices]

    regions: List[ConvexRegion] = []
    for i, verts in enumerate(vertices_list):
        poly = Polygon(np.asarray(verts, dtype=np.float64))
        buffered = poly.buffer(buffer_size, join_style='mitre')
        clipped = buffered.intersection(workspace)

        # If intersection fragments into multiple pieces (rare numerical artifact
        # near obstacle corners), keep only the largest polygon.
        if clipped.geom_type == 'MultiPolygon':
            clipped = max(clipped.geoms, key=lambda g: g.area)

        if clipped.geom_type == 'Polygon' and clipped.area > 1e-10:
            region_verts = np.array(clipped.exterior.coords[:-1])
        else:
            region_verts = np.asarray(verts, dtype=np.float64)

        region = ConvexRegion(vertices=region_verts, index=i,
                              label=region_node_label(i))

        # Tighten H-rep so the optimizer cannot place points inside obstacles.
        if obs_polys:
            region = _tighten_hrep_against_obstacles(
                region, np.asarray(verts, dtype=np.float64), obs_polys
            )

        regions.append(region)

    return regions


def _tighten_hrep_against_obstacles(
    region: "ConvexRegion",
    acd_verts: np.ndarray,
    obs_polys: List[Polygon],
) -> "ConvexRegion":
    """
    Add halfspace constraints from obstacle faces that the buffer crossed.

    For each obstacle O and each face F of O: if the original ACD2D polygon
    centroid is on the *outside* of F and the buffered region polygon crosses
    F into the obstacle interior, add the constraint  n_in · x ≤ b_F  where
    n_in is the inward normal of F (pointing into the obstacle).  This clips
    the H-rep back to the obstacle boundary at F without changing the vertex
    representation used for visualisation.

    Works for both convex and non-convex obstacles because each face is
    tested independently.
    """
    buffered_verts = region.vertices
    acd_poly = Polygon(acd_verts)
    acd_centroid = np.array(acd_poly.centroid.coords[0])
    buffered_poly = Polygon(buffered_verts)

    extra_A: List[np.ndarray] = []
    extra_b: List[float] = []

    for obs in obs_polys:
        # Compute actual intersection; skip if buffer doesn't enter obstacle interior.
        inter_region = buffered_poly.intersection(obs)
        if inter_region.is_empty or inter_region.area < 1e-10:
            continue

        obs_coords = np.array(obs.exterior.coords[:-1])
        # Ensure CCW (positive signed area) for consistent normal direction.
        signed_area = 0.0
        n_obs = len(obs_coords)
        for k in range(n_obs):
            j = (k + 1) % n_obs
            signed_area += (obs_coords[k, 0] * obs_coords[j, 1]
                            - obs_coords[j, 0] * obs_coords[k, 1])
        if signed_area < 0:
            obs_coords = obs_coords[::-1]

        for k in range(n_obs):
            v1 = obs_coords[k]
            v2 = obs_coords[(k + 1) % n_obs]
            edge = v2 - v1
            # Inward normal for CCW polygon (interior is to the LEFT of each
            # directed edge): rotate edge 90° CCW → (-dy, dx).
            n_in = np.array([-edge[1], edge[0]])
            norm = float(np.linalg.norm(n_in))
            if norm < 1e-10:
                continue
            n_in /= norm

            face_val = float(np.dot(n_in, v1))
            centroid_val = float(np.dot(n_in, acd_centroid))

            # ACD2D centroid is "outside" this face when centroid_val < face_val.
            if centroid_val >= face_val - 1e-6:
                continue

            # Only add the constraint if the actual buffer→obstacle intersection
            # polygon touches this face segment.  A global "max vertex > face_val"
            # check would incorrectly fire for faces the buffer never actually
            # crosses (e.g., a far face of a non-convex obstacle whose halfplane
            # extends over the region but whose segment is nowhere near it).
            face_seg = LineString([v1.tolist(), v2.tolist()])
            if not inter_region.intersects(face_seg.buffer(1e-5)):
                continue

            # Add constraint: n_in · x ≤ face_val (clip at obstacle face).
            extra_A.append(n_in)
            extra_b.append(face_val)

    if extra_A:
        new_A = np.vstack([region.A] + [a.reshape(1, -1) for a in extra_A])
        new_b = np.hstack([region.b, extra_b])
        new_verts = _vrep_from_hrep(new_A, new_b)
        if new_verts is None or len(new_verts) < 3:
            new_verts = region.vertices
        return ConvexRegion(vertices=new_verts, A=new_A, b=new_b,
                            index=region.index, label=region.label)
    return region


def _vrep_from_hrep(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute the V-rep (vertices) of a 2D convex polytope {x | Ax <= b}.

    Starts from a large bounding box and progressively intersects each
    halfplane using Shapely, yielding a convex polygon whose vertices
    are exactly those of the H-rep polytope.  This is the correct way
    to synchronise V-rep and H-rep after obstacle-face tightening.

    Returns None if the polytope is empty or degenerate.
    """
    large = 1e4
    poly = Polygon([(-large, -large), (large, -large),
                    (large, large), (-large, large)])
    for i in range(A.shape[0]):
        a = A[i]
        bi = float(b[i])
        norm2 = float(np.dot(a, a))
        if norm2 < 1e-12:
            continue
        # Point on boundary a @ x = bi
        p_on = a * (bi / norm2)
        a_hat = a / np.sqrt(norm2)
        perp = np.array([-a_hat[1], a_hat[0]])
        # Rectangle covering the feasible halfspace a @ x <= bi
        pts = [
            (p_on + perp * large).tolist(),
            (p_on - perp * large).tolist(),
            (p_on - perp * large - a_hat * large).tolist(),
            (p_on + perp * large - a_hat * large).tolist(),
        ]
        poly = poly.intersection(Polygon(pts))
        if poly.is_empty:
            return None
    if poly.is_empty or poly.geom_type != 'Polygon' or poly.area < 1e-10:
        return None
    return np.array(poly.exterior.coords[:-1])


def _tighten_intersection_hrep_against_obstacles(
    region: "ConvexRegion",
    obs_polys: List[Polygon],
) -> "ConvexRegion":
    """
    Add obstacle halfplane constraints to an intersection polygon H-rep.

    Unlike _tighten_hrep_against_obstacles (which uses the ACD2D centroid to
    determine which obstacle faces to clip), this function uses a free-space
    interior check.  Buffered intersection polygons (V-rep) may extend into
    obstacle space, so we subtract the obstacle union before checking whether
    an obstacle face cuts through the intersection.  This answers the correct
    geometric question: "does this face cut through the free-space part of the
    intersection?" and prevents the over-constraint that triggered
    NarrowInterfaceError on all paths (the V-rep / H-rep inconsistency bug).
    """
    poly = region.get_shapely_polygon()
    # Subtract obstacle space so the interior check uses free-space geometry only.
    obs_union = unary_union(obs_polys)
    free_poly = poly.difference(obs_union)
    if free_poly.is_empty:
        return region
    poly_interior = free_poly.buffer(-1e-8)

    extra_A: List[np.ndarray] = []
    extra_b: List[float] = []

    for obs in obs_polys:
        inter_area = poly.intersection(obs)
        if inter_area.is_empty or inter_area.area < 1e-10:
            continue

        obs_coords = np.array(obs.exterior.coords[:-1])
        n_obs = len(obs_coords)
        signed_area = 0.0
        for k in range(n_obs):
            j = (k + 1) % n_obs
            signed_area += (obs_coords[k, 0] * obs_coords[j, 1]
                            - obs_coords[j, 0] * obs_coords[k, 1])
        if signed_area < 0:
            obs_coords = obs_coords[::-1]

        for k in range(n_obs):
            v1 = obs_coords[k]
            v2 = obs_coords[(k + 1) % n_obs]
            edge = v2 - v1
            n_in = np.array([-edge[1], edge[0]])
            norm = float(np.linalg.norm(n_in))
            if norm < 1e-10:
                continue
            n_in /= norm
            face_val = float(np.dot(n_in, v1))

            # Add constraint only if face segment passes through the polygon
            # interior — this skips faces that only touch the polygon boundary
            # (e.g., shared workspace edges) and avoids spurious constraints.
            face_seg = LineString([v1.tolist(), v2.tolist()])
            if poly_interior.is_empty or not face_seg.intersects(poly_interior):
                continue

            # Require the obstacle-intersection area to actually touch this face.
            if not inter_area.intersects(face_seg.buffer(1e-5)):
                continue

            extra_A.append(n_in)
            extra_b.append(face_val)

    if extra_A:
        new_A = np.vstack([region.A] + [a.reshape(1, -1) for a in extra_A])
        new_b = np.hstack([region.b, extra_b])
        return ConvexRegion(vertices=region.vertices, A=new_A, b=new_b,
                            index=region.index, label=region.label)
    return region


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
