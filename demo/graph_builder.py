"""
graph_builder.py - Build directed graph on convex regions.

Constructs graph G = (V, E) where:
- V = V_r union {source, target} (region vertices + source + target)
- E = E_rr union E_s union E_t (inter-region + source + target edges)

Edges are created based on geometric adjacency (shared edge or overlap).
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

from convex_regions import ConvexRegion, regions_intersect, compute_intersection
from graph_types import (
    SOURCE,
    TARGET,
    is_terminal_node_id,
    region_index_from_node_id,
    region_node_label,
)


@dataclass
class GraphEdge:
    """
    Represents an edge in the region graph.
    
    Attributes:
        u, v: Source and destination node IDs
        edge_type: 'rr' (region-region), 's' (source), 't' (target)
        intersection: ConvexRegion of intersection (for rr edges)
    """
    u: str
    v: str
    edge_type: str  # 'rr', 's', 't'
    intersection: Optional[ConvexRegion] = None


@dataclass 
class RegionGraph:
    """
    Directed graph over convex regions with source and target.
    
    Attributes:
        regions: List of ConvexRegion objects
        graph: NetworkX DiGraph
        source_edges: List of edges from source
        target_edges: List of edges to target
        region_edges: List of inter-region edges
        adjacency_regions: Optional unbuffered regions used only to decide
            region-region graph connectivity and interface anchors. If omitted,
            `regions` are used for both optimization geometry and adjacency.
        intersections: Dict mapping edge tuple to intersection region
    """
    regions: List[ConvexRegion]
    start_pos: np.ndarray
    goal_pos: np.ndarray
    adjacency_regions: Optional[List[ConvexRegion]] = None
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    source_edges: List[Tuple[str, str]] = field(default_factory=list)
    target_edges: List[Tuple[str, str]] = field(default_factory=list)
    region_edges: List[Tuple[str, str]] = field(default_factory=list)
    intersections: Dict[Tuple[str, str], ConvexRegion] = field(default_factory=dict)
    _regions_by_index: Dict[int, ConvexRegion] = field(default_factory=dict, init=False, repr=False)
    _adjacency_regions: List[ConvexRegion] = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self):
        """Build the graph after initialization."""
        self._regions_by_index = {}
        for region in self.regions:
            if region.index in self._regions_by_index:
                raise ValueError(f"Duplicate region index in graph: {region.index}")
            self._regions_by_index[region.index] = region

        if self.adjacency_regions is None:
            self._adjacency_regions = self.regions
        else:
            if len(self.adjacency_regions) != len(self.regions):
                raise ValueError(
                    "adjacency_regions must have the same length as regions "
                    f"({len(self.adjacency_regions)} != {len(self.regions)})"
                )
            self._adjacency_regions = self.adjacency_regions

        self._build_graph()
    
    def _build_graph(self):
        """Construct the complete graph structure."""
        # Add source and target nodes
        self.graph.add_node(SOURCE, node_type='source')
        self.graph.add_node(TARGET, node_type='target')
        
        # Add region nodes
        for region in self.regions:
            node_id = region_node_label(region.index)
            self.graph.add_node(node_id, 
                               node_type='region',
                               region=region,
                               index=region.index)
        
        # Add source edges (source -> v for v containing start)
        for region in self.regions:
            if region.contains(self.start_pos, margin=0):
                node_id = region_node_label(region.index)
                self.graph.add_edge(SOURCE, node_id, edge_type='source')
                self.source_edges.append((SOURCE, node_id))
        
        # Add target edges (v -> target for v containing goal)
        for region in self.regions:
            if region.contains(self.goal_pos, margin=0):
                node_id = region_node_label(region.index)
                self.graph.add_edge(node_id, TARGET, edge_type='target')
                self.target_edges.append((node_id, TARGET))
        
        # Add inter-region edges (u -> v if Q_u and Q_v share an edge or overlap).
        # Connectivity may be evaluated on unbuffered adjacency regions so tiny
        # artificial overlaps created only by solver buffers do not create edges.
        n = len(self.regions)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                if regions_intersect(self._adjacency_regions[i], self._adjacency_regions[j]):
                    u_id = region_node_label(self.regions[i].index)
                    v_id = region_node_label(self.regions[j].index)
                    
                    # Compute intersection for interface constraints
                    intersection = compute_intersection(
                        self._adjacency_regions[i],
                        self._adjacency_regions[j],
                    )
                    
                    self.graph.add_edge(u_id, v_id, 
                                       edge_type='region',
                                       intersection=intersection)
                    self.region_edges.append((u_id, v_id))
                    
                    if intersection is not None:
                        self.intersections[(u_id, v_id)] = intersection
    
    def get_region_by_id(self, node_id: str) -> Optional[ConvexRegion]:
        """Get ConvexRegion object by node ID."""
        if is_terminal_node_id(node_id):
            return None
        idx = region_index_from_node_id(node_id)
        try:
            return self._regions_by_index[idx]
        except KeyError as exc:
            raise KeyError(f"Unknown region node ID {node_id!r}") from exc
    
    def get_region_nodes(self) -> List[str]:
        """Get list of region node IDs (excluding source/target)."""
        return [region_node_label(r.index) for r in self.regions]
    
    def get_source_regions(self) -> List[str]:
        """Get regions connected to source."""
        return [v for (u, v) in self.source_edges]
    
    def get_target_regions(self) -> List[str]:
        """Get regions connected to target."""
        return [u for (u, v) in self.target_edges]
    
    def enumerate_simple_paths(self, max_paths: int = 1000) -> List[List[str]]:
        """
        Enumerate all simple paths from source to target.
        
        Args:
            max_paths: Maximum number of paths to return (safety limit)
            
        Returns:
            List of paths, each path is a list of node IDs
        """
        paths = []
        try:
            for path in nx.all_simple_paths(self.graph, SOURCE, TARGET, cutoff=len(self.regions) + 2):
                paths.append(path)
                if len(paths) >= max_paths:
                    break
        except nx.NetworkXNoPath:
            pass
        
        return paths
    
    def get_path_regions(self, path: List[str]) -> List[ConvexRegion]:
        """
        Get ConvexRegion objects for a path (excluding source/target).
        
        Args:
            path: List of node IDs including source and target
            
        Returns:
            List of ConvexRegion objects
        """
        regions = []
        for node_id in path:
            if not is_terminal_node_id(node_id):
                regions.append(self.get_region_by_id(node_id))
        return regions
    
    def get_path_edges(self, path: List[str]) -> List[Tuple[str, str]]:
        """Get list of edges for a path."""
        return [(path[i], path[i+1]) for i in range(len(path) - 1)]
    
    def is_valid_path(self, path: List[str]) -> bool:
        """Check if path is valid in the graph."""
        if not path or path[0] != SOURCE or path[-1] != TARGET:
            return False
        
        for i in range(len(path) - 1):
            if not self.graph.has_edge(path[i], path[i+1]):
                return False
        
        return True
    
    def get_adjacency_info(self) -> Dict:
        """
        Get adjacency information for the optimizer.
        
        Returns:
            Dict with edge lists and intersection data
        """
        return {
            'source_edges': self.source_edges,
            'target_edges': self.target_edges,
            'region_edges': self.region_edges,
            'intersections': self.intersections,
            'n_regions': len(self.regions),
            'n_edges': self.graph.number_of_edges()
        }
    
    def print_summary(self):
        """Print graph summary."""
        print(f"Region Graph Summary:")
        print(f"  Regions: {len(self.regions)}")
        print(f"  Source edges: {len(self.source_edges)}")
        print(f"  Target edges: {len(self.target_edges)}")
        print(f"  Region edges: {len(self.region_edges)}")
        print(f"  Total edges: {self.graph.number_of_edges()}")
        
        # Count paths
        paths = self.enumerate_simple_paths(max_paths=100)
        if len(paths) < 100:
            print(f"  Simple paths: {len(paths)}")
        else:
            print(f"  Simple paths: >= {len(paths)} (limit reached)")


def build_region_graph(regions: List[ConvexRegion], 
                       start_pos: np.ndarray,
                       goal_pos: np.ndarray,
                       adjacency_regions: Optional[List[ConvexRegion]] = None) -> RegionGraph:
    """
    Build region graph from convex regions and start/goal positions.
    
    Args:
        regions: List of ConvexRegion objects
        start_pos: 2D start position
        goal_pos: 2D goal position
        adjacency_regions: Optional regions used for region-region adjacency.
            This is useful when optimization uses buffered regions but graph
            topology should be determined by the original unbuffered geometry.
        
    Returns:
        RegionGraph object
    """
    return RegionGraph(
        regions=regions,
        start_pos=start_pos,
        goal_pos=goal_pos,
        adjacency_regions=adjacency_regions,
    )


def visualize_graph_structure(graph: RegionGraph, filename: Optional[str] = None):
    """
    Create a visualization of the graph structure.
    
    Args:
        graph: RegionGraph object
        filename: Optional file to save the visualization
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Compute positions for NetworkX visualization
    pos = {}
    pos[SOURCE] = graph.start_pos
    pos[TARGET] = graph.goal_pos
    
    for region in graph.regions:
        node_id = region_node_label(region.index)
        pos[node_id] = region.get_centroid()
    
    # Draw nodes
    region_nodes = graph.get_region_nodes()
    nx.draw_networkx_nodes(graph.graph, pos, nodelist=[SOURCE], 
                          node_color='green', node_size=500, ax=ax)
    nx.draw_networkx_nodes(graph.graph, pos, nodelist=[TARGET],
                          node_color='red', node_size=500, ax=ax)
    nx.draw_networkx_nodes(graph.graph, pos, nodelist=region_nodes,
                          node_color='lightblue', node_size=300, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(graph.graph, pos, ax=ax, 
                          edge_color='gray', arrows=True,
                          arrowsize=15, connectionstyle="arc3,rad=0.1")
    
    # Draw labels
    nx.draw_networkx_labels(graph.graph, pos, ax=ax, font_size=8)
    
    ax.set_aspect('equal')
    ax.set_title("Region Graph Structure")
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def chebyshev_center(A: np.ndarray, b: np.ndarray) -> "Tuple[np.ndarray, float]":
    """Compute Chebyshev center and radius of {q | Aq <= b} via LP."""
    from scipy.optimize import linprog
    n = A.shape[1]
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    c_obj = np.zeros(n + 1)
    c_obj[-1] = -1.0
    A_ub = np.hstack([A, norms])
    bounds = [(None, None)] * n + [(0.0, None)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b, bounds=bounds, method='highs')
    if not res.success or res.x is None:
        raise ValueError(f"Chebyshev LP failed: {res.message}")
    return res.x[:n].copy(), float(res.x[n])


def add_composite_costs_to_graph(
    graph: RegionGraph,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    gamma_w: float = 1.0,
    gamma_h: float = 0.64,
) -> "Dict[int, Tuple[np.ndarray, float]]":
    """
    Annotate graph edges with 'composite_cost' = centroid_dist - gamma_w*interface_radius.
    Returns dict mapping region_index -> (chebyshev_center, chebyshev_radius).
    """
    centers: Dict[int, "Tuple[np.ndarray, float]"] = {}
    for region in graph.regions:
        c, r = chebyshev_center(region.A, region.b)
        centers[region.index] = (c, r)

    interface_radii: Dict["Tuple[str, str]", float] = {}
    for edge in graph.region_edges:
        u_id, v_id = edge
        ri = region_index_from_node_id(u_id)
        rj = region_index_from_node_id(v_id)
        A_int = np.vstack([graph.regions[ri].A, graph.regions[rj].A])
        b_int = np.concatenate([graph.regions[ri].b, graph.regions[rj].b])
        try:
            _, rho_ij = chebyshev_center(A_int, b_int)
        except ValueError:
            rho_ij = 0.0
        interface_radii[edge] = rho_ij

    for u, v, data in graph.graph.edges(data=True):
        if u == SOURCE:
            p1 = start_pos.astype(float)
        else:
            p1 = centers[region_index_from_node_id(u)][0]
        if v == TARGET:
            p2 = goal_pos.astype(float)
        else:
            p2 = centers[region_index_from_node_id(v)][0]
        dist = float(np.linalg.norm(p2 - p1))
        rho_interface = interface_radii.get((u, v), 0.0)
        cost = dist - gamma_w * rho_interface
        data['composite_cost'] = max(cost, 1e-9)

    return centers


def k_shortest_paths_generator(
    graph: RegionGraph,
    source: str,
    target: str,
    weight: str = 'composite_cost',
):
    """Generator yielding simple paths source->target in non-decreasing composite cost (Yen's)."""
    return nx.shortest_simple_paths(graph.graph, source, target, weight=weight)
