# Centroid-Refine-DMS Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `CentroidRefineDMSSolver` — a first-feasible motion planner that chains Chebyshev-center graph search → interface QP refinement → barrier-continuation DMS NLP.

**Architecture:** New solver class in `optimizer.py` orchestrates three new modules: `geometric_refiner.py` (interface QP), `warmstart.py` (IVP initialization), and `barrier_dms.py` (log-barrier NLP + continuation). All existing solver classes are untouched. The new mode is dispatched via `solver_mode = "centroid_refine_dms"`.

**Tech Stack:** Python 3.10+, CasADi (NLP/QP), IPOPT (NLP solver), OSQP via CasADi `qpsol`, NetworkX (Yen's K-shortest), SciPy `linprog` (Chebyshev LP), NumPy.

---

## File Map

**New files:**
- `demo/geometric_refiner.py` — Narrow-interface check + interface QP via CasADi `qpsol/osqp`
- `demo/warmstart.py` — IVP warm-start generation from interface points
- `demo/barrier_dms.py` — `BarrierSchedule`, failure taxonomy, `BarrierDMSSolver`
- `tests/test_geometric_refiner.py`
- `tests/test_warmstart.py`
- `tests/test_barrier_dms.py`
- `tests/test_centroid_refine_dms.py`

**Modified files:**
- `demo/dynamics.py` — Add abstract `compute_lipschitz_bound()` to `DynamicsModel`, implement in `UnicycleModel`, add `DoubleIntegratorDynamics`
- `demo/graph_builder.py` — Add `chebyshev_center()`, `add_composite_costs_to_graph()`, `k_shortest_paths_generator()`
- `demo/constraint_layers.py` — Add `build_barrier_log_terms()`, `compute_lipschitz_safety_gap()`
- `demo/optimizer.py` — Add `CentroidRefineDMSConfig`, new fields on `OptimizationResult`, `CentroidRefineDMSSolver`, dispatch in factory
- `demo/config.yaml` — Add `centroid_refine_dms:` config block
- `demo/experiments.py` — Add `ScenarioResult` dataclass with GCS-Bézier comparison slots

---

## Task 1: `dynamics.py` — Lipschitz bound + `DoubleIntegratorDynamics`

**Files:**
- Modify: `demo/dynamics.py`
- Test: `tests/test_dynamics.py`

- [ ] **Step 1: Write the failing tests**

Add at the bottom of `tests/test_dynamics.py`:

```python
class LipschitzBoundTests(unittest.TestCase):
    def test_unicycle_lipschitz_bound(self) -> None:
        dynamics = UnicycleModel(v_max=2.0)
        # Unit square has normals of norm 1; L_s = 1.0 * 2.0 = 2.0
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        L_s = dynamics.compute_lipschitz_bound([A])
        self.assertAlmostEqual(L_s, 2.0)

    def test_unicycle_lipschitz_scaled_normal(self) -> None:
        dynamics = UnicycleModel(v_max=3.0)
        # Normal of norm 2; L_s = 2.0 * 3.0 = 6.0
        A = np.array([[2.0, 0.0]], dtype=float)
        L_s = dynamics.compute_lipschitz_bound([A])
        self.assertAlmostEqual(L_s, 6.0)

    def test_double_integrator_properties(self) -> None:
        from dynamics import DoubleIntegratorDynamics
        dyn = DoubleIntegratorDynamics(v_max=2.0, a_max=3.0)
        self.assertEqual(dyn.n_x, 4)
        self.assertEqual(dyn.n_u, 2)
        self.assertEqual(dyn.n_pos, 2)
        self.assertEqual(dyn.position_indices, (0, 1))
        self.assertEqual(dyn.angle_indices, ())

    def test_double_integrator_dynamics(self) -> None:
        from dynamics import DoubleIntegratorDynamics
        dyn = DoubleIntegratorDynamics()
        x = np.array([1.0, 2.0, 0.5, -0.3])
        u = np.array([0.1, 0.2])
        xdot = dyn.f(x, u)
        np.testing.assert_allclose(xdot, [0.5, -0.3, 0.1, 0.2])

    def test_double_integrator_state_bounds(self) -> None:
        from dynamics import DoubleIntegratorDynamics
        dyn = DoubleIntegratorDynamics(v_max=2.0)
        lb, ub = dyn.state_bounds(np.array([0.0, 0.0]), np.array([5.0, 5.0]))
        self.assertEqual(len(lb), 4)
        self.assertAlmostEqual(lb[2], -2.0)
        self.assertAlmostEqual(ub[2],  2.0)

    def test_double_integrator_lipschitz(self) -> None:
        from dynamics import DoubleIntegratorDynamics
        dyn = DoubleIntegratorDynamics(v_max=2.0)
        A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        self.assertAlmostEqual(dyn.compute_lipschitz_bound([A]), 2.0)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_dynamics.py::LipschitzBoundTests -v 2>&1 | tail -20
```

Expected: `AttributeError: 'UnicycleModel' object has no attribute 'compute_lipschitz_bound'`

- [ ] **Step 3: Add abstract method `compute_lipschitz_bound` to `DynamicsModel`**

In `demo/dynamics.py`, after the `angle_big_m` method (line ~126), add:

```python
    @abstractmethod
    def compute_lipschitz_bound(self, A_list: List[np.ndarray]) -> float:
        """
        Return L_s = max_{i,j} ||a_{i,j}||_2 * v_max.

        Used to certify continuous-time safety between RK4 nodes.
        A_list: list of halfspace matrices (one per region on the path).
        """
        pass
```

- [ ] **Step 4: Implement `compute_lipschitz_bound` in `UnicycleModel`**

After `UnicycleModel.nominal_control` (around line ~256), add:

```python
    def compute_lipschitz_bound(self, A_list: List[np.ndarray]) -> float:
        max_normal = max(
            float(np.linalg.norm(A[j]))
            for A in A_list
            for j in range(A.shape[0])
        )
        return max_normal * self.v_max
```

- [ ] **Step 5: Add `DoubleIntegratorDynamics` class**

After the `UnicycleModel` class (after line ~266 where `ControlParameterization` begins), insert:

```python
@dataclass
class DoubleIntegratorDynamics(DynamicsModel):
    """
    Double integrator model.

    State:   x = [px, py, vx, vy]
    Control: u = [ax, ay]
    Dynamics: dp/dt = v,  dv/dt = u
    """
    v_max: float = 2.0
    a_max: float = 2.0

    @property
    def n_x(self) -> int:
        return 4

    @property
    def n_u(self) -> int:
        return 2

    @property
    def n_pos(self) -> int:
        return 2

    @property
    def position_indices(self) -> Tuple[int, ...]:
        return (0, 1)

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([x[2], x[3], u[0], u[1]], dtype=float)

    def f_casadi(self, x: ca.MX, u: ca.MX) -> ca.MX:
        return ca.vertcat(x[2], x[3], u[0], u[1])

    def project_to_position(self, x: np.ndarray) -> np.ndarray:
        return x[:2].copy()

    def project_to_position_casadi(self, x: ca.MX) -> ca.MX:
        return x[:2]

    def position_velocity(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x[2:4].copy()

    def position_velocity_casadi(self, x: ca.MX, u: ca.MX) -> ca.MX:
        return x[2:4]

    def state_bounds(
        self,
        position_lb: Optional[np.ndarray] = None,
        position_ub: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if position_lb is None:
            position_lb = np.array([-np.inf, -np.inf], dtype=np.float64)
        if position_ub is None:
            position_ub = np.array([np.inf, np.inf], dtype=np.float64)
        lb = np.array([position_lb[0], position_lb[1], -self.v_max, -self.v_max],
                      dtype=np.float64)
        ub = np.array([position_ub[0], position_ub[1],  self.v_max,  self.v_max],
                      dtype=np.float64)
        return lb, ub

    def control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.array([-self.a_max, -self.a_max], dtype=np.float64),
            np.array([ self.a_max,  self.a_max], dtype=np.float64),
        )

    def compute_lipschitz_bound(self, A_list: List[np.ndarray]) -> float:
        max_normal = max(
            float(np.linalg.norm(A[j]))
            for A in A_list
            for j in range(A.shape[0])
        )
        return max_normal * self.v_max
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_dynamics.py::LipschitzBoundTests -v
```

Expected: 6 tests pass.

- [ ] **Step 7: Run full test suite to check no regressions**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all previously passing tests still pass.

- [ ] **Step 8: Commit**

```bash
git -C /home/khoa/ws/GCS_MMS_demo add demo/dynamics.py tests/test_dynamics.py
git -C /home/khoa/ws/GCS_MMS_demo commit -m "feat: add compute_lipschitz_bound and DoubleIntegratorDynamics"
```

---

## Task 2: `graph_builder.py` — Chebyshev center + composite costs

**Files:**
- Modify: `demo/graph_builder.py`
- Test: `tests/test_graph_builder.py`

- [ ] **Step 1: Write the failing tests**

Add at the bottom of `tests/test_graph_builder.py`:

```python
class ChebyshevCenterTests(unittest.TestCase):
    def test_unit_square_center(self) -> None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
        from graph_builder import chebyshev_center
        # Unit square [0,1]^2: Ax <= b with normals of norm 1
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        b = np.array([1.0, 0.0, 1.0, 0.0])
        center, radius = chebyshev_center(A, b)
        np.testing.assert_allclose(center, [0.5, 0.5], atol=1e-6)
        self.assertAlmostEqual(radius, 0.5, places=6)

    def test_rectangle_center(self) -> None:
        from graph_builder import chebyshev_center
        # Rectangle [0,2] x [0,1]: narrower in y so radius = 0.5
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        b = np.array([2.0, 0.0, 1.0, 0.0])
        center, radius = chebyshev_center(A, b)
        self.assertAlmostEqual(radius, 0.5, places=6)

    def test_infeasible_raises(self) -> None:
        from graph_builder import chebyshev_center
        # Empty polytope: x <= -1 AND x >= 1 contradiction
        A = np.array([[1.0], [-1.0]])
        b = np.array([-1.0, -1.0])
        with self.assertRaises(ValueError):
            chebyshev_center(A, b)


class CompositeCostTests(unittest.TestCase):
    def _make_two_region_graph(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
        from convex_regions import create_regions_from_vertices_list
        from graph_builder import build_region_graph
        regions = create_regions_from_vertices_list([
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
            np.array([[0.8, 0], [2, 0], [2, 1], [0.8, 1]], dtype=float),
        ])
        start = np.array([0.1, 0.5])
        goal  = np.array([1.9, 0.5])
        return build_region_graph(regions, start, goal)

    def test_add_composite_costs_returns_centers(self) -> None:
        from graph_builder import add_composite_costs_to_graph
        graph = self._make_two_region_graph()
        centers = add_composite_costs_to_graph(
            graph,
            start_pos=np.array([0.1, 0.5]),
            goal_pos=np.array([1.9, 0.5]),
        )
        self.assertEqual(len(centers), 2)
        for idx, (c, r) in centers.items():
            self.assertEqual(len(c), 2)
            self.assertGreater(r, 0.0)

    def test_composite_cost_on_edges(self) -> None:
        from graph_builder import add_composite_costs_to_graph, SOURCE, TARGET
        graph = self._make_two_region_graph()
        add_composite_costs_to_graph(
            graph,
            start_pos=np.array([0.1, 0.5]),
            goal_pos=np.array([1.9, 0.5]),
        )
        for u, v, data in graph.graph.edges(data=True):
            self.assertIn('composite_cost', data)
            self.assertGreater(data['composite_cost'], 0.0)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_graph_builder.py::ChebyshevCenterTests tests/test_graph_builder.py::CompositeCostTests -v 2>&1 | tail -10
```

Expected: `ImportError` or `AttributeError` (functions not yet defined).

- [ ] **Step 3: Add `chebyshev_center` to `graph_builder.py`**

At the top of `demo/graph_builder.py`, ensure `from scipy.optimize import linprog` is imported. Then add this function before the `GraphEdge` dataclass:

```python
def chebyshev_center(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute the Chebyshev center and radius of the polytope {q | A q <= b}.

    Solves: max rho  s.t.  a_j^T q + ||a_j|| * rho <= b_j for all j, rho >= 0
    Returns (center, radius). Raises ValueError if the LP fails (infeasible/unbounded).
    """
    from scipy.optimize import linprog
    n = A.shape[1]
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    c_obj = np.zeros(n + 1)
    c_obj[-1] = -1.0  # maximize rho
    A_ub = np.hstack([A, norms])
    bounds = [(None, None)] * n + [(0.0, None)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b, bounds=bounds, method='highs')
    if not res.success or res.x is None:
        raise ValueError(f"Chebyshev LP failed: {res.message}")
    return res.x[:n].copy(), float(res.x[n])
```

- [ ] **Step 4: Add `add_composite_costs_to_graph` to `graph_builder.py`**

Add after `chebyshev_center`:

```python
def add_composite_costs_to_graph(
    graph: "RegionGraph",
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    gamma_w: float = 1.0,
    gamma_h: float = 0.64,
) -> Dict[int, Tuple[np.ndarray, float]]:
    """
    Compute Chebyshev centers for all regions and annotate graph edges with
    'composite_cost' = centroid_dist - gamma_w * interface_radius.

    The heading-change penalty (gamma_h * |Δθ|^2) is path-dependent and is NOT
    included in the edge attribute; it is applied during path-level scoring.

    Returns: dict mapping region_index -> (chebyshev_center, chebyshev_radius)
    """
    centers: Dict[int, Tuple[np.ndarray, float]] = {}
    for region in graph.regions:
        c, r = chebyshev_center(region.A, region.b)
        centers[region.index] = (c, r)

    # Precompute interface (intersection) Chebyshev radii for region-region edges
    interface_radii: Dict[Tuple[str, str], float] = {}
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_graph_builder.py::ChebyshevCenterTests tests/test_graph_builder.py::CompositeCostTests -v
```

Expected: all 5 tests pass.

- [ ] **Step 6: Run full test suite**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/ -v --tb=short 2>&1 | tail -15
```

- [ ] **Step 7: Commit**

```bash
git -C /home/khoa/ws/GCS_MMS_demo add demo/graph_builder.py tests/test_graph_builder.py
git -C /home/khoa/ws/GCS_MMS_demo commit -m "feat: add chebyshev_center and add_composite_costs_to_graph"
```

---

## Task 3: `graph_builder.py` — K-shortest paths generator

**Files:**
- Modify: `demo/graph_builder.py`
- Test: `tests/test_graph_builder.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_graph_builder.py`:

```python
class KShortestPathsTests(unittest.TestCase):
    def _make_graph(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
        from convex_regions import create_regions_from_vertices_list
        from graph_builder import build_region_graph, add_composite_costs_to_graph
        # Three regions in a row with overlap
        regions = create_regions_from_vertices_list([
            np.array([[0, 0], [1.2, 0], [1.2, 1], [0, 1]], dtype=float),
            np.array([[0.8, 0], [2.2, 0], [2.2, 1], [0.8, 1]], dtype=float),
            np.array([[1.8, 0], [3.0, 0], [3.0, 1], [1.8, 1]], dtype=float),
        ])
        start = np.array([0.1, 0.5])
        goal  = np.array([2.9, 0.5])
        graph = build_region_graph(regions, start, goal)
        add_composite_costs_to_graph(graph, start, goal)
        return graph

    def test_generator_yields_paths(self) -> None:
        from graph_builder import k_shortest_paths_generator, SOURCE, TARGET
        graph = self._make_graph()
        gen = k_shortest_paths_generator(graph, SOURCE, TARGET)
        path = next(gen)
        self.assertEqual(path[0], SOURCE)
        self.assertEqual(path[-1], TARGET)

    def test_paths_are_simple(self) -> None:
        from graph_builder import k_shortest_paths_generator, SOURCE, TARGET
        graph = self._make_graph()
        gen = k_shortest_paths_generator(graph, SOURCE, TARGET)
        for _ in range(3):
            try:
                path = next(gen)
                self.assertEqual(len(path), len(set(path)), "path has repeated nodes")
            except StopIteration:
                break

    def test_all_yielded_paths_are_distinct(self) -> None:
        from graph_builder import k_shortest_paths_generator, SOURCE, TARGET
        graph = self._make_graph()
        all_paths = []
        gen = k_shortest_paths_generator(graph, SOURCE, TARGET)
        for _ in range(5):
            try:
                all_paths.append(tuple(next(gen)))
            except StopIteration:
                break
        self.assertEqual(len(all_paths), len(set(all_paths)))
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_graph_builder.py::KShortestPathsTests -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'k_shortest_paths_generator'`.

- [ ] **Step 3: Add `k_shortest_paths_generator` to `graph_builder.py`**

Add after `add_composite_costs_to_graph`:

```python
def k_shortest_paths_generator(
    graph: "RegionGraph",
    source: str,
    target: str,
    weight: str = 'composite_cost',
):
    """
    Generator yielding simple source-to-target paths in non-decreasing composite cost.

    Uses Yen's algorithm (networkx.shortest_simple_paths).
    Each yielded path is a list of node IDs with no repeated nodes.
    Call next() to advance; exhausted when StopIteration is raised.
    """
    import networkx as nx
    return nx.shortest_simple_paths(graph.graph, source, target, weight=weight)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_graph_builder.py::KShortestPathsTests -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/ --tb=short 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git -C /home/khoa/ws/GCS_MMS_demo add demo/graph_builder.py tests/test_graph_builder.py
git -C /home/khoa/ws/GCS_MMS_demo commit -m "feat: add k_shortest_paths_generator to graph_builder"
```

---

## Task 4: `geometric_refiner.py` — Narrow-interface check + interface QP

**Files:**
- Create: `demo/geometric_refiner.py`
- Create: `tests/test_geometric_refiner.py`

- [ ] **Step 1: Create `tests/test_geometric_refiner.py`**

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from convex_regions import create_regions_from_vertices_list
from graph_builder import build_region_graph
from geometric_refiner import (
    InterfaceQPConfig,
    NarrowInterfaceError,
    solve_interface_refinement,
)


def _two_region_graph():
    regions = create_regions_from_vertices_list([
        np.array([[0, 0], [1.2, 0], [1.2, 1], [0, 1]], dtype=float),
        np.array([[0.8, 0], [2.0, 0], [2.0, 1], [0.8, 1]], dtype=float),
    ])
    start = np.array([0.1, 0.5])
    goal  = np.array([1.9, 0.5])
    return build_region_graph(regions, start, goal), [0, 1]


class InterfaceQPTests(unittest.TestCase):
    def test_qp_returns_correct_shape(self) -> None:
        graph, path_regions = _two_region_graph()
        cfg = InterfaceQPConfig(delta_safe=0.02, delta_extra=0.01)
        z = solve_interface_refinement(
            graph, path_regions,
            q_start=np.array([0.1, 0.5]),
            q_goal=np.array([1.9, 0.5]),
            config=cfg,
        )
        # z shape: (m+1, n_pos) = (3, 2) for m=2
        self.assertEqual(z.shape, (3, 2))  # (m+1, n_pos) = (3, 2) for m=2

    def test_qp_fixed_endpoints(self) -> None:
        graph, path_regions = _two_region_graph()
        cfg = InterfaceQPConfig(delta_safe=0.02, delta_extra=0.01)
        q_start = np.array([0.1, 0.5])
        q_goal  = np.array([1.9, 0.5])
        z = solve_interface_refinement(graph, path_regions, q_start, q_goal, cfg)
        np.testing.assert_allclose(z[0], q_start, atol=1e-9)
        np.testing.assert_allclose(z[-1], q_goal, atol=1e-9)

    def test_strict_interior_guarantee(self) -> None:
        graph, path_regions = _two_region_graph()
        cfg = InterfaceQPConfig(delta_safe=0.02, delta_extra=0.01)
        z = solve_interface_refinement(
            graph, path_regions,
            np.array([0.1, 0.5]), np.array([1.9, 0.5]), cfg
        )
        margin = cfg.delta_safe + cfg.delta_extra
        # Each free interface point z[1..m] must be in its region with margin
        m = len(path_regions)
        for i in range(1, m):  # interface points z[1],...,z[m-1] only
            # z[i] must be in C_{i-1} ∩ C_i  (0-indexed regions)
            for ri in [path_regions[i-1], path_regions[i]]:
                region = graph.regions[ri]
                pt = z[i]
                slacks = region.b - region.A @ pt
                self.assertTrue(
                    np.all(slacks >= margin - 1e-6),
                    f"Strict interior violated at interface {i}, region {ri}: slacks={slacks}"
                )

    def test_narrow_interface_raises(self) -> None:
        # Create two regions with barely-overlapping interface
        regions = create_regions_from_vertices_list([
            np.array([[0, 0], [1.001, 0], [1.001, 1], [0, 1]], dtype=float),
            np.array([[0.999, 0], [2.0, 0], [2.0, 1], [0.999, 1]], dtype=float),
        ])
        start = np.array([0.1, 0.5])
        goal  = np.array([1.9, 0.5])
        graph = build_region_graph(regions, start, goal)
        # delta_extra = 0.05 makes the required margin 0.07, larger than interface width 0.002
        cfg = InterfaceQPConfig(delta_safe=0.02, delta_extra=0.05)
        with self.assertRaises(NarrowInterfaceError):
            solve_interface_refinement(graph, [0, 1], start, goal, cfg)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to confirm failure**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_geometric_refiner.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'geometric_refiner'`.

- [ ] **Step 3: Create `demo/geometric_refiner.py`**

```python
"""
geometric_refiner.py - Interface QP for Centroid-Refine-DMS.

Finds interface waypoints strictly inside the shrunken intersections of
consecutive convex regions. Guarantees s_{i,j,k}(warm-start) >= delta_extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import casadi as ca
import numpy as np

from graph_builder import RegionGraph, chebyshev_center


@dataclass
class InterfaceQPConfig:
    delta_safe: float = 0.02
    delta_extra: float = 0.01   # = delta_safe / 2 by default; set explicitly
    lambda_s: float = 0.0       # smoothness weight (0 = pure shortest polyline)


class NarrowInterfaceError(Exception):
    """Raised when an interface Chebyshev radius < delta_safe + delta_extra."""
    pass


class InterfaceQPInfeasible(Exception):
    """Raised when the QP solver finds no feasible point (should be pre-empted by narrow check)."""
    pass


def solve_interface_refinement(
    graph: RegionGraph,
    path_regions: List[int],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    config: InterfaceQPConfig,
) -> np.ndarray:
    """
    Solve the interface refinement QP for a given region path.

    Returns z of shape (m+1, n_pos):
      z[0]        = q_start  (fixed)
      z[1..m-1]   = free interface points (optimized); z[i] ∈ C_i ∩ C_{i+1}
      z[m]        = q_goal   (fixed)

    For m=1 (single region): no free variables; returns [[q_start], [q_goal]].

    Each free point z[i] (i=1..m-1) lies strictly inside the shrunken intersection
    C_{path_regions[i-1]} ∩ C_{path_regions[i]} with margin >= delta_safe + delta_extra.

    Raises NarrowInterfaceError before solving if any interface is too narrow.
    Raises InterfaceQPInfeasible if the QP solver fails despite passing the check.
    """
    m = len(path_regions)
    n_pos = graph.regions[path_regions[0]].A.shape[1]
    margin = config.delta_safe + config.delta_extra

    # ── Pre-solve narrow-interface check ────────────────────────────────────
    for i in range(m - 1):
        ri  = path_regions[i]
        ri1 = path_regions[i + 1]
        A_int = np.vstack([graph.regions[ri].A, graph.regions[ri1].A])
        b_int = np.concatenate([graph.regions[ri].b, graph.regions[ri1].b])
        try:
            _, rho_ij = chebyshev_center(A_int, b_int)
        except ValueError:
            raise NarrowInterfaceError(
                f"Interface {i}-{i+1} (regions {ri}/{ri1}) is empty."
            )
        if rho_ij < margin:
            raise NarrowInterfaceError(
                f"Interface {i}-{i+1} radius {rho_ij:.4f} < required {margin:.4f}."
            )

    # ── Build QP ─────────────────────────────────────────────────────────────
    # m-1 free interface points: z[1],...,z[m-1], each at C_i ∩ C_{i+1}
    n_free = m - 1
    z_vars = [ca.MX.sym(f'z_{i}', n_pos) for i in range(n_free)]
    z_chain = [ca.DM(q_start)] + z_vars + [ca.DM(q_goal)]

    # Objective: sum of segment lengths + optional smoothness
    obj = ca.MX(0.0)
    for i in range(m):
        diff = z_chain[i + 1] - z_chain[i]
        obj = obj + ca.dot(diff, diff)
    if config.lambda_s > 0.0:
        for i in range(1, n_free):
            curv = z_chain[i + 1] - 2.0 * z_chain[i] + z_chain[i - 1]
            obj = obj + config.lambda_s * ca.dot(curv, curv)

    # Constraints: z_vars[i] in C_{path_regions[i]} ∩ C_{path_regions[i+1]}
    g_list: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    for i in range(n_free):
        z_i = z_vars[i]
        for ri in [path_regions[i], path_regions[i + 1]]:
            region = graph.regions[ri]
            for j in range(region.A.shape[0]):
                g_list.append(region.A[j, :] @ z_i - (region.b[j] - margin))
                lbg.append(-np.inf)
                ubg.append(0.0)

    if n_free == 0:
        # Single-region path — no interface points to optimize
        z_out = np.zeros((m + 1, n_pos))
        z_out[0] = q_start
        z_out[m] = q_goal
        return z_out

    x_sym = ca.vertcat(*z_vars)
    g_sym = ca.vertcat(*g_list) if g_list else ca.MX(0, 1)

    qp = {'x': x_sym, 'f': obj, 'g': g_sym}
    opts = {'osqp': {'verbose': False}, 'print_time': False, 'error_on_fail': False}
    solver = ca.qpsol('interface_qp', 'osqp', qp, opts)

    x0_vals = []
    for i in range(n_free):
        tau = (i + 1) / (m)
        x0_vals.extend((q_start * (1 - tau) + q_goal * tau).tolist())

    sol = solver(x0=x0_vals, lbg=lbg, ubg=ubg)
    stats = solver.stats()

    if not stats.get('success', False):
        raise InterfaceQPInfeasible(
            f"Interface QP failed: {stats.get('return_status', 'unknown')}"
        )

    x_opt = np.array(sol['x']).flatten()

    z_out = np.zeros((m + 1, n_pos))
    z_out[0] = q_start
    z_out[m] = q_goal
    for i in range(n_free):
        z_out[i + 1] = x_opt[i * n_pos:(i + 1) * n_pos]

    return z_out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_geometric_refiner.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/ --tb=short 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git -C /home/khoa/ws/GCS_MMS_demo add demo/geometric_refiner.py tests/test_geometric_refiner.py
git -C /home/khoa/ws/GCS_MMS_demo commit -m "feat: add geometric_refiner with interface QP and narrow-interface check"
```

---

## Task 5: `warmstart.py` — IVP warm-start generation

**Files:**
- Create: `demo/warmstart.py`
- Create: `tests/test_warmstart.py`

- [ ] **Step 1: Create `tests/test_warmstart.py`**

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from convex_regions import create_regions_from_vertices_list
from graph_builder import build_region_graph
from dynamics import UnicycleModel, DoubleIntegratorDynamics
from geometric_refiner import InterfaceQPConfig, solve_interface_refinement
from warmstart import WarmStartConfig, generate_warm_start_ivp


def _two_region_setup():
    regions = create_regions_from_vertices_list([
        np.array([[0, 0], [1.2, 0], [1.2, 1], [0, 1]], dtype=float),
        np.array([[0.8, 0], [2.0, 0], [2.0, 1], [0.8, 1]], dtype=float),
    ])
    start = np.array([0.1, 0.5])
    goal  = np.array([1.9, 0.5])
    graph = build_region_graph(regions, start, goal)
    path_regions = [0, 1]
    qp_cfg = InterfaceQPConfig(delta_safe=0.02, delta_extra=0.01)
    z = solve_interface_refinement(
        graph, path_regions,
        np.array([0.1, 0.5]), np.array([1.9, 0.5]), qp_cfg
    )
    return graph, path_regions, z, qp_cfg


class WarmStartUnicycleTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.path_regions, self.z, self.qp_cfg = _two_region_setup()
        self.dynamics = UnicycleModel(v_max=2.0, omega_max=np.pi)
        self.ws_cfg = WarmStartConfig(
            delta_min=0.01, delta_max=10.0, v_nom_fraction=0.5, n_int=5
        )

    def test_returns_correct_structure(self):
        ws = generate_warm_start_ivp(
            self.z, self.path_regions, self.dynamics, self.ws_cfg
        )
        m = len(self.path_regions)
        self.assertEqual(len(ws.x_nodes), m)
        self.assertEqual(len(ws.u_list), m)
        self.assertEqual(len(ws.delta_list), m)
        for i in range(m):
            self.assertEqual(ws.x_nodes[i].shape, (self.ws_cfg.n_int + 1, self.dynamics.n_x))

    def test_strict_interior_invariant(self):
        ws = generate_warm_start_ivp(
            self.z, self.path_regions, self.dynamics, self.ws_cfg
        )
        delta_extra = self.qp_cfg.delta_extra
        for seg_i, region_idx in enumerate(self.path_regions):
            region = self.graph.regions[region_idx]
            for k in range(self.ws_cfg.n_int + 1):
                pos = ws.x_nodes[seg_i][k, :2]
                slacks = region.b - self.qp_cfg.delta_safe - region.A @ pos
                self.assertTrue(
                    np.all(slacks >= delta_extra - 1e-6),
                    f"Strict interior violated: seg={seg_i}, node={k}, slacks={slacks}"
                )

    def test_unicycle_heading_from_direction(self):
        ws = generate_warm_start_ivp(
            self.z, self.path_regions, self.dynamics, self.ws_cfg
        )
        for i in range(len(self.path_regions)):
            direction = self.z[i + 1] - self.z[i]
            expected_theta = np.arctan2(direction[1], direction[0])
            actual_theta = ws.x_nodes[i][0, 2]
            self.assertAlmostEqual(actual_theta, expected_theta, places=6)

    def test_duration_positive_and_bounded(self):
        ws = generate_warm_start_ivp(
            self.z, self.path_regions, self.dynamics, self.ws_cfg
        )
        for delta in ws.delta_list:
            self.assertGreaterEqual(delta, self.ws_cfg.delta_min)
            self.assertLessEqual(delta, self.ws_cfg.delta_max)


class WarmStartDoubleIntTests(unittest.TestCase):
    def test_double_integrator_zero_acceleration(self):
        graph, path_regions, z, qp_cfg = _two_region_setup()
        dynamics = DoubleIntegratorDynamics(v_max=2.0)
        ws_cfg = WarmStartConfig(delta_min=0.01, delta_max=10.0, v_nom_fraction=0.5, n_int=5)
        ws = generate_warm_start_ivp(z, path_regions, dynamics, ws_cfg)
        for seg_i in range(len(path_regions)):
            np.testing.assert_allclose(ws.u_list[seg_i], [0.0, 0.0], atol=1e-10)

    def test_double_integrator_constant_velocity(self):
        graph, path_regions, z, qp_cfg = _two_region_setup()
        dynamics = DoubleIntegratorDynamics(v_max=2.0)
        ws_cfg = WarmStartConfig(delta_min=0.01, delta_max=10.0, v_nom_fraction=0.5, n_int=4)
        ws = generate_warm_start_ivp(z, path_regions, dynamics, ws_cfg)
        for seg_i in range(len(path_regions)):
            expected_v = (z[seg_i + 1] - z[seg_i]) / ws.delta_list[seg_i]
            for k in range(ws_cfg.n_int + 1):
                np.testing.assert_allclose(
                    ws.x_nodes[seg_i][k, 2:4], expected_v, atol=1e-9
                )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to confirm failure**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_warmstart.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'warmstart'`.

- [ ] **Step 3: Create `demo/warmstart.py`**

```python
"""
warmstart.py - IVP warm-start generation for Centroid-Refine-DMS.

Generates (x_nodes, u_list, delta_list) from interface waypoints z.
Guarantees s_{i,j,k}(x_nodes) >= delta_extra > 0 (strict-interior invariant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from dynamics import DynamicsModel, DoubleIntegratorDynamics, UnicycleModel


@dataclass
class WarmStartConfig:
    delta_min: float = 0.01
    delta_max: float = 10.0
    v_nom_fraction: float = 0.5  # v_nom = fraction * (v_max + |v_min|) / 2
    n_int: int = 20              # number of RK4 integration steps per segment


@dataclass
class WarmStart:
    """Warm-start data for one path."""
    x_nodes: List[np.ndarray]   # [segment][node, state_dim]
    u_list: List[np.ndarray]    # [segment] constant control
    delta_list: List[float]     # [segment] duration


def generate_warm_start_ivp(
    z: np.ndarray,
    path_regions: List[int],
    dynamics: DynamicsModel,
    config: WarmStartConfig,
) -> WarmStart:
    """
    Generate an initial guess from interface waypoints z (shape m+1, n_pos).
    This is a feasible initial guess for the NLP (state nodes lie in strict interior),
    NOT an IVP-feasible trajectory — see Assumption A3 in the spec.

    For segment i (0-indexed): q linearly interpolates z[i] -> z[i+1].
    Strict-interior guarantee holds by construction of z from the interface QP:
    both z[i] and z[i+1] satisfy A_{region_i} q <= b - (delta_safe + delta_extra),
    so every convex combination also satisfies it, giving slack >= delta_extra.
    """
    m = len(path_regions)
    n_int = config.n_int

    if isinstance(dynamics, UnicycleModel):
        return _warmstart_unicycle(z, m, dynamics, config, n_int)
    elif isinstance(dynamics, DoubleIntegratorDynamics):
        return _warmstart_double_integrator(z, m, dynamics, config, n_int)
    else:
        raise NotImplementedError(
            f"generate_warm_start_ivp not implemented for {type(dynamics).__name__}"
        )


def _compute_v_nom(dynamics: DynamicsModel, config: WarmStartConfig) -> float:
    u_lb, u_ub = dynamics.control_bounds()
    if dynamics.n_u > 0:
        v_max = float(u_ub[0])
        v_min = float(u_lb[0])
        return config.v_nom_fraction * (v_max + abs(v_min)) / 2.0
    return 1.0


def _warmstart_unicycle(
    z: np.ndarray,
    m: int,
    dynamics: UnicycleModel,
    config: WarmStartConfig,
    n_int: int,
) -> WarmStart:
    x_nodes_list: List[np.ndarray] = []
    u_list: List[np.ndarray] = []
    delta_list: List[float] = []

    v_nom = max(_compute_v_nom(dynamics, config), 1e-6)

    for i in range(m):
        z_prev = z[i]
        z_next = z[i + 1]

        seg_len = float(np.linalg.norm(z_next - z_prev))
        delta_i = max(config.delta_min,
                      min(config.delta_max, seg_len / v_nom))

        direction = z_next - z_prev
        theta_i = float(np.arctan2(direction[1], direction[0])) if seg_len > 1e-9 else 0.0

        if i > 0:
            prev_dir = z[i] - z[i - 1]
            prev_len = float(np.linalg.norm(prev_dir))
            theta_prev = float(np.arctan2(prev_dir[1], prev_dir[0])) if prev_len > 1e-9 else theta_i
            dtheta = abs(theta_i - theta_prev)
            dtheta = min(dtheta, 2 * np.pi - dtheta)
            if dtheta > 0 and delta_i > 0:
                omega_needed = dtheta / delta_i
                if omega_needed > dynamics.omega_max:
                    delta_i = min(config.delta_max, dtheta / dynamics.omega_max)

        taus = np.linspace(0.0, 1.0, n_int + 1)
        x_seg = np.zeros((n_int + 1, dynamics.n_x))
        for k, tau in enumerate(taus):
            x_seg[k, 0] = z_prev[0] * (1 - tau) + z_next[0] * tau
            x_seg[k, 1] = z_prev[1] * (1 - tau) + z_next[1] * tau
            x_seg[k, 2] = theta_i

        v_i = seg_len / delta_i if delta_i > 0 else 0.0
        v_i = float(np.clip(v_i, dynamics.v_min, dynamics.v_max))
        omega_i = 0.0
        if i > 0:
            prev_dir = z[i] - z[i - 1]
            prev_len = float(np.linalg.norm(prev_dir))
            theta_prev = float(np.arctan2(prev_dir[1], prev_dir[0])) if prev_len > 1e-9 else theta_i
            raw_omega = (theta_i - theta_prev) / delta_i if delta_i > 0 else 0.0
            omega_i = float(np.clip(raw_omega, dynamics.omega_min, dynamics.omega_max))

        x_nodes_list.append(x_seg)
        u_list.append(np.array([v_i, omega_i], dtype=float))
        delta_list.append(delta_i)

    return WarmStart(x_nodes=x_nodes_list, u_list=u_list, delta_list=delta_list)


def _warmstart_double_integrator(
    z: np.ndarray,
    m: int,
    dynamics: DoubleIntegratorDynamics,
    config: WarmStartConfig,
    n_int: int,
) -> WarmStart:
    x_nodes_list: List[np.ndarray] = []
    u_list: List[np.ndarray] = []
    delta_list: List[float] = []

    v_nom = max(_compute_v_nom(dynamics, config), 1e-6)

    for i in range(m):
        z_prev = z[i]
        z_next = z[i + 1]

        seg_len = float(np.linalg.norm(z_next - z_prev))
        delta_i = max(config.delta_min,
                      min(config.delta_max, seg_len / v_nom))

        vel = (z_next - z_prev) / delta_i
        vel = np.clip(vel, -dynamics.v_max, dynamics.v_max)

        taus = np.linspace(0.0, 1.0, n_int + 1)
        x_seg = np.zeros((n_int + 1, dynamics.n_x))
        for k, tau in enumerate(taus):
            x_seg[k, 0] = z_prev[0] * (1 - tau) + z_next[0] * tau
            x_seg[k, 1] = z_prev[1] * (1 - tau) + z_next[1] * tau
            x_seg[k, 2] = vel[0]
            x_seg[k, 3] = vel[1]

        x_nodes_list.append(x_seg)
        u_list.append(np.zeros(dynamics.n_u, dtype=float))
        delta_list.append(delta_i)

    return WarmStart(x_nodes=x_nodes_list, u_list=u_list, delta_list=delta_list)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_warmstart.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/ --tb=short 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git -C /home/khoa/ws/GCS_MMS_demo add demo/warmstart.py tests/test_warmstart.py
git -C /home/khoa/ws/GCS_MMS_demo commit -m "feat: add warmstart IVP generator with strict-interior guarantee"
```

---

## Task 6: `constraint_layers.py` — Barrier log terms + Lipschitz safety gap

**Files:**
- Modify: `demo/constraint_layers.py`
- Test: `tests/test_constraint_layers.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_constraint_layers.py`:

```python
class BarrierLogTermsTests(unittest.TestCase):
    def setUp(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
        from convex_regions import create_regions_from_vertices_list
        from graph_builder import build_region_graph
        from dynamics import UnicycleModel
        from warmstart import WarmStartConfig, generate_warm_start_ivp
        from geometric_refiner import InterfaceQPConfig, solve_interface_refinement
        import casadi as ca

        regions = create_regions_from_vertices_list([
            np.array([[0, 0], [1.2, 0], [1.2, 1], [0, 1]], dtype=float),
            np.array([[0.8, 0], [2.0, 0], [2.0, 1], [0.8, 1]], dtype=float),
        ])
        start = np.array([0.1, 0.5])
        goal  = np.array([1.9, 0.5])
        self.graph = build_region_graph(regions, start, goal)
        self.path_regions = [0, 1]
        qp_cfg = InterfaceQPConfig(delta_safe=0.02, delta_extra=0.01)
        z = solve_interface_refinement(self.graph, self.path_regions, start, goal, qp_cfg)
        self.dynamics = UnicycleModel()
        ws_cfg = WarmStartConfig(n_int=5)
        ws = generate_warm_start_ivp(z, self.path_regions, self.dynamics, ws_cfg)
        self.ws = ws
        self.n_int = ws_cfg.n_int
        self.delta_safe = qp_cfg.delta_safe

    def test_barrier_is_finite_at_warm_start(self):
        from constraint_layers import build_barrier_log_terms
        import casadi as ca

        n_x = self.dynamics.n_x
        n_int = self.n_int
        m = len(self.path_regions)
        x_node_syms = [
            [ca.MX.sym(f'x_{i}_{k}', n_x) for k in range(n_int + 1)]
            for i in range(m)
        ]
        delta_syms = [ca.MX.sym(f'delta_{i}') for i in range(m)]

        B_sym = build_barrier_log_terms(
            self.path_regions, self.graph, self.dynamics,
            x_node_syms, delta_syms, n_int, self.delta_safe
        )
        fn = ca.Function('B',
            [v for seg in x_node_syms for v in seg] + delta_syms, [B_sym])

        args = [self.ws.x_nodes[i][k] for i in range(m) for k in range(n_int + 1)]
        args += self.ws.delta_list
        val = float(fn(*args))
        self.assertTrue(np.isfinite(val), f"Barrier is not finite: {val}")
        self.assertGreater(val, 0.0, "Barrier should be positive at interior point")

    def test_lipschitz_safety_gap(self):
        from constraint_layers import compute_lipschitz_safety_gap
        gap = compute_lipschitz_safety_gap(
            self.dynamics, self.path_regions, self.graph,
            np.array(self.ws.delta_list), self.n_int
        )
        self.assertGreater(gap, 0.0)
        self.assertTrue(np.isfinite(gap))
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_constraint_layers.py::BarrierLogTermsTests -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'build_barrier_log_terms'`.

- [ ] **Step 3: Add `build_barrier_log_terms` and `compute_lipschitz_safety_gap` to `constraint_layers.py`**

At the end of `demo/constraint_layers.py`, add:

```python
def build_barrier_log_terms(
    path_regions,
    graph,
    dynamics,
    x_node_vars,
    delta_list,
    n_int: int,
    delta_safe: float,
):
    """
    Compute B = -Σ_{i,j,k} h_k * log(s_{i,j,k}) as a CasADi expression.

    s_{i,j,k} = b_{i,j} - delta_safe - a_{i,j}^T @ pos(x_{i,k})
    Trapezoidal weights: h_k = Δ_i/(2*n_int) at endpoints, Δ_i/n_int at interior.
    """
    import casadi as ca
    B = ca.MX(0.0)

    for seg_idx, region_idx in enumerate(path_regions):
        region = graph.regions[region_idx]
        delta_i = delta_list[seg_idx]
        states_i = x_node_vars[seg_idx]

        for k in range(n_int + 1):
            if k == 0 or k == n_int:
                h_k = delta_i / (2.0 * n_int)
            else:
                h_k = delta_i / n_int

            x_k = states_i[k]
            pos_k = dynamics.project_to_position_casadi(x_k)

            for j in range(region.A.shape[0]):
                a_j = region.A[j, :]
                b_j = float(region.b[j])
                s_ijk = b_j - delta_safe - float(a_j[0]) * pos_k[0] - float(a_j[1]) * pos_k[1]
                B = B - h_k * ca.log(s_ijk)

    return B


def compute_lipschitz_safety_gap(
    dynamics,
    path_regions,
    graph,
    delta_arr,
    n_int: int,
) -> float:
    """
    Return L_s * h_rk4_max / 2 — worst-case safety slack deviation between nodes.

    L_s = max_{i,j} ||a_{i,j}||_2 * v_max  (from dynamics.compute_lipschitz_bound)
    h_rk4_max = max(Δ_i) / n_int
    """
    import numpy as np
    A_list = [graph.regions[ri].A for ri in path_regions]
    L_s = dynamics.compute_lipschitz_bound(A_list)
    h_rk4_max = float(np.max(delta_arr)) / n_int
    return L_s * h_rk4_max / 2.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_constraint_layers.py::BarrierLogTermsTests -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/ --tb=short 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git -C /home/khoa/ws/GCS_MMS_demo add demo/constraint_layers.py tests/test_constraint_layers.py
git -C /home/khoa/ws/GCS_MMS_demo commit -m "feat: add build_barrier_log_terms and compute_lipschitz_safety_gap"
```

---

## Task 7: `barrier_dms.py` — Barrier schedule + failure taxonomy

**Files:**
- Create: `demo/barrier_dms.py`
- Create: `tests/test_barrier_dms.py`

- [ ] **Step 1: Create `tests/test_barrier_dms.py`** (schedule + failure tests)

```python
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from barrier_dms import BarrierSchedule, classify_dms_failure, FailureCode


class BarrierScheduleTests(unittest.TestCase):
    def test_mu0_formula(self):
        sched = BarrierSchedule.from_warm_start(f0=10.0, B0=100.0, alpha=0.1, tau=0.1, mu_min=1e-5)
        # mu_0 = clip(0.1 * 10 / 100, 1e-5, 1.0) = clip(0.01, 1e-5, 1.0) = 0.01
        self.assertAlmostEqual(sched.mu_schedule[0], 0.01, places=10)

    def test_n_mu_formula(self):
        sched = BarrierSchedule.from_warm_start(f0=10.0, B0=100.0, alpha=0.1, tau=0.1, mu_min=1e-5)
        # mu_0=0.01, mu_min=1e-5, tau=0.1
        # N_mu = ceil(log(1e-5 / 0.01) / log(0.1)) = ceil(log10(1000)) = 3
        self.assertEqual(sched.n_mu, 3)

    def test_schedule_is_decreasing(self):
        sched = BarrierSchedule.from_warm_start(f0=5.0, B0=50.0, alpha=0.1, tau=0.1, mu_min=1e-5)
        for i in range(len(sched.mu_schedule) - 1):
            self.assertLess(sched.mu_schedule[i + 1], sched.mu_schedule[i])

    def test_final_level_reaches_mu_min(self):
        sched = BarrierSchedule.from_warm_start(f0=10.0, B0=100.0, alpha=0.1, tau=0.1, mu_min=1e-5)
        self.assertLessEqual(sched.mu_schedule[-1], 1e-5 + 1e-12)

    def test_degenerate_f0_fallback(self):
        sched = BarrierSchedule.from_warm_start(
            f0=0.0, B0=100.0, alpha=0.1, tau=0.1, mu_min=1e-5,
            delta_safe=0.02, w_T=1.0
        )
        self.assertGreater(sched.mu_schedule[0], 0.0)

    def test_mu_min_at_least_one_level(self):
        sched = BarrierSchedule.from_warm_start(f0=1e-4, B0=1.0, alpha=0.1, tau=0.1, mu_min=1e-5)
        self.assertGreaterEqual(sched.n_mu, 1)

    def test_fixed_schedule_default(self):
        sched = BarrierSchedule.fixed()
        self.assertEqual(sched.mu_schedule, [1.0, 0.5, 0.1, 0.01])
        self.assertEqual(sched.n_mu, 4)

    def test_fixed_schedule_is_decreasing(self):
        sched = BarrierSchedule.fixed()
        for i in range(len(sched.mu_schedule) - 1):
            self.assertLess(sched.mu_schedule[i + 1], sched.mu_schedule[i])


class FailureClassificationTests(unittest.TestCase):
    def test_restoration_phase(self):
        code = classify_dms_failure("Restoration_Failed", defect_norm=1.0, coupling_gap=0.5)
        self.assertEqual(code, FailureCode.IPOPT_RESTORATION)

    def test_max_iter(self):
        code = classify_dms_failure("Maximum_Iterations_Exceeded", defect_norm=0.01, coupling_gap=0.01)
        self.assertEqual(code, FailureCode.IPOPT_MAX_ITER)

    def test_large_connection_gap(self):
        code = classify_dms_failure("Solve_Succeeded", defect_norm=0.0, coupling_gap=1e-2)
        self.assertEqual(code, FailureCode.CONNECTION_GAP_LARGE)

    def test_success_returns_none(self):
        code = classify_dms_failure("Solve_Succeeded", defect_norm=1e-8, coupling_gap=1e-8)
        self.assertIsNone(code)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_barrier_dms.py::BarrierScheduleTests tests/test_barrier_dms.py::FailureClassificationTests -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'barrier_dms'`.

- [ ] **Step 3: Create `demo/barrier_dms.py`** (schedule + failure taxonomy only; solver in Task 8)

```python
"""
barrier_dms.py - Barrier continuation core for Centroid-Refine-DMS.

Contains:
  - BarrierSchedule: mu schedule computation (Fiacco-McCormick)
  - FailureCode: failure taxonomy enum
  - classify_dms_failure: maps IPOPT status -> FailureCode
  - FailureRecord: structured failure log entry
  - BarrierDMSSolver: multiple-shooting NLP with log-barrier (Task 8)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class FailureCode(str, Enum):
    NARROW_INTERFACE    = "NARROW_INTERFACE"
    QP_INFEASIBLE       = "QP_INFEASIBLE"
    WARM_START_VIOLATION = "WARM_START_VIOLATION"
    IPOPT_RESTORATION   = "IPOPT_RESTORATION"
    IPOPT_MAX_ITER      = "IPOPT_MAX_ITER"
    CONTROL_SATURATION  = "CONTROL_SATURATION"
    DURATION_SATURATION = "DURATION_SATURATION"
    CONNECTION_GAP_LARGE = "CONNECTION_GAP_LARGE"
    KKT_DEGENERATE      = "KKT_DEGENERATE"
    NO_PATH_FOUND       = "NO_PATH_FOUND"


@dataclass
class FailureRecord:
    path: Tuple[str, ...]
    code: FailureCode
    barrier_level: int = -1
    ipopt_status: str = ""
    defect_norm: float = float("nan")
    coupling_gap: float = float("nan")


@dataclass
class BarrierSchedule:
    mu_schedule: List[float]
    n_mu: int

    # Default fixed barrier schedule (interior-point: large mu first, decreasing)
    DEFAULT_LEVELS: List[float] = [1.0, 0.5, 0.1, 0.01]

    @classmethod
    def fixed(cls, levels: List[float] = None) -> "BarrierSchedule":
        """
        Create a fixed barrier schedule.

        Interior-point starts with large mu (loose safety, fast convergence) and
        decreases. Each level warm-starts from the previous solution.
        The schedule stops early if the Lipschitz safety certificate passes.
        """
        if levels is None:
            levels = list(cls.DEFAULT_LEVELS)
        return cls(mu_schedule=levels, n_mu=len(levels))

    @classmethod
    def from_warm_start(
        cls,
        f0: float,
        B0: float,
        alpha: float = 0.1,
        tau: float = 0.1,
        mu_min: float = 1e-5,
        delta_safe: float = 0.02,
        w_T: float = 1.0,
    ) -> "BarrierSchedule":
        """
        Legacy Fiacco-McCormick schedule (kept for ablation A4 comparison).
        Production path uses BarrierSchedule.fixed() instead.

        WARNING: B0 = -Σ hₖ log s_{i,j,k} may be negative when some slacks > 1,
        making mu_0 undefined. The fallback (delta_safe * w_T) is used in that case.
        """
        if f0 <= 0.0 or not (B0 > 0.0):
            mu_0 = delta_safe * w_T
        else:
            mu_0 = float(np.clip(alpha * f0 / B0, mu_min, 1.0))

        if mu_0 <= mu_min:
            n_mu = 1
        else:
            n_mu = max(1, math.ceil(math.log(mu_min / mu_0) / math.log(tau)))

        schedule = [mu_0 * (tau ** r) for r in range(n_mu)]
        if schedule and schedule[-1] > mu_min:
            schedule[-1] = mu_min

        return cls(mu_schedule=schedule, n_mu=n_mu)


_IPOPT_MAX_ITER_STATUSES = {
    "Maximum_Iterations_Exceeded",
    "max_iter_exceeded",
}

_IPOPT_RESTORATION_STATUSES = {
    "Restoration_Failed",
    "restoration_failed",
    "Error_In_Step_Computation",
}

_IPOPT_DEGENERATE_STATUSES = {
    "Converged_To_A_Locally_Infeasible_Point",
    "Infeasible_Problem_Detected",
    "locally_infeasible",
}

_COUPLING_GAP_THRESHOLD = 1e-4


def classify_dms_failure(
    ipopt_status: str,
    defect_norm: float,
    coupling_gap: float,
) -> Optional[FailureCode]:
    """
    Map IPOPT return status + residuals to a FailureCode.
    Returns None if the solve is considered successful.
    """
    if ipopt_status in _IPOPT_RESTORATION_STATUSES:
        return FailureCode.IPOPT_RESTORATION
    if ipopt_status in _IPOPT_MAX_ITER_STATUSES:
        return FailureCode.IPOPT_MAX_ITER
    if ipopt_status in _IPOPT_DEGENERATE_STATUSES:
        return FailureCode.KKT_DEGENERATE
    if coupling_gap > _COUPLING_GAP_THRESHOLD:
        return FailureCode.CONNECTION_GAP_LARGE
    if ipopt_status in {"Solve_Succeeded", "Solved_To_Acceptable_Level", "acceptable"}:
        return None
    return FailureCode.KKT_DEGENERATE
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_barrier_dms.py::BarrierScheduleTests tests/test_barrier_dms.py::FailureClassificationTests -v
```

Expected: all 10 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/ --tb=short 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git -C /home/khoa/ws/GCS_MMS_demo add demo/barrier_dms.py tests/test_barrier_dms.py
git -C /home/khoa/ws/GCS_MMS_demo commit -m "feat: add BarrierSchedule, FailureCode, classify_dms_failure to barrier_dms"
```

---

## Task 8: `barrier_dms.py` — `BarrierDMSSolver`

**Files:**
- Modify: `demo/barrier_dms.py`
- Modify: `tests/test_barrier_dms.py`

- [ ] **Step 1: Write integration tests**

Add to `tests/test_barrier_dms.py`:

```python
class BarrierDMSSolverTests(unittest.TestCase):
    def _make_simple_scenario(self):
        from convex_regions import create_regions_from_vertices_list
        from graph_builder import build_region_graph
        from dynamics import UnicycleModel
        from geometric_refiner import InterfaceQPConfig, solve_interface_refinement
        from warmstart import WarmStartConfig, generate_warm_start_ivp
        from barrier_dms import BarrierDMSSolverConfig

        regions = create_regions_from_vertices_list([
            np.array([[0, 0], [1.2, 0], [1.2, 1], [0, 1]], dtype=float),
            np.array([[0.8, 0], [2.0, 0], [2.0, 1], [0.8, 1]], dtype=float),
        ])
        x_start = np.array([0.1, 0.5, 0.0])
        x_goal  = np.array([1.9, 0.5, 0.0])
        graph = build_region_graph(regions, x_start[:2], x_goal[:2])
        path_regions = [0, 1]
        dynamics = UnicycleModel(v_max=2.0, omega_max=np.pi)

        qp_cfg = InterfaceQPConfig(delta_safe=0.02, delta_extra=0.01)
        z = solve_interface_refinement(graph, path_regions, x_start[:2], x_goal[:2], qp_cfg)
        ws_cfg = WarmStartConfig(delta_min=0.01, delta_max=10.0, n_int=10)
        ws = generate_warm_start_ivp(z, path_regions, dynamics, ws_cfg)

        solver_cfg = BarrierDMSSolverConfig(
            n_int=10, delta_safe=0.02,
            w_T=1.0, w_L=1.0, w_U=0.5, w_S=0.1,
            alpha_mu=0.1, tau=0.1, mu_min=1e-4,
            epsilon_final=1e-4,
        )
        return graph, path_regions, dynamics, x_start, x_goal, ws, solver_cfg

    def test_solver_returns_result(self):
        from barrier_dms import BarrierDMSSolver
        graph, path_regions, dynamics, x_start, x_goal, ws, cfg = self._make_simple_scenario()
        solver = BarrierDMSSolver(dynamics, cfg)
        result = solver.solve_path(graph, path_regions, x_start, x_goal, ws)
        self.assertIsNotNone(result)

    def test_successful_solve_has_finite_cost(self):
        from barrier_dms import BarrierDMSSolver
        graph, path_regions, dynamics, x_start, x_goal, ws, cfg = self._make_simple_scenario()
        solver = BarrierDMSSolver(dynamics, cfg)
        result = solver.solve_path(graph, path_regions, x_start, x_goal, ws)
        if result.success:
            self.assertTrue(np.isfinite(result.cost_unbarred))
            self.assertGreater(result.cost_unbarred, 0.0)
            self.assertIsNotNone(result.x_nodes_opt)
            self.assertIsNotNone(result.u_list_opt)
            self.assertIsNotNone(result.delta_opt)

    def test_safety_margins_populated(self):
        from barrier_dms import BarrierDMSSolver
        graph, path_regions, dynamics, x_start, x_goal, ws, cfg = self._make_simple_scenario()
        solver = BarrierDMSSolver(dynamics, cfg)
        result = solver.solve_path(graph, path_regions, x_start, x_goal, ws)
        if result.success:
            self.assertTrue(np.isfinite(result.s_min_sampled))
            self.assertTrue(np.isfinite(result.s_min_certified))
            self.assertLessEqual(result.s_min_certified, result.s_min_sampled)

    def test_optimality_note_no_global_claim(self):
        from barrier_dms import BarrierDMSSolver
        graph, path_regions, dynamics, x_start, x_goal, ws, cfg = self._make_simple_scenario()
        solver = BarrierDMSSolver(dynamics, cfg)
        result = solver.solve_path(graph, path_regions, x_start, x_goal, ws)
        if result.success:
            self.assertNotIn("globally optimal", result.optimality_note.lower())
            self.assertIn("KKT", result.optimality_note)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_barrier_dms.py::BarrierDMSSolverTests -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'BarrierDMSSolver'`.

- [ ] **Step 3: Append `BarrierDMSSolverConfig`, `BarrierPathResult`, and `BarrierDMSSolver` to `barrier_dms.py`**

Append to `demo/barrier_dms.py`:

```python
@dataclass
class BarrierDMSSolverConfig:
    n_int: int = 20
    delta_safe: float = 0.02
    delta_min: float = 0.01
    delta_max: float = 10.0
    w_T: float = 1.0
    w_L: float = 1.0
    w_U: float = 1.0
    w_S: float = 0.2
    alpha_mu: float = 0.1
    tau: float = 0.1
    mu_min: float = 1e-5
    epsilon_final: float = 1e-6
    ipopt_max_iter_schedule: Tuple[int, ...] = (300, 200, 150, 100, 100)


@dataclass
class BarrierPathResult:
    success: bool
    failure_code: Optional[FailureCode] = None
    cost_unbarred: float = float("nan")
    x_nodes_opt: Optional[List[np.ndarray]] = None
    u_list_opt: Optional[List[np.ndarray]] = None
    delta_opt: Optional[List[float]] = None
    defect_norm: float = float("nan")
    coupling_gap: float = float("nan")
    s_min_sampled: float = float("nan")
    s_min_certified: float = float("nan")
    lipschitz_gap: float = float("nan")
    n_barrier_levels_run: int = 0
    n_ipopt_iters_total: int = 0
    optimality_note: str = (
        "KKT-feasible under LICQ+SOSC; no global optimality certificate."
    )
    failure_log: List[FailureRecord] = field(default_factory=list)


class BarrierDMSSolver:
    """
    Multiple-shooting DMS NLP with log-barrier on safety constraints.

    Runs N_mu barrier levels (full continuation) from a provided warm-start.
    Decision variables at every RK4 node; defect constraints as hard equalities.
    Returns a BarrierPathResult.
    """

    def __init__(self, dynamics, config: BarrierDMSSolverConfig):
        import casadi as ca
        self.dynamics = dynamics
        self.config = config
        n_x = dynamics.n_x
        n_u = dynamics.n_u

        x_sym = ca.MX.sym('x_rk4', n_x)
        u_sym = ca.MX.sym('u_rk4', n_u)
        h_sym = ca.MX.sym('h_rk4', 1)
        f = dynamics.f_casadi
        k1 = h_sym * f(x_sym, u_sym)
        k2 = h_sym * f(x_sym + k1 / 2, u_sym)
        k3 = h_sym * f(x_sym + k2 / 2, u_sym)
        k4 = h_sym * f(x_sym + k3, u_sym)
        x_next = x_sym + (k1 + 2*k2 + 2*k3 + k4) / 6
        self._rk4_step = ca.Function('rk4', [x_sym, u_sym, h_sym], [x_next])

    def solve_path(self, graph, path_regions, x_start, x_goal, warm_start) -> BarrierPathResult:
        """Run barrier continuation for a fixed path and warm-start."""
        import casadi as ca
        from constraint_layers import build_barrier_log_terms, compute_lipschitz_safety_gap

        cfg = self.config
        dynamics = self.dynamics
        n_x = dynamics.n_x
        n_u = dynamics.n_u
        n_int = cfg.n_int
        m = len(path_regions)
        failure_log: List[FailureRecord] = []

        f0, B0 = self._evaluate_f0_B0(graph, path_regions, warm_start)
        schedule = BarrierSchedule.from_warm_start(
            f0=f0, B0=B0, alpha=cfg.alpha_mu, tau=cfg.tau,
            mu_min=cfg.mu_min, delta_safe=cfg.delta_safe, w_T=cfg.w_T
        )

        (x_all_sym, lbx, ubx, x0_flat,
         g_sym, lbg, ubg,
         cost_no_barrier, x_node_syms, u_syms, delta_syms) = self._build_nlp_symbols(
            graph, path_regions, x_start, x_goal
        )

        # Fill warm-start into x0_flat
        x0_flat = self._flatten_warm_start(warm_start, m, n_x, n_u, n_int)

        x_w = x0_flat.copy()
        n_iters_total = 0

        for r, mu_r in enumerate(schedule.mu_schedule):
            eps_r = max(cfg.epsilon_final, mu_r)
            max_iter_r = (
                cfg.ipopt_max_iter_schedule[r]
                if r < len(cfg.ipopt_max_iter_schedule)
                else cfg.ipopt_max_iter_schedule[-1]
            )

            B_sym = build_barrier_log_terms(
                path_regions, graph, dynamics,
                x_node_syms, delta_syms, n_int, cfg.delta_safe
            )
            obj_full = cost_no_barrier - mu_r * B_sym

            nlp = {'x': x_all_sym, 'f': obj_full, 'g': g_sym}
            opts = {
                'ipopt.max_iter': max_iter_r,
                'ipopt.tol': eps_r,
                'ipopt.print_level': 0,
                'print_time': False,
            }
            solver = ca.nlpsol('barrier_dms', 'ipopt', nlp, opts)
            sol = solver(x0=x_w, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            stats = solver.stats()
            ipopt_status = stats.get('return_status', 'unknown')
            n_iters_total += int(stats.get('iter_count', 0))

            x_w = np.array(sol['x']).flatten()

            g_val = np.array(sol['g']).flatten()
            n_defect = m * n_x * n_int
            n_coupling = (m - 1) * n_x + 2 * n_x

            defect_residuals = g_val[:n_defect] if len(g_val) >= n_defect else g_val
            coupling_residuals = g_val[n_defect:] if len(g_val) > n_defect else np.array([0.0])

            defect_norm_r  = float(np.max(np.abs(defect_residuals))) if defect_residuals.size else 0.0
            coupling_gap_r = float(np.max(np.abs(coupling_residuals))) if coupling_residuals.size else 0.0

            failure_code = classify_dms_failure(ipopt_status, defect_norm_r, coupling_gap_r)

            if failure_code is not None:
                failure_log.append(FailureRecord(
                    path=(), code=failure_code, barrier_level=r,
                    ipopt_status=ipopt_status,
                    defect_norm=defect_norm_r, coupling_gap=coupling_gap_r,
                ))
                return BarrierPathResult(
                    success=False, failure_code=failure_code,
                    n_barrier_levels_run=r + 1, n_ipopt_iters_total=n_iters_total,
                    defect_norm=defect_norm_r, coupling_gap=coupling_gap_r,
                    failure_log=failure_log,
                )

        return self._parse_final_solution(x_w, graph, path_regions, cfg, schedule.n_mu, n_iters_total)

    def _evaluate_f0_B0(self, graph, path_regions, ws) -> Tuple[float, float]:
        cfg = self.config
        f0 = 0.0
        B0 = 0.0
        n_int = cfg.n_int
        for seg_i, region_idx in enumerate(path_regions):
            region = graph.regions[region_idx]
            delta_i = ws.delta_list[seg_i]
            u_i = ws.u_list[seg_i]
            f0 += cfg.w_T * delta_i
            for k in range(n_int + 1):
                h_k = delta_i / (2 * n_int) if (k == 0 or k == n_int) else delta_i / n_int
                x_k = ws.x_nodes[seg_i][k]
                pos_k = self.dynamics.project_to_position(x_k)
                vel_k = self.dynamics.position_velocity(x_k, u_i)
                f0 += h_k * (cfg.w_L * float(np.dot(vel_k, vel_k)) +
                              cfg.w_U * float(np.dot(u_i, u_i)))
                for j in range(region.A.shape[0]):
                    s_ijk = float(region.b[j]) - cfg.delta_safe - float(region.A[j] @ pos_k)
                    if s_ijk <= 0:
                        s_ijk = 1e-9
                    B0 -= h_k * np.log(s_ijk)
        return f0, B0

    def _flatten_warm_start(self, ws, m, n_x, n_u, n_int) -> np.ndarray:
        """Flatten WarmStart into the NLP decision-variable order."""
        flat = []
        for i in range(m):
            for k in range(n_int + 1):
                flat.extend(ws.x_nodes[i][k].tolist())
            flat.extend(ws.u_list[i].tolist())
            flat.append(ws.delta_list[i])
        return np.array(flat, dtype=float)

    def _build_nlp_symbols(self, graph, path_regions, x_start, x_goal):
        """Build CasADi symbolic NLP (decision vars, constraints, unbarred cost)."""
        import casadi as ca
        cfg = self.config
        dynamics = self.dynamics
        n_x = dynamics.n_x
        n_u = dynamics.n_u
        n_int = cfg.n_int
        m = len(path_regions)

        u_lb, u_ub = dynamics.control_bounds()
        if hasattr(graph, 'regions') and graph.regions:
            all_verts = np.vstack([r.vertices for r in graph.regions])
            pos_lb = np.array([np.min(all_verts[:, 0]) - 1e-6,
                               np.min(all_verts[:, 1]) - 1e-6])
            pos_ub = np.array([np.max(all_verts[:, 0]) + 1e-6,
                               np.max(all_verts[:, 1]) + 1e-6])
        else:
            pos_lb, pos_ub = np.array([-100.0, -100.0]), np.array([100.0, 100.0])
        state_lb, state_ub = dynamics.state_bounds(pos_lb, pos_ub)

        x_node_syms = []
        u_syms      = []
        delta_syms  = []
        all_vars    = []
        lbx, ubx    = [], []

        for i in range(m):
            nodes_i = []
            for k in range(n_int + 1):
                xk = ca.MX.sym(f'x_{i}_{k}', n_x)
                nodes_i.append(xk)
                all_vars.append(xk)
                lbx.extend(state_lb.tolist())
                ubx.extend(state_ub.tolist())
            x_node_syms.append(nodes_i)

            u_i = ca.MX.sym(f'u_{i}', n_u)
            u_syms.append(u_i)
            all_vars.append(u_i)
            lbx.extend(u_lb.tolist())
            ubx.extend(u_ub.tolist())

            d_i = ca.MX.sym(f'delta_{i}')
            delta_syms.append(d_i)
            all_vars.append(d_i)
            lbx.append(cfg.delta_min)
            ubx.append(cfg.delta_max)

        x_all_sym = ca.vertcat(*all_vars)
        x0_flat = np.zeros(len(lbx))  # placeholder; real values set by caller

        g_list, lbg, ubg = [], [], []

        for i in range(m):
            d_i = delta_syms[i]
            h_i = d_i / n_int
            nodes_i = x_node_syms[i]
            u_i = u_syms[i]
            for k in range(n_int):
                x_next_rk4 = self._rk4_step(nodes_i[k], u_i, h_i)
                defect = nodes_i[k + 1] - x_next_rk4
                g_list.append(defect)
                lbg.extend([0.0] * n_x)
                ubg.extend([0.0] * n_x)

        for i in range(m - 1):
            coupling = x_node_syms[i][n_int] - x_node_syms[i + 1][0]
            g_list.append(coupling)
            lbg.extend([0.0] * n_x)
            ubg.extend([0.0] * n_x)

        g_list.append(x_node_syms[0][0] - ca.DM(x_start))
        lbg.extend([0.0] * n_x)
        ubg.extend([0.0] * n_x)
        g_list.append(x_node_syms[-1][n_int] - ca.DM(x_goal))
        lbg.extend([0.0] * n_x)
        ubg.extend([0.0] * n_x)

        g_sym = ca.vertcat(*g_list)

        cost_no_barrier = ca.MX(0.0)
        for i in range(m):
            d_i = delta_syms[i]
            u_i = u_syms[i]
            cost_no_barrier = cost_no_barrier + cfg.w_T * d_i
            nodes_i = x_node_syms[i]
            for k in range(n_int + 1):
                h_k = d_i / (2 * n_int) if (k == 0 or k == n_int) else d_i / n_int
                vel_k = dynamics.position_velocity_casadi(nodes_i[k], u_i)
                cost_no_barrier = (cost_no_barrier
                    + h_k * cfg.w_L * ca.dot(vel_k, vel_k)
                    + h_k * cfg.w_U * ca.dot(u_i, u_i))
        if m > 1:
            for i in range(m - 1):
                jump = u_syms[i] - u_syms[i + 1]
                cost_no_barrier = cost_no_barrier + cfg.w_S * ca.dot(jump, jump)

        return (x_all_sym, lbx, ubx, x0_flat,
                g_sym, lbg, ubg,
                cost_no_barrier, x_node_syms, u_syms, delta_syms)

    def _parse_final_solution(self, x_opt, graph, path_regions, cfg, n_mu, n_iters_total) -> BarrierPathResult:
        from constraint_layers import compute_lipschitz_safety_gap
        dynamics = self.dynamics
        n_x = dynamics.n_x
        n_u = dynamics.n_u
        n_int = cfg.n_int
        m = len(path_regions)

        x_nodes_opt: List[np.ndarray] = []
        u_list_opt:  List[np.ndarray] = []
        delta_opt:   List[float]      = []

        offset = 0
        for i in range(m):
            seg_states = np.zeros((n_int + 1, n_x))
            for k in range(n_int + 1):
                seg_states[k] = x_opt[offset:offset + n_x]
                offset += n_x
            x_nodes_opt.append(seg_states)
            u_list_opt.append(x_opt[offset:offset + n_u].copy())
            offset += n_u
            delta_opt.append(float(x_opt[offset]))
            offset += 1

        defect_norms = []
        for i in range(m):
            h_i = delta_opt[i] / n_int
            for k in range(n_int):
                x_k   = x_nodes_opt[i][k]
                x_kp1 = x_nodes_opt[i][k + 1]
                u_i   = u_list_opt[i]
                rk4_val = np.array(self._rk4_step(x_k, u_i, h_i)).flatten()
                defect_norms.append(float(np.linalg.norm(x_kp1 - rk4_val)))
        defect_norm = max(defect_norms) if defect_norms else 0.0

        coupling_gaps = []
        for i in range(m - 1):
            gap = np.linalg.norm(x_nodes_opt[i][n_int] - x_nodes_opt[i + 1][0])
            coupling_gaps.append(float(gap))
        coupling_gap = max(coupling_gaps) if coupling_gaps else 0.0

        cost = 0.0
        for i in range(m):
            d = delta_opt[i]
            u = u_list_opt[i]
            cost += cfg.w_T * d
            for k in range(n_int + 1):
                h_k = d / (2 * n_int) if (k == 0 or k == n_int) else d / n_int
                vel = dynamics.position_velocity(x_nodes_opt[i][k], u)
                cost += float(h_k) * (cfg.w_L * float(np.dot(vel, vel)) +
                                      cfg.w_U * float(np.dot(u, u)))
        if m > 1:
            for i in range(m - 1):
                jump = u_list_opt[i] - u_list_opt[i + 1]
                cost += cfg.w_S * float(np.dot(jump, jump))

        s_min_sampled = float("inf")
        for i, region_idx in enumerate(path_regions):
            region = graph.regions[region_idx]
            for k in range(n_int + 1):
                pos_k = dynamics.project_to_position(x_nodes_opt[i][k])
                slacks = region.b - cfg.delta_safe - region.A @ pos_k
                s_min_sampled = min(s_min_sampled, float(np.min(slacks)))

        lipschitz_gap = compute_lipschitz_safety_gap(
            dynamics, path_regions, graph,
            np.array(delta_opt), n_int
        )
        s_min_certified = s_min_sampled - lipschitz_gap

        return BarrierPathResult(
            success=True,
            cost_unbarred=cost,
            x_nodes_opt=x_nodes_opt,
            u_list_opt=u_list_opt,
            delta_opt=delta_opt,
            defect_norm=defect_norm,
            coupling_gap=coupling_gap,
            s_min_sampled=s_min_sampled,
            s_min_certified=s_min_certified,
            lipschitz_gap=lipschitz_gap,
            n_barrier_levels_run=n_mu,
            n_ipopt_iters_total=n_iters_total,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_barrier_dms.py::BarrierDMSSolverTests -v --timeout=120
```

Expected: all 4 tests pass (IPOPT solve may take 10–30 s).

- [ ] **Step 5: Run full test suite**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/ --tb=short -q 2>&1 | tail -15
```

- [ ] **Step 6: Commit**

```bash
git -C /home/khoa/ws/GCS_MMS_demo add demo/barrier_dms.py tests/test_barrier_dms.py
git -C /home/khoa/ws/GCS_MMS_demo commit -m "feat: add BarrierDMSSolver with multiple-shooting NLP and barrier continuation"
```

---

## Task 9: `optimizer.py` — `CentroidRefineDMSConfig` + new fields + `CentroidRefineDMSSolver`

**Files:**
- Modify: `demo/optimizer.py`
- Create: `tests/test_centroid_refine_dms.py`

- [ ] **Step 1: Create `tests/test_centroid_refine_dms.py`**

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from convex_regions import create_regions_from_vertices_list
from graph_builder import build_region_graph
from dynamics import UnicycleModel
from optimizer import CentroidRefineDMSConfig, CentroidRefineDMSSolver, OptimizationResult


def _make_two_region_scenario():
    regions = create_regions_from_vertices_list([
        np.array([[0, 0], [1.2, 0], [1.2, 1], [0, 1]], dtype=float),
        np.array([[0.8, 0], [2.0, 0], [2.0, 1], [0.8, 1]], dtype=float),
    ])
    x_start = np.array([0.1, 0.5, 0.0])
    x_goal  = np.array([1.9, 0.5, 0.0])
    graph = build_region_graph(regions, x_start[:2], x_goal[:2])
    dynamics = UnicycleModel(v_max=2.0)
    return graph, dynamics, x_start, x_goal


class CentroidRefineDMSConfigTests(unittest.TestCase):
    def test_default_config_valid(self):
        cfg = CentroidRefineDMSConfig()
        self.assertEqual(cfg.mode, "first_feasible")
        self.assertGreater(cfg.mu_min, 0.0)
        self.assertGreater(cfg.epsilon_final, 0.0)
        self.assertTrue(cfg.use_centroid_cost)
        self.assertTrue(cfg.use_interface_qp)
        self.assertTrue(cfg.use_log_barrier)

    def test_ablation_flags_exist(self):
        cfg = CentroidRefineDMSConfig(
            use_centroid_cost=False,
            use_interface_qp=False,
            use_log_barrier=False,
            use_barrier_continuation=False,
            use_inexact_tolerance=False,
        )
        self.assertFalse(cfg.use_centroid_cost)


class OptimizationResultNewFieldsTests(unittest.TestCase):
    def test_new_fields_exist_with_defaults(self):
        result = OptimizationResult(
            success=False, path=[], path_regions=[],
            total_cost=0.0, solve_time=0.0, n_paths_evaluated=0
        )
        self.assertTrue(np.isnan(result.certified_safety_margin))
        self.assertTrue(np.isnan(result.lipschitz_gap))
        self.assertEqual(result.safety_certification, "NOT_SET")
        self.assertEqual(result.lb_geometric, 0.0)
        self.assertEqual(result.optimality_gap, float("inf"))
        self.assertEqual(result.n_barrier_levels, 0)


class CentroidRefineDMSSolverTests(unittest.TestCase):
    def test_solver_returns_result(self):
        graph, dynamics, x_start, x_goal = _make_two_region_scenario()
        cfg = CentroidRefineDMSConfig(
            n_int=8, mu_min=1e-3, epsilon_final=1e-3, time_limit_s=60.0
        )
        solver = CentroidRefineDMSSolver(graph, dynamics, cfg)
        result = solver.solve(x_start, x_goal)
        self.assertIsInstance(result, OptimizationResult)

    def test_mandatory_disclaimer_set(self):
        graph, dynamics, x_start, x_goal = _make_two_region_scenario()
        cfg = CentroidRefineDMSConfig(n_int=8, mu_min=1e-3, epsilon_final=1e-3)
        solver = CentroidRefineDMSSolver(graph, dynamics, cfg)
        result = solver.solve(x_start, x_goal)
        self.assertIn("KKT", result.global_optimality_claim)
        self.assertNotIn("globally optimal", result.global_optimality_claim.lower())

    def test_successful_result_has_both_safety_margins(self):
        graph, dynamics, x_start, x_goal = _make_two_region_scenario()
        cfg = CentroidRefineDMSConfig(n_int=8, mu_min=1e-3, epsilon_final=1e-3)
        solver = CentroidRefineDMSSolver(graph, dynamics, cfg)
        result = solver.solve(x_start, x_goal)
        if result.success:
            self.assertTrue(np.isfinite(result.min_safety_margin))
            self.assertTrue(np.isfinite(result.certified_safety_margin))
            self.assertLessEqual(result.certified_safety_margin, result.min_safety_margin)
            self.assertNotEqual(result.safety_certification, "NOT_SET")

    def test_failure_log_is_list(self):
        graph, dynamics, x_start, x_goal = _make_two_region_scenario()
        cfg = CentroidRefineDMSConfig(n_int=5, mu_min=1e-3)
        solver = CentroidRefineDMSSolver(graph, dynamics, cfg)
        result = solver.solve(x_start, x_goal)
        self.assertIsInstance(result.failure_log, list)

    def test_solver_mode_dispatch(self):
        from optimizer import create_integrated_optimizer_from_config, CentroidRefineDMSSolver
        graph, dynamics, x_start, x_goal = _make_two_region_scenario()
        cfg_dict = {
            'optimizer': {'solver_mode': 'centroid_refine_dms'},
            'dynamics': {},
            'shooting': {},
            'cost': {},
            'control': {},
            'centroid_refine_dms': {'n_int': 5, 'mu_min': 1e-3},
        }
        solver = create_integrated_optimizer_from_config(graph, dynamics, cfg_dict)
        self.assertIsInstance(solver, CentroidRefineDMSSolver)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_centroid_refine_dms.py -v 2>&1 | tail -15
```

Expected: `ImportError: cannot import name 'CentroidRefineDMSConfig'`.

- [ ] **Step 3: Add `CentroidRefineDMSConfig` to `optimizer.py`**

After the `OptimizationConfig` dataclass, add:

```python
@dataclass
class CentroidRefineDMSConfig:
    """Configuration for CentroidRefineDMSSolver."""
    gamma_w: float = 1.0
    gamma_h: float = 0.64
    alpha_s: float = 0.0
    delta_safe: float = 0.02
    delta_extra: float = 0.01
    v_nom_fraction: float = 0.5
    alpha_mu: float = 0.1
    tau: float = 0.1
    mu_min: float = 1e-5
    n_int: int = 20
    delta_min: float = 0.01
    delta_max: float = 10.0
    w_T: float = 1.0
    w_L: float = 1.0
    w_U: float = 1.0
    w_S: float = 0.2
    epsilon_final: float = 1e-6
    epsilon_gap: float = 0.05
    time_limit_s: float = 60.0
    mode: str = "first_feasible"  # "first_feasible" | "anytime"
    use_centroid_cost: bool = True
    use_interface_qp: bool = True
    use_log_barrier: bool = True
    use_barrier_continuation: bool = True
    use_inexact_tolerance: bool = True
```

- [ ] **Step 4: Add new fields to `OptimizationResult`**

In the `OptimizationResult` dataclass, after `formulation_mode`, add:

```python
    certified_safety_margin: float = float("nan")
    lipschitz_gap: float = float("nan")
    safety_certification: str = "NOT_SET"
    lb_geometric: float = 0.0
    optimality_gap: float = float("inf")
    n_barrier_levels: int = 0
    failure_log: list = field(default_factory=list)
```

- [ ] **Step 5: Add `CentroidRefineDMSSolver` to `optimizer.py`**

Add before `create_integrated_optimizer_from_config`:

```python
class CentroidRefineDMSSolver:
    """
    Centroid-Refine-DMS solver: first-feasible motion planner.

    Stages per path candidate:
      A. Chebyshev composite K-shortest path enumeration
      B. Interface QP refinement
      C. Warm-start IVP generation
      D+E. Barrier continuation DMS NLP
      F. Diagnostics + UB update + termination check
      G. Failure classification + blacklist
    """

    MANDATORY_DISCLAIMER = (
        "LB is a geometric lower bound on time component only. "
        "optimality_gap is NOT a certificate for the full DMS objective. "
        "No global optimality claim is made. "
        "The reported solution is a KKT point of the fixed-path "
        "barrier-augmented NLP under LICQ + SOSC."
    )

    def __init__(self, graph, dynamics, config: CentroidRefineDMSConfig):
        self.graph = graph
        self.dynamics = dynamics
        self.config = config

    def solve(self, x_start: np.ndarray, x_goal: np.ndarray) -> OptimizationResult:
        import time as _time
        from graph_builder import (
            add_composite_costs_to_graph, k_shortest_paths_generator, SOURCE, TARGET
        )
        from geometric_refiner import (
            InterfaceQPConfig, NarrowInterfaceError, InterfaceQPInfeasible,
            solve_interface_refinement
        )
        from warmstart import WarmStartConfig, generate_warm_start_ivp
        from barrier_dms import (
            BarrierDMSSolver, BarrierDMSSolverConfig,
            FailureCode, FailureRecord
        )

        cfg = self.config
        t_wall_start = _time.time()

        if cfg.use_centroid_cost:
            add_composite_costs_to_graph(
                self.graph, x_start[:cfg.n_int and 2], x_goal[:2],
                cfg.gamma_w, cfg.gamma_h
            )
            weight = 'composite_cost'
        else:
            weight = None

        lb_geom = self._compute_geometric_lb(x_start, x_goal)

        qp_cfg = InterfaceQPConfig(
            delta_safe=cfg.delta_safe,
            delta_extra=cfg.delta_extra,
            lambda_s=cfg.alpha_s,
        )
        ws_cfg = WarmStartConfig(
            delta_min=cfg.delta_min, delta_max=cfg.delta_max,
            v_nom_fraction=cfg.v_nom_fraction, n_int=cfg.n_int,
        )
        barrier_cfg = BarrierDMSSolverConfig(
            n_int=cfg.n_int, delta_safe=cfg.delta_safe,
            delta_min=cfg.delta_min, delta_max=cfg.delta_max,
            w_T=cfg.w_T, w_L=cfg.w_L, w_U=cfg.w_U, w_S=cfg.w_S,
            alpha_mu=cfg.alpha_mu, tau=cfg.tau, mu_min=cfg.mu_min,
            epsilon_final=cfg.epsilon_final,
        )
        barrier_solver = BarrierDMSSolver(self.dynamics, barrier_cfg)

        tried_paths: set = set()
        UB = float("inf")
        best_result = None
        all_failure_log: list = []

        try:
            path_gen = k_shortest_paths_generator(self.graph, SOURCE, TARGET, weight=weight)
        except Exception:
            return OptimizationResult(
                success=False, path=[], path_regions=[],
                total_cost=float("inf"),
                solve_time=_time.time() - t_wall_start,
                n_paths_evaluated=0,
                solver_status="No path in region graph",
                global_optimality_claim=self.MANDATORY_DISCLAIMER,
                lb_geometric=lb_geom,
            )

        n_evaluated = 0

        while True:
            if _time.time() - t_wall_start > cfg.time_limit_s:
                break
            try:
                path = next(path_gen)
            except StopIteration:
                break

            key = tuple(path)
            if key in tried_paths:
                continue
            tried_paths.add(key)
            n_evaluated += 1

            path_regions = self._extract_region_indices(path)
            if not path_regions:
                continue

            pos_idx = list(self.dynamics.position_indices)
            q_start = x_start[pos_idx]
            q_goal  = x_goal[pos_idx]

            if cfg.use_interface_qp:
                try:
                    z = solve_interface_refinement(self.graph, path_regions, q_start, q_goal, qp_cfg)
                except NarrowInterfaceError as e:
                    all_failure_log.append({'path': list(path), 'code': FailureCode.NARROW_INTERFACE, 'detail': str(e)})
                    continue
                except InterfaceQPInfeasible as e:
                    all_failure_log.append({'path': list(path), 'code': FailureCode.QP_INFEASIBLE, 'detail': str(e)})
                    continue
            else:
                z = self._naive_interface_points(path_regions, q_start, q_goal)

            try:
                ws = generate_warm_start_ivp(z, path_regions, self.dynamics, ws_cfg)
            except Exception as e:
                all_failure_log.append({'path': list(path), 'code': FailureCode.WARM_START_VIOLATION, 'detail': str(e)})
                continue

            if cfg.use_log_barrier:
                barrier_result = barrier_solver.solve_path(self.graph, path_regions, x_start, x_goal, ws)
            else:
                barrier_result = self._solve_with_hard_constraints(path, path_regions, x_start, x_goal)

            if barrier_result.success:
                J_star = barrier_result.cost_unbarred
                if J_star < UB:
                    UB = J_star
                    best_result = self._build_optimization_result(
                        barrier_result, path, path_regions,
                        lb_geom, UB, n_evaluated, all_failure_log,
                        _time.time() - t_wall_start
                    )
                if cfg.mode == "first_feasible":
                    break
                gap = (UB - lb_geom * cfg.w_T) / max(1.0, abs(UB))
                if gap <= cfg.epsilon_gap:
                    break
                if _time.time() - t_wall_start > cfg.time_limit_s:
                    break
            else:
                if barrier_result.failure_log:
                    for fr in barrier_result.failure_log:
                        all_failure_log.append({
                            'path': list(path), 'code': fr.code,
                            'barrier_level': fr.barrier_level,
                            'ipopt_status': fr.ipopt_status,
                            'defect_norm': fr.defect_norm,
                        })

        if best_result is None:
            return OptimizationResult(
                success=False, path=[], path_regions=[],
                total_cost=float("inf"),
                solve_time=_time.time() - t_wall_start,
                n_paths_evaluated=n_evaluated,
                solver_status="CRD: no feasible path found",
                global_optimality_claim=self.MANDATORY_DISCLAIMER,
                lb_geometric=lb_geom,
                failure_log=all_failure_log,
            )

        best_result.n_paths_evaluated = n_evaluated
        best_result.solve_time = _time.time() - t_wall_start
        return best_result

    def _extract_region_indices(self, path) -> list:
        from graph_builder import SOURCE, TARGET
        from graph_types import region_index_from_node_id
        return [
            region_index_from_node_id(node)
            for node in path
            if node not in (SOURCE, TARGET)
        ]

    def _compute_geometric_lb(self, x_start, x_goal) -> float:
        from graph_builder import SOURCE, TARGET
        from graph_types import region_index_from_node_id
        import networkx as nx

        pos_idx = list(self.dynamics.position_indices)
        u_lb, u_ub = self.dynamics.control_bounds()
        v_max = float(u_ub[0]) if u_ub.size else 1.0

        def _w(u, v, _d):
            if u == SOURCE:
                p1 = x_start[pos_idx]
            else:
                ri = region_index_from_node_id(u)
                verts = self.graph.regions[ri].vertices
                p1 = np.mean(verts, axis=0)
            if v == TARGET:
                p2 = x_goal[pos_idx]
            else:
                ri = region_index_from_node_id(v)
                verts = self.graph.regions[ri].vertices
                p2 = np.mean(verts, axis=0)
            return float(np.linalg.norm(p2 - p1))

        try:
            length = nx.shortest_path_length(self.graph.graph, SOURCE, TARGET, weight=_w)
            return length / max(v_max, 1e-6)
        except Exception:
            return 0.0

    def _naive_interface_points(self, path_regions, q_start, q_goal):
        # Returns z of shape (m+1, n_pos): z[0]=q_start, z[1..m-1]=interface midpoints, z[m]=q_goal
        m = len(path_regions)
        n_pos = self.graph.regions[path_regions[0]].A.shape[1]
        z = np.zeros((m + 1, n_pos))
        z[0] = q_start
        z[m] = q_goal
        for i in range(m - 1):  # m-1 interface points
            # Midpoint of adjacent region centroids as naive interface guess
            c_i  = np.mean(self.graph.regions[path_regions[i]].vertices,     axis=0)
            c_i1 = np.mean(self.graph.regions[path_regions[i + 1]].vertices, axis=0)
            z[i + 1] = 0.5 * (c_i + c_i1)
        return z

    def _solve_with_hard_constraints(self, path, path_regions, x_start, x_goal):
        from barrier_dms import BarrierPathResult, FailureCode
        try:
            path_solver = PathNLPSolver(self.graph, self.dynamics, OptimizationConfig(
                n_integration_steps=self.config.n_int,
                n_mesh_points=5,
                safety_margin=self.config.delta_safe,
            ))
            nlp_result = path_solver.solve_path(path, x_start, x_goal)
            if nlp_result.success:
                return BarrierPathResult(
                    success=True,
                    cost_unbarred=nlp_result.total_cost,
                    defect_norm=nlp_result.defect_norm,
                    coupling_gap=nlp_result.max_connection_gap,
                    s_min_sampled=nlp_result.min_safety_margin
                        if np.isfinite(nlp_result.min_safety_margin) else 0.0,
                    s_min_certified=float("nan"),
                    lipschitz_gap=float("nan"),
                )
        except Exception:
            pass
        from barrier_dms import BarrierPathResult, FailureCode
        return BarrierPathResult(success=False, failure_code=FailureCode.KKT_DEGENERATE)

    def _build_optimization_result(self, br, path, path_regions, lb_geom, UB, n_evaluated, failure_log, elapsed):
        gap = (UB - lb_geom * self.config.w_T) / max(1.0, abs(UB))
        safety_cert = (
            "CERTIFIED_CONTINUOUS_SAFE"
            if np.isfinite(br.s_min_certified) and br.s_min_certified > 0
            else "SAMPLED_SAFE_ONLY"
        )
        result = OptimizationResult(
            success=True,
            path=list(path),
            path_regions=path_regions,
            total_cost=UB,
            solve_time=elapsed,
            n_paths_evaluated=n_evaluated,
            solver_status="CRD: first-feasible KKT solution",
            defect_norm=br.defect_norm,
            min_safety_margin=br.s_min_sampled,
            certified_safety_margin=br.s_min_certified,
            lipschitz_gap=br.lipschitz_gap,
            safety_certification=safety_cert,
            lb_geometric=lb_geom,
            optimality_gap=gap,
            n_barrier_levels=br.n_barrier_levels_run,
            global_optimality_claim=self.MANDATORY_DISCLAIMER,
            formulation_mode="CENTROID_REFINE_DMS",
            failure_log=failure_log,
        )
        if br.x_nodes_opt is not None:
            for seg_i, region_idx in enumerate(path_regions):
                states = br.x_nodes_opt[seg_i]
                result.entry_states[region_idx] = states[0]
                result.exit_states[region_idx]  = states[-1]
                result.control_params[region_idx] = br.u_list_opt[seg_i]
                result.time_durations[region_idx] = br.delta_opt[seg_i]
        if result.time_durations:
            result.total_duration = float(sum(result.time_durations.values()))
        result.max_connection_gap = br.coupling_gap
        return result
```

- [ ] **Step 6: Update `create_integrated_optimizer_from_config` to handle `centroid_refine_dms` mode**

In `create_integrated_optimizer_from_config`, before the existing `if solver_mode == "two_stage":` branch, insert:

```python
    if solver_mode == "centroid_refine_dms":
        crd_dict = config_dict.get('centroid_refine_dms', {})
        crd_config = CentroidRefineDMSConfig(
            gamma_w=crd_dict.get('gamma_w', 1.0),
            gamma_h=crd_dict.get('gamma_h', 0.64),
            alpha_s=crd_dict.get('alpha_s', 0.0),
            delta_safe=shooting_config.get('safety_margin', 0.02),
            delta_extra=crd_dict.get('delta_extra', 0.01),
            v_nom_fraction=crd_dict.get('v_nom_fraction', 0.5),
            alpha_mu=crd_dict.get('alpha_mu', 0.1),
            tau=crd_dict.get('tau', 0.1),
            mu_min=crd_dict.get('mu_min', 1e-5),
            n_int=shooting_config.get('n_integration_steps', 20),
            delta_min=config_dict.get('dynamics', {}).get('delta_min', 0.01),
            delta_max=config_dict.get('dynamics', {}).get('delta_max', 10.0),
            w_T=cost_config.get('a', 1.0),
            w_L=cost_config.get('w_L', 1.0),
            w_U=cost_config.get('w_E', 1.0),
            w_S=cost_config.get('w_u_smooth', 0.2),
            epsilon_final=crd_dict.get('epsilon_final', 1e-6),
            epsilon_gap=crd_dict.get('epsilon_gap', 0.05),
            time_limit_s=crd_dict.get('time_limit_s', 60.0),
            mode=crd_dict.get('mode', 'first_feasible'),
            use_centroid_cost=crd_dict.get('use_centroid_cost', True),
            use_interface_qp=crd_dict.get('use_interface_qp', True),
            use_log_barrier=crd_dict.get('use_log_barrier', True),
            use_barrier_continuation=crd_dict.get('use_barrier_continuation', True),
            use_inexact_tolerance=crd_dict.get('use_inexact_tolerance', True),
        )
        return CentroidRefineDMSSolver(graph, dynamics, crd_config)
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_centroid_refine_dms.py -v --timeout=120
```

Expected: all 8 tests pass.

- [ ] **Step 8: Run full test suite**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/ --tb=short -q 2>&1 | tail -15
```

- [ ] **Step 9: Commit**

```bash
git -C /home/khoa/ws/GCS_MMS_demo add demo/optimizer.py tests/test_centroid_refine_dms.py
git -C /home/khoa/ws/GCS_MMS_demo commit -m "feat: add CentroidRefineDMSSolver, CentroidRefineDMSConfig, and new OptimizationResult fields"
```

---

## Task 10: `config.yaml` + `experiments.py` — YAML block + `ScenarioResult`

**Files:**
- Modify: `demo/config.yaml`
- Modify: `demo/experiments.py`
- Test: `tests/test_regression.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_regression.py`:

```python
class ScenarioResultTests(unittest.TestCase):
    def test_scenario_result_gcs_defaults(self):
        import sys, math
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
        from experiments import ScenarioResult
        from optimizer import OptimizationResult

        dummy_opt = OptimizationResult(
            success=True, path=[], path_regions=[],
            total_cost=1.0, solve_time=0.1, n_paths_evaluated=1
        )
        sr = ScenarioResult(
            scenario_name="test",
            optimization_result=dummy_opt,
            n_regions=5, n_edges=10, n_paths=3, setup_time=0.1,
            crd_solve_time=0.5, crd_time_to_first_feasible=0.5,
            crd_success=True, crd_objective=1.0, crd_path_length=2.0,
            crd_travel_time=1.0, crd_defect_norm=1e-7, crd_connection_gap=1e-7,
            crd_s_min_sampled=0.01, crd_s_min_certified=0.005,
            crd_n_paths_tried=1, crd_n_barrier_levels=3,
            crd_n_nlp_iterations=150, crd_optimality_gap=0.1,
            crd_lb_geometric=0.8,
        )
        self.assertTrue(math.isnan(sr.gcs_bezier_solve_time))
        self.assertTrue(math.isnan(sr.gcs_bezier_objective))
        self.assertEqual(sr.gcs_bezier_source, "not_set")
        self.assertEqual(sr.gcs_bezier_degree, -1)
```

- [ ] **Step 2: Run test to confirm failure**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_regression.py::ScenarioResultTests -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'ScenarioResult'`.

- [ ] **Step 3: Add `ScenarioResult` to `experiments.py`**

In `demo/experiments.py`, after the `ExperimentResult` dataclass, add:

```python
@dataclass
class ScenarioResult:
    """Per-scenario result with Centroid-Refine-DMS diagnostics and GCS-Bezier comparison slots."""
    scenario_name: str
    optimization_result: OptimizationResult
    n_regions: int
    n_edges: int
    n_paths: int
    setup_time: float

    # Centroid-Refine-DMS diagnostics
    crd_solve_time: float = 0.0
    crd_time_to_first_feasible: float = 0.0
    crd_success: bool = False
    crd_objective: float = float("nan")
    crd_path_length: float = float("nan")
    crd_travel_time: float = float("nan")
    crd_defect_norm: float = float("nan")
    crd_connection_gap: float = float("nan")
    crd_s_min_sampled: float = float("nan")
    crd_s_min_certified: float = float("nan")
    crd_n_paths_tried: int = 0
    crd_n_barrier_levels: int = 0
    crd_n_nlp_iterations: int = 0
    crd_optimality_gap: float = float("inf")
    crd_lb_geometric: float = 0.0

    # GCS-Bezier baseline (filled from external JSON or direct assignment; never from solver)
    gcs_bezier_solve_time: float = float("nan")
    gcs_bezier_objective: float = float("nan")
    gcs_bezier_path_length: float = float("nan")
    gcs_bezier_travel_time: float = float("nan")
    gcs_bezier_s_min: float = float("nan")
    gcs_bezier_degree: int = -1
    gcs_bezier_source: str = "not_set"  # e.g. "Drake GCS", "paper Table 2"
```

- [ ] **Step 4: Add `centroid_refine_dms` block to `config.yaml`**

Append to `demo/config.yaml` after the `optimizer:` block:

```yaml
# -----------------------------------------------------------------------------
# Centroid-Refine-DMS Solver Configuration
# -----------------------------------------------------------------------------
centroid_refine_dms:
  # Graph search
  gamma_w: 1.0                  # interface clearance bonus weight
  gamma_h: 0.64                 # heading-change penalty weight

  # Geometric QP
  alpha_s: 0.0                  # smoothness weight (0 = pure shortest polyline)
  delta_extra: 0.01             # strict-interior margin = delta_safe / 2

  # Warm-start
  v_nom_fraction: 0.5           # v_nom = fraction * (v_max + |v_min|) / 2

  # Barrier schedule (Fiacco-McCormick)
  alpha_mu: 0.1                 # mu_0 = alpha_mu * f0 / B0
  tau: 0.1                      # reduction factor
  mu_min: 1.0e-5                # final barrier parameter

  # Stopping criteria
  epsilon_final: 1.0e-6         # final KKT tolerance
  epsilon_gap: 0.05             # near-optimality gap threshold
  time_limit_s: 60.0            # wall-clock budget

  # Solver mode
  mode: "first_feasible"        # "first_feasible" | "anytime"

  # Ablation flags (all true = full method)
  use_centroid_cost: true
  use_interface_qp: true
  use_log_barrier: true
  use_barrier_continuation: true
  use_inexact_tolerance: true
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/test_regression.py::ScenarioResultTests -v
```

Expected: 1 test passes.

- [ ] **Step 6: Smoke-test full pipeline from config dispatch**

```bash
cd /home/khoa/ws/GCS_MMS_demo/demo && python -c "
import numpy as np
from convex_regions import create_regions_from_vertices_list
from graph_builder import build_region_graph
from dynamics import UnicycleModel
from optimizer import create_integrated_optimizer_from_config, CentroidRefineDMSSolver

regions = create_regions_from_vertices_list([
    np.array([[0,0],[1.2,0],[1.2,1],[0,1]],dtype=float),
    np.array([[0.8,0],[2,0],[2,1],[0.8,1]],dtype=float),
])
x_start = np.array([0.1, 0.5, 0.0])
x_goal  = np.array([1.9, 0.5, 0.0])
graph   = build_region_graph(regions, x_start[:2], x_goal[:2])
dyn     = UnicycleModel()
cfg     = {'optimizer': {'solver_mode': 'centroid_refine_dms'},
           'centroid_refine_dms': {'n_int': 5, 'mu_min': 1e-3, 'epsilon_final': 1e-3},
           'dynamics': {}, 'shooting': {}, 'cost': {}, 'control': {}}
solver  = create_integrated_optimizer_from_config(graph, dyn, cfg)
assert isinstance(solver, CentroidRefineDMSSolver)
result  = solver.solve(x_start, x_goal)
print('success:', result.success)
print('cost:', result.total_cost)
print('safety_cert:', result.safety_certification)
print('disclaimer:', result.global_optimality_claim[:60])
"
```

Expected output: prints `success: True` (or `False` with failure log) and the mandatory KKT disclaimer.

- [ ] **Step 7: Run full test suite**

```bash
cd /home/khoa/ws/GCS_MMS_demo && python -m pytest tests/ --tb=short -q 2>&1 | tail -15
```

- [ ] **Step 8: Commit**

```bash
git -C /home/khoa/ws/GCS_MMS_demo add demo/config.yaml demo/experiments.py tests/test_regression.py
git -C /home/khoa/ws/GCS_MMS_demo commit -m "feat: add centroid_refine_dms config block, ScenarioResult with GCS-Bezier comparison slots"
```

---

## Self-Review Against Spec

| Spec Section | Covered By |
|---|---|
| Stage A: K-shortest with composite cost | Tasks 2, 3 |
| Stage B: Interface QP + narrow check | Task 4 |
| Stage C: Warm-start IVP | Task 5 |
| Stage D: Barrier init (μ₀, B₀) | Tasks 6, 7 |
| Stage E: Barrier continuation | Task 8 |
| Stage F: Diagnostics (sampled + certified safety) | Task 8 |
| Stage G: Failure + blacklist | Tasks 7, 9 |
| R1: Defect as hard equality | Task 8 (equality in g_sym) |
| R2: Barrier only on safety | Task 8 (only safety slacks in B) |
| R3: Strict-interior assertion | Task 5 (generate_warm_start_ivp) |
| R4: No global optimality claim | Task 9 (MANDATORY_DISCLAIMER) |
| R5: Both safety margins populated | Tasks 8, 9 |
| R6: Barrier params derived at runtime | Tasks 7, 8 |
| R7: Mandatory failure logging | Tasks 7, 9 |
| R8: QP and NLP independent | Tasks 4, 8 (separate CasADi instances) |
| R9: compute_lipschitz_bound abstract | Task 1 |
| R10: Ablation flags respected | Task 9 (use_interface_qp, use_log_barrier branches) |
| R11: No mesh points in barrier path | Task 8 (no mesh_sampler call) |
| R12: GCS-Bézier slots are data fields only | Task 10 |
| DoubleIntegratorDynamics | Task 1 |
| config.yaml centroid_refine_dms block | Task 10 |
| ScenarioResult comparison slots | Task 10 |

**Known gaps for implementer to resolve:**
- `use_barrier_continuation=False`: currently `BarrierDMSSolver` always runs the full schedule. Add branch: `if not cfg.use_barrier_continuation: schedule = BarrierSchedule(mu_schedule=[cfg.mu_min], n_mu=1)`.
- `use_inexact_tolerance=False`: change `eps_r = cfg.epsilon_final` (not `max(cfg.epsilon_final, mu_r)`).
- `_build_nlp_symbols` returns placeholder `x0_flat`; the real warm-start vector is separately computed in `solve_path` via `_flatten_warm_start` and passed directly to IPOPT. This is correct as written.
- `_extract_region_indices` imports from `graph_types` — verify this module exists in the codebase and exports `region_index_from_node_id`. If not, inline the function.
