from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from convex_regions import ConvexRegion, compute_intersection, regions_intersect


class ConvexRegionTests(unittest.TestCase):
    def test_halfspace_contains_square_points(self) -> None:
        region = ConvexRegion(
            vertices=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ]
            )
        )

        self.assertTrue(region.contains(np.array([0.5, 0.5])))
        self.assertTrue(region.contains(np.array([0.0, 0.5])))
        self.assertFalse(region.contains(np.array([1.1, 0.5])))

    def test_intersection_for_overlapping_squares(self) -> None:
        left = ConvexRegion(
            vertices=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ]
            )
        )
        right = ConvexRegion(
            vertices=np.array(
                [
                    [0.5, 0.0],
                    [1.5, 0.0],
                    [1.5, 1.0],
                    [0.5, 1.0],
                ]
            )
        )

        self.assertTrue(regions_intersect(left, right))
        intersection = compute_intersection(left, right)
        self.assertIsNotNone(intersection)
        self.assertTrue(intersection.contains(np.array([0.75, 0.5])))


if __name__ == "__main__":
    unittest.main()
