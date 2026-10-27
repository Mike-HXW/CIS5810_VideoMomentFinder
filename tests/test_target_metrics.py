"""Unit tests for target-aware ranking metrics."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from target_metrics import (
    PRESET_WEIGHTS,
    composition_score,
    robust_normalize,
    tenengrad_sharpness,
    track_stability_for_frame,
    visibility_components,
)


class TargetMetricsTests(unittest.TestCase):
    def test_tenengrad_prefers_sharp_edges(self):
        checker = np.indices((160, 160)).sum(axis=0) % 2
        sharp = (checker * 255).astype(np.uint8)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
        blurred = cv2.GaussianBlur(sharp, (21, 21), 5)
        box = (0, 0, 160, 160)
        self.assertGreater(
            tenengrad_sharpness(sharp, box),
            tenengrad_sharpness(blurred, box),
        )

    def test_visibility_prefers_large_centered_stable_target(self):
        good = visibility_components((300, 120, 700, 900), 1000, 1000, 1.0)
        poor = visibility_components((0, 0, 80, 120), 1000, 1000, 0.3)
        self.assertGreater(good["visibility_score"], poor["visibility_score"])
        self.assertEqual(poor["edge_clearance_score"], 0.0)

    def test_composition_prefers_rule_of_thirds_to_corner(self):
        near_third = composition_score((233, 233, 433, 433), 1000, 1000)
        corner = composition_score((0, 0, 100, 100), 1000, 1000)
        self.assertGreater(near_third, corner)

    def test_robust_normalize_is_bounded_and_ordered(self):
        normalized = robust_normalize([1.0, 2.0, 3.0, 100.0])
        self.assertEqual(len(normalized), 4)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in normalized))
        self.assertEqual(normalized, sorted(normalized))

    def test_track_stability_uses_neighboring_frames(self):
        presence = [True, True, False, True, False]
        self.assertAlmostEqual(track_stability_for_frame(presence, 2, 1), 2 / 3)

    def test_preset_weights_sum_to_one(self):
        for preset, weights in PRESET_WEIGHTS.items():
            with self.subTest(preset=preset):
                self.assertAlmostEqual(sum(weights.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
