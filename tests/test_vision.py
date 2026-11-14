"""Check Chapter 22 patch order, fixed probes, and video pair counts."""

import math
import unittest

import numpy as np

from examples.ch22_vision.model import (
    patch_embeddings,
    patchify,
    probe_metrics,
    space_time_pairs,
    train_linear_probe,
)
from examples.ch22_vision.run_examples import (
    TEST_PROFILES,
    TRAIN_PROFILES,
    patch_example,
    patch_mean_features,
    probe_example,
    profile_images,
)


class PatchTests(unittest.TestCase):
    def test_worked_image_shapes_and_patch_start_values(self):
        result = patch_example()
        self.assertEqual(result["patch_shape"], [4, 48])
        self.assertEqual(result["projection_shape"], [48, 6])
        self.assertEqual(result["token_shape_with_cls"], [5, 6])
        self.assertEqual([row[0] for row in result["first_six_values_per_patch"]],
                         [0, 12, 96, 108])
        np.testing.assert_allclose(result["tokens"][1], np.arange(6) + 0.1)

    def test_rectangular_and_noncontiguous_images_keep_row_major_patch_order(self):
        image = np.arange(6 * 4 * 3, dtype=np.uint8).reshape(6, 4, 3)[:, ::-1]
        expected = np.stack([
            image[row:row + 2, col:col + 2, :].reshape(-1)
            for row in range(0, 6, 2)
            for col in range(0, 4, 2)
        ])
        actual = patchify(image, 2)
        self.assertEqual(actual.dtype, np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_projection_and_optional_class_position_rows(self):
        patches = [[1, 2], [3, 4]]
        weight = [[1, -1, 0], [0, 2, 1]]
        base = [[1, 3, 2], [3, 5, 4]]
        np.testing.assert_array_equal(patch_embeddings(patches, weight), base)
        positioned = patch_embeddings(
            patches, weight, cls=[1, 0, 0], positions=[[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        )
        np.testing.assert_array_equal(positioned, [[1, 1, 0], [1, 3, 3], [4, 5, 4]])
        with self.assertRaisesRegex(ValueError, "one positional vector per token"):
            patch_embeddings(patches, weight, cls=[1, 0, 0], positions=np.zeros((2, 3)))

    def test_patch_inputs_reject_bad_spatial_shapes_and_numeric_types(self):
        with self.assertRaisesRegex(ValueError, "divide"):
            patchify(np.zeros((7, 8, 3)), 4)
        with self.assertRaisesRegex(ValueError, "channels"):
            patchify(np.zeros((8, 8)), 4)
        with self.assertRaisesRegex(ValueError, "finite real"):
            patchify(np.full((4, 4, 3), 1j), 4)
        with self.assertRaisesRegex(ValueError, "widths"):
            patch_embeddings(np.zeros((4, 48)), np.zeros((49, 6)))


class ProbeTests(unittest.TestCase):
    def test_fixture_pairs_share_histograms_and_train_test_are_disjoint(self):
        train, train_labels = profile_images(TRAIN_PROFILES)
        test, test_labels = profile_images(TEST_PROFILES)
        self.assertFalse(set(map(tuple, train.reshape(len(train), -1))) &
                         set(map(tuple, test.reshape(len(test), -1))))
        for images, labels in ((train, train_labels), (test, test_labels)):
            means = patch_mean_features(images)
            for row in range(0, len(images), 2):
                np.testing.assert_array_equal(labels[row:row + 2], [0, 1])
                np.testing.assert_array_equal(
                    np.sort(images[row], axis=None), np.sort(images[row + 1], axis=None)
                )
                np.testing.assert_array_equal(np.sort(means[row]), np.sort(means[row + 1]))
                self.assertFalse(np.array_equal(means[row], means[row + 1]))

    def test_linear_probe_matches_a_scalar_first_gradient_and_preserves_features(self):
        x = np.array([[1.0, 2.0], [3.0, -1.0]])
        y = [0, 1]
        before = x.copy()
        weight, bias = train_linear_probe(x, y, steps=1, lr=0.2)
        np.testing.assert_allclose(weight, [[-0.1, 0.1], [0.15, -0.15]], rtol=0, atol=1e-16)
        np.testing.assert_array_equal(bias, [0, 0])
        np.testing.assert_array_equal(x, before)

    def test_held_out_comparison_keeps_same_head_capacity(self):
        result = probe_example()
        ordered, orderless = result["ordered"], result["orderless_sorted"]
        self.assertEqual(ordered["parameters"], orderless["parameters"])
        self.assertEqual(ordered["parameters"], 10)
        self.assertEqual(result["train_images"], 12)
        self.assertEqual(result["held_out_images"], 8)
        self.assertEqual(ordered["held_out"]["accuracy"], 1)
        self.assertLess(ordered["held_out"]["loss"], 0.1)
        self.assertEqual(orderless["held_out"]["accuracy"], 0.5)
        self.assertAlmostEqual(orderless["held_out"]["loss"], math.log(2))

    def test_probe_checks_labels_and_computes_stable_loss(self):
        self.assertAlmostEqual(
            probe_metrics([[1]], [0], [[1000, 1000]], [0, 0])["loss"], math.log(2)
        )
        with self.assertRaisesRegex(ValueError, "0 or 1"):
            train_linear_probe([[1]], [2])
        with self.assertRaisesRegex(ValueError, "positive"):
            train_linear_probe([[1]], [1], steps=0)


class VideoTests(unittest.TestCase):
    def test_video_counts_match_explicit_ordered_pairs(self):
        positions = [(t, s) for t in range(3) for s in range(4)]
        pairs = [(left, right) for left in positions for right in positions]
        spatial = sum(left[0] == right[0] for left, right in pairs)
        temporal = sum(left[1] == right[1] for left, right in pairs)
        self.assertEqual(space_time_pairs(3, 4), {
            "joint": len(pairs), "spatial": spatial, "temporal": temporal,
            "divided": spatial + temporal,
        })
        self.assertEqual(space_time_pairs(1, 1)["divided"], 2)


if __name__ == "__main__":
    unittest.main()
