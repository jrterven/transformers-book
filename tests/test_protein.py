"""Check Chapter 29's composition, masks, and contact conventions."""

import unittest

import numpy as np

from examples.ch29_protein.model import (
    AMINO_ACIDS,
    ca_lddt,
    contact_map,
    cosine_matrix,
    dipeptide_frequencies,
    mean_pool_residues,
    residue_composition,
    superpose_coordinates,
)
from examples.ch29_protein.run_examples import contact_example, pooling_example, sequence_example


class SequenceTests(unittest.TestCase):
    def test_composition_fixes_the_alphabet_and_counts_each_residue(self):
        np.testing.assert_array_equal(residue_composition(AMINO_ACIDS), np.full(20, 0.05))
        result = sequence_example()["sequences"]
        self.assertEqual(result["ACAC"]["nonzero_composition"], {"A": 0.5, "C": 0.5})
        self.assertEqual(result["AACC"]["nonzero_composition"], {"A": 0.5, "C": 0.5})

    def test_ordered_pairs_match_the_worked_sequences_and_reverse_on_reversal(self):
        result = sequence_example()["sequences"]
        self.assertEqual(result["ACAC"]["nonzero_dipeptides"], {"AC": 2 / 3, "CA": 1 / 3})
        self.assertEqual(result["AACC"]["nonzero_dipeptides"],
                         {"AA": 1 / 3, "AC": 1 / 3, "CC": 1 / 3})
        sequence = "ACYWDAC"
        forward = dipeptide_frequencies(sequence)
        np.testing.assert_array_equal(dipeptide_frequencies(sequence[::-1]), forward.T)
        self.assertAlmostEqual(forward.sum(), 1)

    def test_one_residue_has_no_pairs_and_invalid_sequences_raise(self):
        np.testing.assert_array_equal(dipeptide_frequencies("W"), np.zeros((20, 20)))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            residue_composition("")
        for sequence in ("AX", "ac", "A C"):
            with self.subTest(sequence=sequence), self.assertRaisesRegex(ValueError, "standard"):
                dipeptide_frequencies(sequence)


class PoolingTests(unittest.TestCase):
    def test_cosine_normalizes_before_comparing(self):
        vectors = [[1, 2], [2, 1], [-1, -2], [2, 4]]
        matrix = cosine_matrix(vectors)
        np.testing.assert_allclose(matrix[0], [1, 0.8, -1, 1])
        np.testing.assert_allclose(matrix, matrix.T)

    def test_masked_pooling_fixture_and_nonfinite_excluded_rows(self):
        self.assertEqual(pooling_example()["pooled"], [1, 1])
        embeddings = np.array([[np.nan, np.inf], [2, 0], [0, 4]], dtype=np.float32)
        with np.errstate(all="raise"):
            pooled = mean_pool_residues(embeddings, [False, True, True])
        np.testing.assert_array_equal(pooled, [1, 2])
        self.assertEqual(pooled.dtype, np.float64)

    def test_pooling_rejects_missing_mask_shape_and_selected_nan(self):
        with self.assertRaisesRegex(ValueError, "Boolean"):
            mean_pool_residues([[1, 2]], [1])
        with self.assertRaisesRegex(ValueError, "at least one"):
            mean_pool_residues([[1, 2]], [False])
        with self.assertRaisesRegex(ValueError, "selected"):
            mean_pool_residues([[np.nan, 2]], [True])


class GeometryTests(unittest.TestCase):
    def test_superposition_removes_rigid_motion_and_preserves_handedness(self):
        measured = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=float)
        rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        mobile = measured @ rotation + [5, -2, 1]
        np.testing.assert_allclose(superpose_coordinates(mobile, measured), measured, atol=1e-12)
        reflection = measured * [-1, 1, 1]
        self.assertGreater(np.linalg.norm(superpose_coordinates(reflection, measured) - measured), 1)

    def test_ca_lddt_uses_reference_neighbors_and_all_four_thresholds(self):
        measured = [[0, 0, 0], [6, 0, 0], [0, 6, 0], [0, 0, 30]]
        predicted = [[0, 0, 0], [7.2, 0, 0], [0, 6.4, 0], [0, 0, 30]]
        global_score, local = ca_lddt(predicted, measured)
        self.assertAlmostEqual(global_score, (0.5 + 1 + 0.5) / 3)
        np.testing.assert_allclose(local, [0.75, 0.5, 0.75, np.nan])

    def test_worked_contacts_include_equal_distance_and_exclude_adjacent_rows(self):
        result = contact_example()
        self.assertEqual(result["unique_pairs_one_based"], [[1, 3], [2, 4]])
        contacts = np.asarray(result["contact_map"])
        np.testing.assert_array_equal(contacts, contacts.T)
        np.testing.assert_array_equal(np.diag(contacts), False)
        self.assertFalse(contacts[1, 2])
        self.assertTrue(contacts[1, 3])
        just_below = contact_map(result["coordinates"], np.nextafter(5.0, 0), 2)
        self.assertFalse(just_below[1, 3])

    def test_separation_zero_distance_and_empty_geometry(self):
        points = [[0, 0, 0], [5, 0, 0], [0, 0, 0]]
        contacts = contact_map(points, 0, 2)
        self.assertTrue(contacts[0, 2])
        self.assertEqual(int(contacts.sum()), 2)
        self.assertFalse(contact_map(points, 100, 3).any())
        self.assertEqual(contact_map(np.empty((0, 3)), 5).shape, (0, 0))

    def test_distances_use_all_three_axes_and_handle_large_coordinates(self):
        coordinates = [[0, 0, 0], [0, 0, 3], [3e200, 4e200, 0]]
        contacts = contact_map(coordinates, 2.9)
        self.assertFalse(contacts[0, 1])
        self.assertTrue(contact_map(coordinates, 3)[0, 1])
        self.assertTrue(contact_map(coordinates, 5.1e200)[0, 2])
        self.assertFalse(contact_map([[-1e308, 0, 0], [1e308, 0, 0]], 1.5e308)[0, 1])

    def test_exact_pythagorean_contact_at_the_cutoff(self):
        coordinates = [[0, 0, 0], [4, 4, -2]]
        self.assertTrue(contact_map(coordinates, 6)[0, 1])
        self.assertFalse(contact_map(coordinates, np.nextafter(6.0, 0))[0, 1])

    def test_contact_inputs_are_real_finite_and_well_shaped(self):
        with self.assertRaisesRegex(ValueError, "residues, 3"):
            contact_map([[0, 0]], 5)
        with self.assertRaisesRegex(ValueError, "finite"):
            contact_map([[0, np.inf, 0]], 5)
        with self.assertRaisesRegex(ValueError, "threshold"):
            contact_map([[0, 0, 0]], -1)
        with self.assertRaisesRegex(ValueError, "at least one"):
            contact_map([[0, 0, 0]], 5, 0)


if __name__ == "__main__":
    unittest.main()
