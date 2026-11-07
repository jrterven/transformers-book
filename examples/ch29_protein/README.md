# Chapter 29. Scientific Sequences and Biological Structure

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch29_protein.run_examples --output output/ch29.json
python -m unittest discover -s tests -p test_protein.py -v
```

## Count residues and adjacent pairs

`residue_composition` takes a nonempty sequence of standard
uppercase amino acid symbols. It returns the fractions in the
order `ACDEFGHIKLMNPQRSTVWY`. `dipeptide_frequencies` uses the same
alphabet on both axes. Rows index the earlier residue and columns
index the later one. For a sequence of length `L > 1`, each count
is divided by `L - 1`. A single residue gives a zero pair matrix.

The fixture uses `ACAC` and `AACC`. Their residue fractions match.
Their adjacent-pair frequencies expose a difference in order.

## Pool residue positions

`mean_pool_residues` accepts an embedding matrix
`[positions, width]` and a matching Boolean mask. Mark the actual
residue rows and exclude special and padding positions. The
function averages the selected rows over the position axis. The
fixture uses hand-chosen rows to show what the mask changes.

## Compute a contact map

`contact_map` takes one Cartesian point per residue in row order.
It marks pairs whose Euclidean distance is at most the supplied
threshold and whose absolute index difference is at least the
minimum separation. Both off-diagonal orientations of a contact
appear in the Boolean matrix. The example reads the upper
triangle to report each pair once and adds one to the indices for
display. Its coordinates and thresholds illustrate the calculation.

## Keep the represented quantities distinct

Composition divides by L; adjacent-pair counts divide by L-1 when L>1.
The one-residue pair result is an all-zero special case. Two different sequences
can have identical composition and dipeptide frequencies.

The residue mask controls the pooling numerator and denominator; it does not
replace the encoder's attention mask. Contact maps use both a distance cutoff
and sequence separation. The helper assumes consecutive positions in one chain;
missing residues or multiple chains require explicit indexing rules.

A learned pair vector is not a distance or contact probability. PAE is predicted
aligned error and need not be symmetric. High local confidence does not establish
the orientation of two domains or a proposed binding function.

See the [11 exercises](EXERCISES.md) and
[complete solutions](../../solutions/ch29_protein/README.md).
