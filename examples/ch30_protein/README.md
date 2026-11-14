# Chapter 30. Scientific Sequences and Biological Structure

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch30_protein.run_examples --output output/ch30.json
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
[complete solutions](../../solutions/ch30_protein/README.md).

## Embeddings from ESMC-300M

The demo encodes six deposited protein sequences in one padded batch and
uses the last hidden layer. It pools only residue rows, before comparing
the 960-dimensional vectors by cosine similarity. The composition
baseline uses the same six sequences. The recorded cosine for the
human and yeast ubiquitins is 0.989 in the embedding space. Human
ubiquitin and calmodulin also score 0.917, so a high cosine in this
small collection is not a family label.

## Structures from ESMFold2-Fast

The structure demo predicts one chain for each of ubiquitin (1UBQ),
villin headpiece (1YU5), and calmodulin (1CLL). It uses a fixed seed
(2900 for each chain), `num_loops=3` (three additional trunk passes after
the initial pass), and 14 diffusion steps.
The model receives the deposited sequence. Experimental coordinates
enter only in evaluation. The calmodulin prediction is also compared
with 1CFD, a calcium-free reference with the same 148-residue sequence.
The 1CLL reference lacks C-alpha coordinates at positions 1–3 and 148.
Both calmodulin comparisons use the remaining 144 shared positions,
with their original sequence indices.

The geometry is evaluated on matched C-alpha coordinates. RMSD uses
the Kabsch rotation and translation. The C-alpha lDDT uses reference
neighbors at distances below 15 angstroms and four error thresholds
(0.5, 1, 2, and 4 angstroms). Contact counts use an inclusive 8 angstroms
cutoff and a separation of at least six sequence positions. The
predicted C-alpha pLDDT is scaled from 0–1 to 0–100. It and the PAE
matrix are confidence outputs from the model. These long-established
structures illustrate the measurements and can overlap training data.

### Recreate the figures from saved outputs

The [embedding record](records/embedding_runs.json) contains the pooled
vectors and cosine matrices. The [structure record](records/structure_runs.json)
contains predicted C-alpha coordinates, per-position confidence,
evaluation scores and the calmodulin PAE. The reference coordinates
and all sequence-to-chain mappings live in [pdb_references.json](data/pdb_references.json).

```sh
python3.12 -m venv .venv-protein
source .venv-protein/bin/activate
python -m pip install "numpy==2.5.4" "matplotlib==3.11.2"
python -m examples.ch30_protein.make_figures --output-dir output/ch30-figures
```

### Rerun the model demos

```sh
python -m pip install -r examples/ch30_protein/requirements-pretrained.txt
python -m examples.ch30_protein.run_pretrained embeddings --output output/ch30-embeddings.json
python -m examples.ch30_protein.run_pretrained structures --output output/ch30-structures.json
python -m examples.ch30_protein.make_figures --embeddings output/ch30-embeddings.json \
  --structures output/ch30-structures.json --output-dir output/ch30-figures
```

The models run on CPU using six threads by default. ESMC uses float32
and ESMFold2-Fast requests bfloat16 through the native Transformers
implementation. The first run downloads about 1.3 GB for
[ESMC-300M](https://huggingface.co/biohub/ESMC-300M)
and about 26 GB for [ESMFold2-Fast](https://huggingface.co/biohub/ESMFold2-Fast).
The runner verifies the weight checksums and accepts
`--checkpoints DIR` for an existing local mirror with subdirectories
`ESMC-300M` and `ESMFold2-Fast`. The structure command needs enough
RAM for the bundled ESMC-6B backbone and folding heads. The plot
command uses only the small records above.

### Data and weights

The protein examples are drawn from the PDB archive, whose data files
are [CC0](https://www.wwpdb.org/about/usage-policies). The companion
includes a reduced snapshot with deposited sequences and, where
needed, experimental C-alpha coordinates. To rebuild it from the
archived input revisions held locally, use
`python -m examples.ch30_protein.reference_data --mmcif-dir DIR --output FILE`.
The SHA-256 of each original mmCIF is recorded and checked by the tool.

| PDB entry and chain | Protein | Data attribution |
|---|---|---|
| [1UBQ](https://doi.org/10.2210/pdb1UBQ/pdb), A | Human ubiquitin | Vijay-Kumar, Bugg and Cook |
| [2G3Q](https://doi.org/10.2210/pdb2G3Q/pdb), B | Yeast ubiquitin | Swanson, Hicke and Radhakrishnan |
| [1U7S](https://doi.org/10.2210/pdb1U7S/pdb), A | Sperm-whale myoglobin | Zhang and Phillips |
| [1WLA](https://doi.org/10.2210/pdb1WLA/pdb), A | Horse myoglobin | Maurus and Brayer |
| [1YU5](https://doi.org/10.2210/pdb1YU5/pdb), X | Chicken villin headpiece | Meng and colleagues |
| [1CLL](https://doi.org/10.2210/pdb1CLL/pdb), A | Calmodulin with calcium | Chattopadhyaya and Quiocho |
| [1CFD](https://doi.org/10.2210/pdb1CFD/pdb), A | Calcium-free calmodulin | Kuboniwa and colleagues |

The model family is described by [Candido et al. (2026)](https://biohub.ai/papers/esm_protein.pdf).
