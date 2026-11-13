"""Read the pinned UCI tables and split according to the unit being tested."""

import csv
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


DATA = Path(__file__).with_name("data")
SEED = 2800
DATA_SHA256 = {
    "raisin": "175ecbc52d351e0e0d9aa00605de09425b3a4826c5db294fabcbc24bcba955ab",
    "concrete": "8d4b15b6fc68cd932d745cbd663d5ceae66dd54422e99c1e4865f2936ab7e2af",
}
RAISIN_FEATURES = (
    "Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity",
    "ConvexArea", "Extent", "Perimeter",
)
CONCRETE_FEATURES = (
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer",
    "Coarse Aggregate", "Fine Aggregate", "Age",
)


@dataclass(frozen=True)
class Table:
    features: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray
    row_ids: np.ndarray
    source_rows: int
    groups: np.ndarray | None = None


def read_rows(name, expected_header):
    path = DATA / f"{name}.csv"
    if sha256(path.read_bytes()).hexdigest() != DATA_SHA256[name]:
        raise ValueError(f"{path} differs from the pinned UCI snapshot")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        if tuple(next(reader)) != expected_header:
            raise ValueError(f"unexpected columns in {path}")
        return list(reader)


# snippet:recipe_groups
def recipe_groups(features):
    """Group exact ingredient vectors and keep curing age as a feature."""
    _, group = np.unique(features[:, :7], axis=0, return_inverse=True)
    return group
# end-snippet:recipe_groups


def load_tables():
    raisin_rows = read_rows("raisin", (*RAISIN_FEATURES, "Class"))
    class_id = {"Kecimen": 0, "Besni": 1}
    raisin = Table(
        RAISIN_FEATURES,
        np.array([row[:-1] for row in raisin_rows], dtype=np.float64),
        np.array([class_id[row[-1]] for row in raisin_rows]),
        np.arange(len(raisin_rows)), len(raisin_rows),
    )
    concrete_rows = read_rows(
        "concrete", (*CONCRETE_FEATURES, "Concrete compressive strength")
    )
    concrete_data = np.asarray(concrete_rows, dtype=np.float64)
    # Drop exact repeats of the full record, keeping the first source row ID.
    _, first = np.unique(concrete_data, axis=0, return_index=True)
    first.sort()
    x, y = concrete_data[first, :-1], concrete_data[first, -1]
    concrete = Table(CONCRETE_FEATURES, x, y, first, len(concrete_data), recipe_groups(x))
    return {"raisin": raisin, "concrete": concrete}


def split_table(table):
    """Return an ordered training pool and a fixed set of test row indices."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    rng = np.random.default_rng(SEED)
    if table.groups is None:
        train, test = train_test_split(
            np.arange(len(table.y)), test_size=0.2, random_state=SEED, stratify=table.y
        )
        negative = rng.permutation(train[table.y[train] == 0])
        positive = rng.permutation(train[table.y[train] == 1])
        # The pinned Raisin table has equal numbers of both classes.
        train = np.column_stack((negative, positive)).ravel()
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        train, test = next(splitter.split(table.x, groups=table.groups))
        train = rng.permutation(train)
    return train, np.sort(test)
