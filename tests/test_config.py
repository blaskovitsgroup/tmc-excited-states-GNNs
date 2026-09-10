import pandas as pd

from tmqmg_es import config as C
from .conftest import requires_data


def test_no_overlap_between_inputs_and_leakage():
    leak = set(C.LEAKAGE_COLS)
    # permitted Regime-B inputs must never be excited-state labels
    assert leak.isdisjoint(set(C.PARENT_DESCRIPTOR_COLS))
    # ground-state star columns are targets, not in the leakage list
    assert leak.isdisjoint(set(C.STAR_GROUNDSTATE_COLS))


def test_leakage_registry_has_no_duplicates():
    assert len(C.LEAKAGE_COLS) == len(set(C.LEAKAGE_COLS))


@requires_data
def test_leakage_registry_exactly_partitions_star_columns():
    cols = set(pd.read_csv(C.STAR_CSV, nrows=0).columns)
    registry = {"id"} | set(C.LEAKAGE_COLS) | set(C.STAR_GROUNDSTATE_COLS)
    # every label column in the registry exists, and nothing is unaccounted for
    assert cols == registry
