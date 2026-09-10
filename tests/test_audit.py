import numpy as np

N_EXPECTED = 74273


def test_row_count_and_join(phase1):
    cleaned = phase1["cleaned"]
    assert len(cleaned) == N_EXPECTED
    # every star molecule has a parent (inner join) and an XYZ file
    assert cleaned["has_xyz"].all()
    assert phase1["manifest"]["has_parent"].all()


def test_yejxob_invalid_masked(phase1):
    cleaned = phase1["cleaned"]
    invalid = phase1["invalid"]
    assert "YEJXOB" in set(invalid["id"])
    row = cleaned.loc[cleaned["id"] == "YEJXOB"]
    # raw value preserved, eV target masked to NaN
    assert np.isclose(row["lambda_1_acetone"].iloc[0], -4321.96)
    assert np.isnan(row["E_1_acetone_eV"].iloc[0])


def test_only_one_invalid_state_wavelength(phase1):
    wl = phase1["invalid"][
        phase1["invalid"]["reason"] == "non_physical_wavelength_le_0"
    ]
    assert len(wl) == 1


def test_energy_columns_are_physical(phase1):
    cleaned = phase1["cleaned"]
    e = cleaned["E_1_gasphase_eV"].dropna()
    # state-1 excitation energies sit in a sane optical window
    assert (e > 0).all()
    assert e.max() < 15.0


def test_mask_consistency_visible_ct_nto(phase1):
    cleaned = phase1["cleaned"]
    # CT class and NTO are defined on exactly the visible-band molecules
    for s in ("gasphase", "acetone"):
        vis = cleaned[f"mask_peak_vis_{s}"]
        assert (cleaned[f"mask_ct_{s}"] == vis).all()
        assert (cleaned[f"mask_nto_{s}"] == vis).all()


def test_ct_class_counts(phase1):
    vc = phase1["cleaned"]["transition_nature_vis_gasphase"].value_counts()
    assert vc["LLCT"] == 18593
    assert vc["MLCT"] == 9583
    assert vc["LMCT"] == 2190
    assert vc["ddT"] == 409


def test_metal_fraction_in_unit_interval(phase1):
    m = phase1["cleaned"]["M_frac_occ_gasphase"].dropna()
    assert (m >= 0).all() and (m <= 1).all()


def test_corrupted_source_shift_labels_are_audited_and_excluded(phase1):
    summary = phase1["shift_summary"]
    assert summary["numeric_mismatches"] == 188
    assert summary["relation_mismatches"] == 188
    assert summary["undefined_stored_as_zero"] == 39934
    assert summary["training_use"] == "excluded"
    fenzon = phase1["shift_issues"].set_index("id").loc["FENZON"]
    assert fenzon["corrected_relation"] == "vis_to_nir"
    assert np.isclose(fenzon["corrected_lambda_delta"], 16.21)
