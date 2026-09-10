"""Project constants, paths, and the label/leakage column registry.

Single source of truth for physical constants, region definitions, and the
exhaustive list of columns that are TD-DFT excited-state *labels* (and therefore
must never be fed to a model as input). See workflow sections 6 and 7.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- physical constants ---------------------------------------------------
# E[eV] = HC_EV_NM / lambda[nm]
HC_EV_NM: float = 1239.841984

# --- dataset geometry -----------------------------------------------------
N_STATES: int = 30  # first 30 excitations extracted in tmQMg*
SOLVENTS: tuple[str, ...] = ("gasphase", "acetone")

# Region boundaries in nm, exactly as defined in
# tmQMg_star-main/analysis/tddft_data_parser.py::_split_spectrum_into_uv_vis_nir
REGION_BOUNDS_NM = {
    "uv": (0.0, 350.0),  # nm < 350
    "vis": (350.0, 825.0),  # 350 <= nm <= 825
    "nir": (825.0, float("inf")),  # nm > 825
}
REGIONS: tuple[str, ...] = ("uv", "vis", "nir")

# A region "band" is reported when its maximum oscillator strength is at least
# this value. The parser rejects f_max < 0.01, so equality is included.
F_THRESHOLD: float = 0.01

# Visible charge-transfer transition classes (analysis/analyze.py).
CT_CLASSES: tuple[str, ...] = ("ddT", "LLCT", "MLCT", "LMCT")

# 3d / 4d / 5d transition metals (analysis/analyze.py).
TRANSITION_METALS: tuple[str, ...] = (
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "La",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
)
# Period (row) of each metal: 4 = 3d, 5 = 4d, 6 = 5d. Used for row-holdout split.
METAL_ROW = {
    m: r
    for r, ms in {
        4: ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"),
        5: ("Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"),
        6: ("La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"),
    }.items()
    for m in ms
}

# --- periodic table (symbol -> Z) -----------------------------------------
# Built in so the geometry/training path has no rdkit dependency (rdkit ships a
# dummy wheel on some Alliance clusters). rdkit stays optional for scaffolds.
_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I "
    "Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr"
).split()
SYMBOL_TO_Z = {s: i + 1 for i, s in enumerate(_SYMBOLS)}


# --- paths ----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "protocol" / "frozen_protocol.yaml"


def _default_source_root() -> Path:
    """Locate bundled source inputs, falling back to the original layout."""
    bundled = REPO_ROOT / "source_inputs"
    if (bundled / "tmQMg_star-main").exists():
        return bundled
    sibling = REPO_ROOT.parent
    if (sibling / "tmQMg_star-main").exists():
        return sibling
    return REPO_ROOT


def _env_path(var: str, default: Path) -> Path:
    return Path(os.environ[var]) if var in os.environ else default


SOURCE_ROOT = _env_path("TMQMG_SOURCE_ROOT", _default_source_root())
STAR_CSV = _env_path("TMQMG_STAR_CSV", SOURCE_ROOT / "tmQMg_star-main" / "tmQMg*.csv")
PARENT_CSV = _env_path(
    "TMQMG_PARENT_CSV",
    SOURCE_ROOT / "tmQMg-main" / "data" / "tmQMg_properties_and_targets.csv",
)
XYZ_DIR = _env_path("TMQMG_XYZ_DIR", SOURCE_ROOT / "tmQMg-main" / "data" / "xyz")
OUTPUT_DIR = _env_path("TMQMG_OUTPUT_DIR", REPO_ROOT / "outputs")
PHASE1_DIR = OUTPUT_DIR / "phase1"


# --- label / leakage column registry --------------------------------------
def _state_cols(prefix: str) -> list[str]:
    return [f"{prefix}_{i}_{s}" for s in SOLVENTS for i in range(1, N_STATES + 1)]


def state_lambda_cols() -> list[str]:
    return _state_cols("lambda")


def state_f_cols() -> list[str]:
    return _state_cols("f")


def peak_cols() -> list[str]:
    cols: list[str] = []
    for s in SOLVENTS:
        for r in REGIONS:
            cols += [f"lambda_max_{r}_{s}", f"f_max_{r}_{s}", f"sigma_{r}_{s}"]
    return cols


def ct_cols() -> list[str]:
    return [f"transition_nature_vis_{s}" for s in SOLVENTS]


def nto_cols() -> list[str]:
    cols: list[str] = []
    for s in SOLVENTS:
        for kind in ("occupied", "virtual"):
            cols += [f"M_contribution_{kind}_{s}", f"L_contribution_{kind}_{s}"]
    return cols


SHIFT_BOOL_COLS: list[str] = [
    "vis_to_vis",
    "uv_to_vis",
    "nir_to_vis",
    "vis_to_uv",
    "vis_to_nir",
    "bathochromic",
    "hypsochromic",
    "hyperchromic",
    "hypochromic",
]
SHIFT_NUM_COLS: list[str] = ["lambda_delta", "f_delta"]

# These three are computed by the TD-DFT job's ground state. They are valid as
# auxiliary *targets* (workflow section 4) but are not structure-only inputs.
STAR_GROUNDSTATE_COLS: list[str] = [
    f"{p}_{s}"
    for s in SOLVENTS
    for p in ("homo_lumo_gap", "dipole_moment", "metal_charge")
]


def leakage_cols() -> list[str]:
    """Every column derived from TD-DFT excited states. Never a model input."""
    return (
        state_lambda_cols()
        + state_f_cols()
        + peak_cols()
        + ct_cols()
        + nto_cols()
        + SHIFT_NUM_COLS
        + SHIFT_BOOL_COLS
    )


LEAKAGE_COLS: list[str] = leakage_cols()

# Parent tmQMg ground-state descriptors that ARE permitted as Regime-B inputs
# (PBE0/PBE-D3BJ quantities; do not require the excited-state calculation).
PARENT_DESCRIPTOR_COLS: list[str] = [
    "charge",
    "molecular_mass",
    "n_atoms",
    "n_electrons",
    "tzvp_homo_energy",
    "tzvp_lumo_energy",
    "tzvp_homo_lumo_gap",
    "tzvp_dipole_moment",
    "polarisability",
]
