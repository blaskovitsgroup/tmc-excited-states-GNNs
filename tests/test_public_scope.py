from pathlib import Path
import json

from predictor_webapp.run_painn import environment_python
from tmqmg_es.models.excitonnet import BACKBONES
from tmqmg_es.train.trainer import MODEL_SPECS
from tools.retrain import MODELS


def test_exported_models_are_exactly_the_paper_configurations():
    expected = {
        "xgboost_descriptors", "schnet_3d", "painn_3d", "graph_baseline",
        "graph_unatq", "graph_dnatq", "painn_baseline_fusion",
        "painn_unatq_fusion", "painn_dnatq_fusion",
    }
    assert set(MODELS) == expected
    assert set(MODEL_SPECS) == expected - {"xgboost_descriptors"}
    assert set(BACKBONES) == {"schnet", "painn"}


def test_environment_paths_cover_windows_and_posix(tmp_path):
    assert environment_python(tmp_path, windows=True) == tmp_path / ".venv/Scripts/python.exe"
    assert environment_python(tmp_path, windows=False) == tmp_path / ".venv/bin/python"


def test_retained_comparisons_have_registered_models():
    root = Path(__file__).parents[1]
    for name in ("outputs/results/locked_test_results.json", "manifests/locked_evaluation_release.json"):
        record = json.loads((root / name).read_text())
        comparisons = record["confirmatory_comparisons"]
        assert len(comparisons) == 2
        assert record["decision_rule"]["preregistered_family_size"] == 3
        for comparison in comparisons:
            assert comparison["model_A"] in MODELS
            assert comparison["model_B"] in MODELS


def test_distributed_configs_have_no_machine_specific_paths():
    import yaml
    root = Path(__file__).parents[1]
    for p in (root / "configs/models").glob("*.yaml"):
        cfg = yaml.safe_load(p.read_text())
        for key in ["cache_dir", "natqg_root", "development_table", "development_split", "data_fingerprint_file", "out_dir"]:
            if key in cfg:
                assert not Path(cfg[key]).is_absolute(), (p, key)
