import pytest

from tmqmg_es import config as C


def _data_present() -> bool:
    return C.STAR_CSV.exists() and C.PARENT_CSV.exists() and C.XYZ_DIR.exists()


requires_data = pytest.mark.skipif(
    not _data_present(), reason="tmQMg* source data not available locally"
)


@pytest.fixture(scope="session")
def phase1():
    """Run the source-data audit once for integration tests."""
    if not _data_present():
        pytest.skip("tmQMg* source data not available locally")
    from tmqmg_es.data.audit import build, index_xyz, load_parent, load_star

    star, parent = load_star(), load_parent()
    return build(star, parent, index_xyz())
