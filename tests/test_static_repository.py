from pathlib import Path
import ast
import re

ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCT = ROOT / "scripts" / "reconstruct_article_ljubescica.py"
BENCHMARK = ROOT / "scripts" / "benchmark_rf_svm_manual_and_pseudo.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_public_scripts_are_valid_python():
    for path in [RECONSTRUCT, BENCHMARK]:
        ast.parse(read(path), filename=str(path))


def test_reconstruction_uses_environment_credentials():
    text = read(RECONSTRUCT)
    assert 'os.getenv("SENTINELHUB_CLIENT_ID")' in text
    assert 'os.getenv("SENTINELHUB_CLIENT_SECRET")' in text
    assert re.search(r"CLIENT_(ID|SECRET)\s*=\s*['\"]", text) is None


def test_article_parameters_are_locked_in_reconstruction_script():
    text = read(RECONSTRUCT)
    expected = [
        "NDVI_PCTL = 30",
        "BSI_PCTL = 65",
        "NDBI_PCTL = 65",
        'REF_MONTHS = ["2017-07", "2017-10", "2024-08"]',
        "SIGNATURE_DIST_THR = 4.5",
        "KEEP_TOP_COMPONENTS = 3",
        "MIN_OBJECT_PIXELS = 200",
        "CLOSING_RADIUS = 5",
        "MASK_ARTIFACT_FRACTION_THR = 0.45",
        "MAX_REASONABLE_AREA_HA = 40.0",
        'MANUAL_EXCLUDE_MONTHS = {"2018-07", "2022-07"}',
    ]
    for item in expected:
        assert item in text


def test_benchmark_script_states_validation_limits():
    text = read(BENCHMARK).lower()
    assert "manual_cross_year" in text
    assert "pseudo_to_manual" in text
    assert "not independent validation" in text
    assert "rgb proxy" in text


def test_data_folder_documents_expected_inputs():
    readme = ROOT / "data" / "README.md"
    assert readme.exists()
    text = read(readme).lower()
    assert "sentinel" in text
    assert "manual" in text or "benchmark" in text
