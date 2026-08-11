from backend.core.config import (
    DEFAULT_NUM_VIEWS,
    SCENE_ALPHA_THRESHOLD,
    SCENE_OUTLIER_PERCENTILE,
    SPLAT_SCALE_FACTOR,
)


def test_evidence_backed_reconstruction_defaults():
    assert DEFAULT_NUM_VIEWS == 6
    assert SCENE_ALPHA_THRESHOLD == 0.02
    assert SCENE_OUTLIER_PERCENTILE == 1
    assert SPLAT_SCALE_FACTOR == 1.0
