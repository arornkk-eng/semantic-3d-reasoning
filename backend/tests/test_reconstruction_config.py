from backend.core.config import (
    MAX_RECONSTRUCTION_IMAGES,
    SCENE_ALPHA_THRESHOLD,
    SCENE_OUTLIER_PERCENTILE,
    SPLAT_SCALE_FACTOR,
)


def test_evidence_backed_reconstruction_defaults():
    assert MAX_RECONSTRUCTION_IMAGES == 12
    assert SCENE_ALPHA_THRESHOLD == 0.02
    assert SCENE_OUTLIER_PERCENTILE == 1
    assert SPLAT_SCALE_FACTOR == 1.0
