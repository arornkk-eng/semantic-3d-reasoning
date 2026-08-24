import numpy as np
import pytest

from backend.segmentation.object_cleaning import ObjectCleaningError, clean_object_gaussians


def _attributes(count: int) -> tuple[np.ndarray, np.ndarray]:
    scales = np.full((count, 3), 0.01)
    rotations = np.zeros((count, 4))
    rotations[:, 0] = 1.0
    return scales, rotations


def test_depth_anchor_selects_observed_cluster_instead_of_largest_background():
    rng = np.random.default_rng(21)
    observed = rng.normal([0.0, 0.5, 0.0], [0.04, 0.12, 0.04], size=(240, 3))
    background = rng.normal([2.0, -0.5, 1.5], [0.35, 0.04, 0.35], size=(700, 3))
    positions = np.vstack([observed, background])
    scales, rotations = _attributes(len(positions))

    result = clean_object_gaussians(
        positions,
        scales,
        rotations,
        observation_anchor=np.array([0.0, 0.5, 0.0]),
    )

    assert np.linalg.norm(np.median(result.positions, axis=0) - [0.0, 0.5, 0.0]) < 0.1
    assert len(result.positions) < len(background)
    assert result.report["status"] == "warning"
    assert result.report["retained_ratio"] < 0.5
    assert any(
        item["code"] == "SEMANTIC_GAUSSIAN_CONTAMINATION"
        for item in result.report["warnings"]
    )


def test_clean_single_component_retains_dense_object():
    rng = np.random.default_rng(22)
    positions = rng.normal([1.0, 2.0, 3.0], [0.1, 0.2, 0.1], size=(500, 3))
    scales, rotations = _attributes(len(positions))

    result = clean_object_gaussians(positions, scales, rotations)

    assert result.report["retained_ratio"] > 0.8
    assert np.linalg.norm(np.median(result.positions, axis=0) - [1.0, 2.0, 3.0]) < 0.05


def test_cleaning_rejects_too_few_gaussians():
    positions = np.zeros((10, 3))
    scales, rotations = _attributes(len(positions))

    with pytest.raises(ObjectCleaningError, match="少于 20"):
        clean_object_gaussians(positions, scales, rotations)
