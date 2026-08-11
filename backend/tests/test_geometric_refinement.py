import numpy as np

from backend.segmentation.geometric_refinement import refine_gaussian_selection


def _geometry() -> np.ndarray:
    points = []
    for y in range(3):
        for x in range(3):
            points.append([x * 0.01, y * 0.01, 0, 0.7, 0.2, 0.1, 0.006, 0.9])
    points.append([1, 1, 1, 0.7, 0.2, 0.1, 0.006, 0.9])
    points.append([0.015, 0.015, 0, 0.7, 0.2, 0.1, 0.006, 0.01])
    return np.asarray(points, dtype=np.float32)


def test_refinement_grows_into_supported_local_candidate():
    result = refine_gaussian_selection(_geometry(), list(range(8)), [8, 9, 10], scene_radius=1)

    assert 8 in result["indices"]
    assert 9 not in result["indices"]
    assert 10 not in result["indices"]
    assert result["added_count"] == 1
    assert result["engine"] == "open3d-style-scipy"


def test_refinement_is_deterministic_and_growth_is_bounded():
    geometry = _geometry()
    first = refine_gaussian_selection(geometry, list(range(8)), [8], 1)
    second = refine_gaussian_selection(geometry, list(range(8)), [8], 1)

    assert first["indices"] == second["indices"]
    assert first["added_count"] <= 2
