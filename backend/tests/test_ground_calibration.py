import numpy as np
import pytest

from backend.physics import ground_calibration


def test_ransac_fits_selected_ground_layer():
    rng = np.random.default_rng(8)
    ground = np.column_stack(
        [rng.uniform(-2, 2, 1200), rng.normal(0.4, 0.002, 1200), rng.uniform(-1, 1, 1200)]
    )
    outliers = rng.uniform([-2, -1, -1], [2, 2, 1], size=(120, 3))

    origin, normal, inliers, ratio, rmse = ground_calibration._fit_plane_ransac(
        np.vstack([ground, outliers])
    )

    assert origin[1] == pytest.approx(0.4, abs=0.01)
    assert normal[1] > 0.99
    assert len(inliers) > 1100
    assert ratio > 0.8
    assert rmse < 0.01


def test_three_points_flip_and_confirm(monkeypatch):
    stored = {}

    def save(_task_id, value):
        stored.clear()
        stored.update(value)
        return value

    monkeypatch.setattr(ground_calibration, "save_ground_calibration", save)
    monkeypatch.setattr(
        ground_calibration,
        "get_ground_calibration",
        lambda _task_id: dict(stored) if stored else None,
    )
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]

    created = ground_calibration.calibrate_from_points("task1", points)
    original_normal = np.asarray(created["normal"])
    assert created["method"] == "three_points"
    assert created["confirmed"] is False

    flipped = ground_calibration.flip_ground_normal("task1")
    assert np.allclose(flipped["normal"], -original_normal)
    assert flipped["flipped"] is True
    assert flipped["confirmed"] is False

    confirmed = ground_calibration.confirm_ground_calibration("task1")
    assert confirmed["confirmed"] is True
    assert confirmed["confirmed_at"]


def test_three_points_reject_collinear_input(monkeypatch):
    monkeypatch.setattr(ground_calibration, "save_ground_calibration", lambda *_args: None)
    with pytest.raises(ground_calibration.GroundCalibrationError, match="共线"):
        ground_calibration.calibrate_from_points(
            "task1", [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )
