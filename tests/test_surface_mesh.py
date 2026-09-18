"""
Tests for :func:`src.ui.utils.plotting.build_surface_mesh`.

The mesh builder is the pure, Streamlit-free core behind the Step 6
Prediction Profiler contour / 3D surface plots. The returned mesh must be
oriented the way plotly ``go.Contour``/``go.Surface`` expects it:
``Z[j, i]`` is the predicted response at ``(x = x_grid[i], y = y_grid[j])``
(rows follow Y, columns follow X). A transposed mesh is exactly the
reported issue #36 symptom (contour axes appear swapped).
"""

import numpy as np
import pytest

from src.ui.utils.plotting import build_surface_mesh


def _direct(x, y):
    return 2.0 * x + 7.0 * y


class TestBuildSurfaceMesh:
    @pytest.mark.parametrize(
        "x_grid,y_grid",
        [
            (np.linspace(0, 10, 25), np.linspace(100, 200, 17)),
            (np.linspace(-3, 3, 8), np.linspace(0.5, 2.5, 31)),
        ],
    )
    def test_shape_follows_y_then_x(self, x_grid, y_grid):
        base = {"held": 4.0}
        Z = build_surface_mesh(
            x_grid, y_grid, "X", "Y", base, lambda pt: _direct(pt["X"], pt["Y"])
        )
        assert Z.shape == (len(y_grid), len(x_grid))

    @pytest.mark.parametrize(
        "x_grid,y_grid",
        [
            (np.linspace(0, 10, 25), np.linspace(100, 200, 17)),
            (np.linspace(-3, 3, 8), np.linspace(0.5, 2.5, 31)),
        ],
    )
    def test_values_match_plotly_row_y_col_x(self, x_grid, y_grid):
        Z = build_surface_mesh(
            x_grid, y_grid, "X", "Y", {},
            lambda pt: _direct(pt["X"], pt["Y"]),
        )
        for i, j in [(0, 0), (-1, -1), (-1, 0), (0, -1), (3, 5), (len(x_grid) // 2, len(y_grid) // 2)]:
            assert Z[j, i] == pytest.approx(_direct(x_grid[i], y_grid[j]))
            if i != j and j < len(x_grid) and i < len(y_grid):
                assert Z[j, i] != pytest.approx(_direct(x_grid[j], y_grid[i]))

    def test_asymmetric_regression_guard(self):
        n = 17
        x_grid = np.linspace(0, 10, n)
        y_grid = np.linspace(100, 200, n)
        Z = build_surface_mesh(
            x_grid, y_grid, "X", "Y", {},
            lambda pt: _direct(pt["X"], pt["Y"]),
        )
        assert Z.shape == (n, n)
        assert not np.allclose(Z, Z.T)

    def test_base_settings_are_used_and_not_mutated(self):
        base = {"held": 4.0, "X": 999.0, "Y": 999.0}
        expected_base = dict(base)
        x_grid = np.linspace(0, 2, 4)
        y_grid = np.linspace(0, 3, 5)

        def predict(pt):
            assert pt["held"] == 4.0
            return pt["X"] * 10 + pt["Y"] * 100

        Z = build_surface_mesh(x_grid, y_grid, "X", "Y", base, predict)
        assert Z[0, 0] == pytest.approx(x_grid[0] * 10 + y_grid[0] * 100)
        assert Z[-1, -1] == pytest.approx(x_grid[-1] * 10 + y_grid[-1] * 100)
        assert base == expected_base

    def test_grid_requires_more_than_one_value(self):
        with pytest.raises(ValueError):
            build_surface_mesh(
                np.linspace(0, 0, 5), np.linspace(0, 1, 5), "X", "Y", {},
                lambda pt: 0.0,
            )
        with pytest.raises(ValueError):
            build_surface_mesh(
                np.linspace(0, 1, 5), np.linspace(0, 0, 5), "X", "Y", {},
                lambda pt: 0.0,
            )

    def test_empty_grid_rejected(self):
        with pytest.raises(ValueError):
            build_surface_mesh(
                np.array([]), np.linspace(0, 1, 5), "X", "Y", {},
                lambda pt: 0.0,
            )

    def test_2d_grids_rejected(self):
        with pytest.raises(ValueError):
            build_surface_mesh(
                np.zeros((3, 3)), np.linspace(0, 1, 5), "X", "Y", {},
                lambda pt: 0.0,
            )