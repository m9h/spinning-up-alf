"""Numerical checks of the code and plots that students actually execute."""

import contextlib
import io
import json
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alf.benchmarks.t_maze import ACT_CUE, ACT_LEFT, ACTION_NAMES, build_t_maze_model
from utils.plotting import plot_efe_decomposition

ROOT = Path(__file__).resolve().parents[1]


class LessonScienceTests(unittest.TestCase):
    def test_plot_total_uses_minimized_efe_convention(self):
        pragmatic, epistemic = np.array([1.0, -0.5]), np.array([0.2, 0.0])
        ax = plot_efe_decomposition(pragmatic, epistemic)
        try:
            np.testing.assert_allclose(ax.lines[0].get_ydata(), -pragmatic - epistemic)
        finally:
            plt.close(ax.figure)

    def test_manual_rollout_uses_information_gain(self):
        notebook = json.loads((ROOT / "notebooks/10_efe_decomposition.ipynb").read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if "=== Per-Step Analysis:" in "".join(cell["source"]))
        namespace = dict(np=np, gm=build_t_maze_model(), ACT_CUE=ACT_CUE,
                         ACT_LEFT=ACT_LEFT, ACTION_NAMES=ACTION_NAMES)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, "module10_manual_rollout", "exec"), namespace)
        # Final action reaches an arm: its outcome reveals one bit about
        # reward location, despite zero conditional observation entropy.
        np.testing.assert_allclose(namespace["epistemic"], np.log(2), atol=1e-10)
        np.testing.assert_allclose(namespace["pragmatic"], 0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
