"""Shared visualization helpers for the Spinning Up in Active Inference curriculum.

Provides consistent styling across all 16 modules for:
- Value function heatmaps
- Belief evolution plots
- EFE decomposition bar charts
- Learning curves
- Factor graph diagrams
- RL↔AIF comparison panels
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Union
from IPython.display import display, HTML

# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "rl": "#2196F3",         # Blue for RL concepts
    "aif": "#FF5722",        # Orange for AIF concepts
    "reward": "#4CAF50",     # Green for reward/pragmatic
    "epistemic": "#9C27B0",  # Purple for epistemic/information
    "neutral": "#607D8B",    # Gray for neutral elements
    "highlight": "#FFC107",  # Gold for highlights
    "danger": "#F44336",     # Red for punishment/error
    "success": "#4CAF50",    # Green for success
}

RL_AIF_CMAP = LinearSegmentedColormap.from_list(
    "rl_aif", [COLORS["rl"], "#FFFFFF", COLORS["aif"]]
)


def _setup_style():
    """Apply consistent matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 100,
    })


def plot_value_heatmap(
    values: np.ndarray,
    grid_size: int,
    title: str = "Value Function",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    show_values: bool = True,
    goal_pos: Optional[tuple] = None,
    agent_pos: Optional[tuple] = None,
) -> plt.Axes:
    """Plot a value function as a grid heatmap.

    Args:
        values: 1D array of state values (grid_size^2,) or 2D (grid_size, grid_size).
        grid_size: Side length of the grid.
        title: Plot title.
        cmap: Matplotlib colormap name.
        ax: Optional axes to draw on.
        show_values: Annotate cells with numeric values.
        goal_pos: (row, col) to mark with a star.
        agent_pos: (row, col) to mark with a circle.
    """
    _setup_style()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5))

    if values.ndim == 1:
        grid = values.reshape(grid_size, grid_size)
    else:
        grid = values

    im = ax.imshow(grid, cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8)

    if show_values and grid_size <= 12:
        for i in range(grid_size):
            for j in range(grid_size):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if grid[i, j] < grid.mean() else "black")

    if goal_pos is not None:
        ax.plot(goal_pos[1], goal_pos[0], marker="*", color=COLORS["highlight"],
                markersize=20, markeredgecolor="black")

    if agent_pos is not None:
        ax.plot(agent_pos[1], agent_pos[0], marker="o", color=COLORS["rl"],
                markersize=15, markeredgecolor="black")

    ax.set_title(title)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    return ax


def plot_belief_evolution(
    belief_history: list,
    state_names: Optional[list] = None,
    title: str = "Belief Evolution",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot how beliefs over hidden states evolve over time.

    Args:
        belief_history: List of belief arrays, one per timestep.
        state_names: Labels for each state.
        title: Plot title.
    """
    _setup_style()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 4))

    beliefs = np.array(belief_history)
    n_states = beliefs.shape[1]
    if state_names is None:
        state_names = [f"s{i}" for i in range(n_states)]

    cmap = plt.cm.Set2
    for i in range(n_states):
        ax.plot(beliefs[:, i], label=state_names[i], color=cmap(i / n_states),
                linewidth=2)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("P(state)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    return ax


def plot_efe_decomposition(
    pragmatic: Union[np.ndarray, list],
    epistemic: Union[np.ndarray, list],
    action_names: Optional[list] = None,
    title: str = "EFE Decomposition",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Bar chart decomposing EFE into pragmatic and epistemic components.

    Args:
        pragmatic: Pragmatic value per action (higher = more preferred).
        epistemic: Epistemic value per action (higher = more informative).
        action_names: Labels for each action.
    """
    _setup_style()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    pragmatic = np.asarray(pragmatic)
    epistemic = np.asarray(epistemic)
    n_actions = len(pragmatic)
    if action_names is None:
        action_names = [f"a{i}" for i in range(n_actions)]

    x = np.arange(n_actions)
    width = 0.35

    ax.bar(x - width / 2, pragmatic, width, label="Pragmatic (reward-seeking)",
           color=COLORS["reward"], alpha=0.85)
    ax.bar(x + width / 2, epistemic, width, label="Epistemic (info-seeking)",
           color=COLORS["epistemic"], alpha=0.85)

    # Total EFE line
    total = pragmatic + epistemic
    ax.plot(x, total, "ko-", label="Total G(a)", linewidth=2, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(action_names, rotation=30, ha="right")
    ax.set_ylabel("Free Energy (nats)")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_learning_curve(
    curves: dict,
    xlabel: str = "Episode",
    ylabel: str = "Cumulative Reward",
    title: str = "Learning Curves",
    window: int = 10,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot smoothed learning curves for multiple agents.

    Args:
        curves: Dict mapping agent_name -> list of per-episode values.
        window: Smoothing window size.
    """
    _setup_style()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))

    for name, vals in curves.items():
        vals = np.array(vals, dtype=float)
        if len(vals) >= window:
            smoothed = np.convolve(vals, np.ones(window) / window, mode="valid")
        else:
            smoothed = vals
        color = COLORS["rl"] if "rl" in name.lower() or "td" in name.lower() or "q" in name.lower() else COLORS["aif"]
        ax.plot(smoothed, label=name, linewidth=2, color=color)
        ax.fill_between(range(len(smoothed)), smoothed * 0.9, smoothed * 1.1,
                        alpha=0.1, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax


def plot_rpe_trace(
    rpes: list,
    title: str = "Reward Prediction Errors",
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot reward prediction errors over time (dopamine analogy).

    Args:
        rpes: List of RPE values per timestep.
    """
    _setup_style()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 3))

    rpes = np.array(rpes)
    if color is None:
        color = COLORS["rl"]

    ax.bar(range(len(rpes)), rpes,
           color=[COLORS["success"] if r > 0 else COLORS["danger"] for r in rpes],
           alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("δ (RPE)")
    ax.set_title(title)
    return ax


def plot_matrix_heatmap(
    matrix: np.ndarray,
    title: str = "Matrix",
    row_labels: Optional[list] = None,
    col_labels: Optional[list] = None,
    cmap: str = "Blues",
    ax: Optional[plt.Axes] = None,
    fmt: str = ".2f",
) -> plt.Axes:
    """Heatmap for A, B, or transition matrices.

    Args:
        matrix: 2D array to visualize.
        row_labels: Labels for rows.
        col_labels: Labels for columns.
    """
    _setup_style()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5))

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8)

    if matrix.shape[0] <= 12 and matrix.shape[1] <= 12:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:{fmt}}", ha="center", va="center",
                        fontsize=8)

    if row_labels:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
    if col_labels:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=9, rotation=45, ha="right")

    ax.set_title(title)
    return ax


def plot_policy_distribution(
    policy_probs: np.ndarray,
    action_names: Optional[list] = None,
    title: str = "Policy Distribution",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Bar chart of action probabilities.

    Args:
        policy_probs: Array of action probabilities.
        action_names: Labels for each action.
    """
    _setup_style()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    n = len(policy_probs)
    if action_names is None:
        action_names = [f"a{i}" for i in range(n)]

    colors = [COLORS["aif"] if p == max(policy_probs) else COLORS["neutral"]
              for p in policy_probs]
    ax.bar(range(n), policy_probs, color=colors, alpha=0.85)
    ax.set_xticks(range(n))
    ax.set_xticklabels(action_names, rotation=30, ha="right")
    ax.set_ylabel("P(action)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    return ax


def plot_comparison_curves(
    rl_data: dict,
    aif_data: dict,
    metric: str = "reward",
    title: str = "RL vs AIF Comparison",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Side-by-side RL vs AIF performance comparison.

    Args:
        rl_data: Dict with 'mean' and optionally 'std' arrays.
        aif_data: Dict with 'mean' and optionally 'std' arrays.
        metric: Y-axis label.
    """
    _setup_style()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))

    rl_mean = np.array(rl_data["mean"])
    aif_mean = np.array(aif_data["mean"])

    ax.plot(rl_mean, label="RL Agent", color=COLORS["rl"], linewidth=2)
    ax.plot(aif_mean, label="AIF Agent", color=COLORS["aif"], linewidth=2)

    if "std" in rl_data:
        rl_std = np.array(rl_data["std"])
        ax.fill_between(range(len(rl_mean)), rl_mean - rl_std, rl_mean + rl_std,
                        alpha=0.15, color=COLORS["rl"])
    if "std" in aif_data:
        aif_std = np.array(aif_data["std"])
        ax.fill_between(range(len(aif_mean)), aif_mean - aif_std, aif_mean + aif_std,
                        alpha=0.15, color=COLORS["aif"])

    ax.set_xlabel("Episode")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    return ax


def plot_agent_trajectory(
    positions: list,
    grid_size: int,
    title: str = "Agent Trajectory",
    goal_pos: Optional[tuple] = None,
    walls: Optional[set] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot an agent's trajectory on a grid.

    Args:
        positions: List of (row, col) tuples.
        grid_size: Side length of the grid.
        goal_pos: (row, col) of goal.
        walls: Set of (row, col) wall positions.
    """
    _setup_style()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Draw grid
    grid = np.ones((grid_size, grid_size, 3))
    if walls:
        for (r, c) in walls:
            grid[r, c] = [0.3, 0.3, 0.3]

    ax.imshow(grid, interpolation="nearest")

    # Draw trajectory
    rows = [p[0] for p in positions]
    cols = [p[1] for p in positions]
    ax.plot(cols, rows, "o-", color=COLORS["rl"], linewidth=2, markersize=6, alpha=0.7)
    ax.plot(cols[0], rows[0], "s", color=COLORS["success"], markersize=14,
            markeredgecolor="black", label="Start")
    ax.plot(cols[-1], rows[-1], "D", color=COLORS["highlight"], markersize=14,
            markeredgecolor="black", label="End")

    if goal_pos is not None:
        ax.plot(goal_pos[1], goal_pos[0], "*", color=COLORS["highlight"],
                markersize=20, markeredgecolor="black", label="Goal")

    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    return ax


def dual_perspective_box(rl_text: str, aif_text: str, title: str = "Dual Perspective"):
    """Display an RL↔AIF comparison box in a Jupyter notebook.

    Args:
        rl_text: The RL perspective explanation.
        aif_text: The AIF perspective explanation.
        title: Box title.
    """
    html = f"""
    <div style="display: flex; border: 2px solid #333; border-radius: 8px;
                margin: 15px 0; overflow: hidden; font-family: sans-serif;">
        <div style="flex: 1; padding: 15px; background: {COLORS['rl']}15;
                    border-right: 1px solid #ccc;">
            <h4 style="color: {COLORS['rl']}; margin-top: 0;">
                🎯 RL Perspective
            </h4>
            <p style="font-size: 14px; line-height: 1.5;">{rl_text}</p>
        </div>
        <div style="flex: 1; padding: 15px; background: {COLORS['aif']}15;">
            <h4 style="color: {COLORS['aif']}; margin-top: 0;">
                🧠 Active Inference Perspective
            </h4>
            <p style="font-size: 14px; line-height: 1.5;">{aif_text}</p>
        </div>
    </div>
    <p style="text-align: center; font-weight: bold; color: #333;
              font-family: sans-serif;">{title}</p>
    """
    display(HTML(html))
