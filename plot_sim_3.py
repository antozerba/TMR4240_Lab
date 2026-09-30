# plot_reference_model_comparison.py
# -----------------------------------------------------------------------------
# Not part of the graded interface. Runs the same setpoint-change scenario as
# Simulation 3 (10 m / 10 m / 270 deg step, no wind or current), once WITH
# the reference model and once WITHOUT it, using whatever controller and
# gains are currently committed. Produces time-series and XY plots for each,
# reusing simulation/plotter.py so the style matches what check.py's own
# demo plots already look like.
#
# Run: python plot_reference_model_comparison.py
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from part_1.config import SimConfig
from simulation.checks import _run
from simulation.plotter import plot_time_histories

TARGET_3 = [10.0, 10.0, 3 * np.pi / 2]  # [N, E, psi], same as Simulation 3
TARGET = np.zeros(6)
TARGET[0], TARGET[1], TARGET[5] = TARGET_3
T = 600.0


def plot_xy_with_reference(logs, title):
    """XY trajectory with the reference/setpoint path drawn as a full line
    in the SAME axes as the vessel's actual path, not just a point marker.

    If the reference never moves (use_reference=False -- logs.sp is the
    same point for the whole run), the "path" has zero length and would be
    invisible on its own, so an explicit marker is always added at the
    final reference point to guarantee it's visible either way.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(logs.sp[:, 1], logs.sp[:, 0], "--", color="C1", linewidth=1.5,
             label="reference / setpoint path")
    ax.plot(logs.sp[-1, 1], logs.sp[-1, 0], "*", color="C1", markersize=14,
             markeredgecolor="black", markeredgewidth=0.5, label="setpoint", zorder=5)
    ax.plot(logs.eta[:, 1], logs.eta[:, 0], color="C0", linewidth=1.5,
             label="actual trajectory")
    ax.plot(logs.eta[0, 1], logs.eta[0, 0], "o", color="green", markersize=8, label="start")
    ax.plot(logs.eta[-1, 1], logs.eta[-1, 0], "s", color="red", markersize=7, label="end")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def main():
    for use_reference, label, suffix in [(True, "WITH reference model", "with_ref"),
                                          (False, "WITHOUT reference model", "without_ref")]:
        print(f"Running setpoint change {label} (no wind/current)...")
        cfg = SimConfig(T=T, use_reference=use_reference)
        # current=None, wind=None -> both default to zero/inactive, per _run()
        logs = _run(cfg, TARGET, current=None, wind=None)

        fig1 = plot_time_histories(logs)
        fig1.suptitle(f"Position, heading, and velocities -- {label}")
        fig1.savefig(f"time_series_{suffix}.png", dpi=130)
        plt.close(fig1)

        fig2 = plot_xy_with_reference(logs, f"Horizontal-plane trajectory -- {label}")
        fig2.savefig(f"xy_{suffix}.png", dpi=130)
        plt.close(fig2)

        print(f"  saved time_series_{suffix}.png and xy_{suffix}.png")


if __name__ == "__main__":
    main()