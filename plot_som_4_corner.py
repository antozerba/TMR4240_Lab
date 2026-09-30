# visualize_four_corner.py
# -----------------------------------------------------------------------------
# Not part of the graded interface. Runs the four-corner test (Simulation 4)
# exactly as check.py does, using whatever controller, gains, and thrust
# allocator are currently committed in part_1/ -- no changes made here.
#
# Produces one figure: NED trajectory (with heading arrows) on the left,
# North / East / heading time series on the right.
#
# Run: python visualize_four_corner.py
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from simulation.checks import check_sim4_four_corner

CORNERS = [
    ("eta1 [50,0,0]", [50.0, 0.0, 0.0]),
    ("eta2 [50,-50,0]", [50.0, -50.0, 0.0]),
    ("eta3 [50,-50,-pi/4]", [50.0, -50.0, -np.pi / 4]),
    ("eta4 [0,-50,-pi/4]", [0.0, -50.0, -np.pi / 4]),
    ("eta5 [0,0,0]", [0.0, 0.0, 0.0]),
]
HOLD = 300.0  # [s] per leg, matches check_sim4_four_corner


def main():
    result = check_sim4_four_corner()
    print(result.status, result.name)
    for line in result.details:
        print(" ", line)

    logs = result.logs["four-corner test"]
    dt = logs.t[1] - logs.t[0]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1.4])

    # --- Left: NED trajectory with heading arrows ---
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.plot(logs.eta[:, 1], logs.eta[:, 0], color="C0", linewidth=1.5,
                 label="actual path", zorder=2)

    corner_xy = [(0.0, 0.0)] + [(c[1], c[0]) for _, c in CORNERS]
    ax_traj.plot([p[0] for p in corner_xy], [p[1] for p in corner_xy],
                 color="C1", linestyle="--", linewidth=1, label="commanded path", zorder=1)
    for i, (label, c) in enumerate(CORNERS):
        ax_traj.plot(c[1], c[0], "o", color="C1", markersize=6, zorder=3)
        ax_traj.annotate(f"{i+1}", (c[1], c[0]), textcoords="offset points",
                          xytext=(6, 6), fontsize=9)
    ax_traj.plot(0, 0, "o", color="green", markersize=8, label="start", zorder=3)

    # Heading arrows every 75 s. NED heading psi (0 = North, clockwise
    # positive) maps to plot-plane direction (sin(psi), cos(psi)) since the
    # plot's x-axis is East and y-axis is North.
    step = int(round(75.0 / dt))
    idx = np.arange(0, len(logs.t), step)
    psi, E, N = logs.eta[idx, 5], logs.eta[idx, 1], logs.eta[idx, 0]
    arrow_len = 3.0
    ax_traj.quiver(E, N, arrow_len * np.sin(psi), arrow_len * np.cos(psi),
                   angles="xy", scale_units="xy", scale=1.0,
                   color="crimson", width=0.004, zorder=4, label="heading")

    ax_traj.set_xlabel("East [m]")
    ax_traj.set_ylabel("North [m]")
    ax_traj.set_title("Trajectory (NED) with heading")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.legend(loc="best", fontsize=9)
    ax_traj.grid(True, linewidth=0.4)

    # --- Right: time series, N / E / heading ---
    ax_n = fig.add_subplot(gs[0, 1])
    ax_n.plot(logs.t, logs.eta[:, 0], color="C0")
    for i, (_, c) in enumerate(CORNERS):
        ax_n.hlines(c[0], i * HOLD, (i + 1) * HOLD, colors="C1", linestyles="--")
    ax_n.set_ylabel("North [m]")
    ax_n.legend(["actual", "setpoint"], fontsize=8)

    ax_e = fig.add_subplot(gs[1, 1], sharex=ax_n)
    ax_e.plot(logs.t, logs.eta[:, 1], color="C0")
    for i, (_, c) in enumerate(CORNERS):
        ax_e.hlines(c[1], i * HOLD, (i + 1) * HOLD, colors="C1", linestyles="--")
    ax_e.set_ylabel("East [m]")

    ax_psi = fig.add_subplot(gs[2, 1], sharex=ax_n)
    ax_psi.plot(logs.t, np.rad2deg(logs.eta[:, 5]), color="C0")
    for i, (_, c) in enumerate(CORNERS):
        ax_psi.hlines(np.rad2deg(c[2]), i * HOLD, (i + 1) * HOLD, colors="C1", linestyles="--")
    ax_psi.set_ylabel("Heading [deg]")
    ax_psi.set_xlabel("Time [s]")

    for i in range(1, len(CORNERS)):
        for ax in [ax_n, ax_e, ax_psi]:
            ax.axvline(i * HOLD, color="gray", linewidth=0.7, linestyle=":")

    fig.suptitle("Four-corner test: trajectory (with heading) and time series")
    fig.tight_layout()
    fig.savefig("four_corner_visualization_main.png", dpi=130)
    print("\nSaved four_corner_visualization_main.png")

    
if __name__ == "__main__":
    main()