# plot_sim2_analysis.py
# -----------------------------------------------------------------------------
# Not part of the graded interface. Runs Simulation 2 (current rotating
# north -> east over 300 s), using whatever controller/gains are currently
# committed. Plots position, heading, and the actual NED current velocity
# components V_c,N(t), V_c,E(t) as returned to the plant -- reconstructed
# from the logged speed/direction (Uc, beta_c) via
#     V_c,N = Uc * cos(beta_c),   V_c,E = Uc * sin(beta_c)
# (verified against Simulation 1a: "0.5 m/s from east" logs beta_c = -90 deg,
# giving [V_N, V_E] = [0, -0.5] -- i.e. flowing TOWARD west, matching the
# "from east" semantics check.py itself tests for.)
#
# Run: python plot_sim2_analysis.py
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from simulation.checks import check_sim2_rotating_current

ROTATION_DURATION = 300.0  # [s], matches Current(..., duration=300.0) in check.py


def main():
    result = check_sim2_rotating_current()
    print(result.status, result.name)
    for line in result.details:
        print(" ", line)

    logs = result.logs["rotating current"]
    t = logs.t
    N, E, psi = logs.eta[:, 0], logs.eta[:, 1], logs.eta[:, 5]
    Vc_N = logs.Uc * np.cos(logs.beta_c)
    Vc_E = logs.Uc * np.sin(logs.beta_c)

    print(f"\nCurrent direction check: at t=0, beta_c={np.rad2deg(logs.beta_c[0]):.1f} deg "
          f"-> [Vc_N, Vc_E] = [{Vc_N[0]:.2f}, {Vc_E[0]:.2f}] m/s")
    print(f"At t={ROTATION_DURATION:.0f}s (end of rotation), "
          f"beta_c={np.rad2deg(logs.beta_c[int(ROTATION_DURATION/(t[1]-t[0]))]):.1f} deg "
          f"-> [Vc_N, Vc_E] = [{Vc_N[int(ROTATION_DURATION/(t[1]-t[0]))]:.2f}, "
          f"{Vc_E[int(ROTATION_DURATION/(t[1]-t[0]))]:.2f}] m/s")

    fig, axs = plt.subplots(5, 1, figsize=(9, 12), sharex=True)

    axs[0].plot(t, N, color="C0")
    axs[0].axhline(0.0, color="C1", linestyle="--")
    axs[0].set_ylabel("North [m]")

    axs[1].plot(t, E, color="C0")
    axs[1].axhline(0.0, color="C1", linestyle="--")
    axs[1].set_ylabel("East [m]")

    axs[2].plot(t, np.rad2deg(psi), color="C0")
    axs[2].axhline(0.0, color="C1", linestyle="--")
    axs[2].set_ylabel("Heading [deg]")

    axs[3].plot(t, Vc_N, color="C2", label="$V_{c,N}$")
    axs[3].plot(t, Vc_E, color="C3", label="$V_{c,E}$")
    axs[3].set_ylabel("Current velocity [m/s]")
    axs[3].legend(loc="best")

    axs[4].plot(t, np.rad2deg(logs.beta_c), color="C4")
    axs[4].set_ylabel("Current direction\n(toward) [deg]")
    axs[4].set_xlabel("Time [s]")

    for ax in axs:
        ax.axvline(ROTATION_DURATION, color="gray", linewidth=0.8, linestyle=":")
        ax.grid(True, alpha=0.3)
    axs[0].set_title(f"Simulation 2: rotating current -- dotted line marks "
                      f"end of {ROTATION_DURATION:.0f} s rotation")

    fig.tight_layout()
    fig.savefig("sim2_analysis.png", dpi=130)
    print("\nSaved sim2_analysis.png")


if __name__ == "__main__":
    main()