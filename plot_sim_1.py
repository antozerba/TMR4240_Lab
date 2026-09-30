# plot_sim1_analysis.py
# -----------------------------------------------------------------------------
# Not part of the graded interface. Runs Simulation 1a (current) and
# Simulation 1b (wind) separately, using whatever controller/gains are
# currently committed. For EACH case, produces:
#   - N(t), E(t), psi(t) time series
#   - XY trajectory
#   - the five discussion metrics: max position error, steady-state position
#     error, settling time, steady-state heading error, actuator effort
#
# Run: python plot_sim1_analysis.py
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from simulation.checks import check_sim1_current, check_sim1_wind

SETTLE_BAND_FRACTION = 0.05  # 5% of peak deviation, standard control-systems convention


def actuator_effort(logs):
    """Total commanded thrust magnitude across all thrusters, as a time
    series [kN]. Peak and RMS of this are the two effort numbers reported."""
    total_u = np.sum(np.abs(logs.u), axis=1) / 1000.0  # (N,) kN
    return total_u


def settling_time(t, pos_err, threshold):
    """Last moment pos_err exceeds `threshold` and stays under it for the
    rest of the run. Returns None if it never permanently settles."""
    above = np.where(pos_err > threshold)[0]
    if len(above) == 0:
        return 0.0
    if above[-1] == len(pos_err) - 1:
        return None  # never permanently settles
    return t[above[-1] + 1]


def analyze_and_plot(logs, target_ned, label, filename_prefix):
    t = logs.t
    N, E, psi = logs.eta[:, 0], logs.eta[:, 1], logs.eta[:, 5]
    pos_err = np.hypot(N - target_ned[0], E - target_ned[1])
    heading_err_deg = np.rad2deg(np.abs(((psi - target_ned[2] + np.pi) % (2 * np.pi)) - np.pi))
    effort = actuator_effort(logs)

    # last-100s steady-state window, matching check.py's own convention
    win = int(round(100.0 / (t[1] - t[0])))
    ss_pos_err = np.mean(pos_err[-win:])
    ss_heading_err = np.mean(heading_err_deg[-win:])
    max_pos_err = np.max(pos_err)
    # settling time: standard control-systems definition -- time after which
    # error stays within SETTLE_BAND_FRACTION of the peak deviation. A fixed
    # absolute threshold (e.g. 1.0 m) is meaningless here since the whole
    # disturbance-rejection transient never gets anywhere near that large.
    settle_band = max(SETTLE_BAND_FRACTION * max_pos_err, 1e-6)
    t_settle = settling_time(t, pos_err, settle_band)
    peak_effort = np.max(effort)
    rms_effort = np.sqrt(np.mean(effort ** 2))

    print(f"\n--- {label} ---")
    print(f"  Max position error:        {max_pos_err:.2f} m")
    print(f"  Steady-state position err: {ss_pos_err:.3f} m (last 100 s)")
    print(f"  Settling time (within {SETTLE_BAND_FRACTION*100:.0f}% of peak): "
          f"{t_settle:.1f} s" if t_settle is not None else "  Settling time: never settles")
    print(f"  Steady-state heading err:  {ss_heading_err:.3f} deg (last 100 s)")
    print(f"  Actuator effort: peak={peak_effort:.1f} kN, RMS={rms_effort:.1f} kN")

    # --- time series: N, E, psi ---
    fig1, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(t, N, color="C0")
    axs[0].axhline(target_ned[0], color="C1", linestyle="--")
    axs[0].set_ylabel("North [m]")
    axs[1].plot(t, E, color="C0")
    axs[1].axhline(target_ned[1], color="C1", linestyle="--")
    axs[1].set_ylabel("East [m]")
    axs[2].plot(t, np.rad2deg(psi), color="C0")
    axs[2].axhline(np.rad2deg(target_ned[2]), color="C1", linestyle="--")
    axs[2].set_ylabel("Heading [deg]")
    axs[2].set_xlabel("Time [s]")
    fig1.suptitle(f"Position and heading vs time -- {label}")
    fig1.tight_layout()
    fig1.savefig(f"{filename_prefix}_timeseries.png", dpi=130)
    plt.close(fig1)

    # --- xy trajectory ---
    fig2, ax = plt.subplots(figsize=(6, 6))
    ax.plot(E, N, color="C0", label="actual trajectory")
    ax.plot(target_ned[1], target_ned[0], "*", color="C1", markersize=14,
            markeredgecolor="black", markeredgewidth=0.5, label="target")
    ax.plot(E[0], N[0], "o", color="green", markersize=8, label="start")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"Trajectory -- {label}")
    fig2.tight_layout()
    fig2.savefig(f"{filename_prefix}_xy.png", dpi=130)
    plt.close(fig2)

    return dict(max_pos_err=max_pos_err, ss_pos_err=ss_pos_err, t_settle=t_settle,
                ss_heading_err=ss_heading_err, peak_effort=peak_effort, rms_effort=rms_effort)


def main():
    target = np.zeros(6)  # both Sim1a/1b station-keep at the origin

    r_current = check_sim1_current()
    logs_current = r_current.logs["current 0.5 m/s from east"]
    analyze_and_plot(logs_current, target[[0, 1, 5]], "Simulation 1a: current 0.5 m/s from east",
                      "sim1a_current")

    r_wind = check_sim1_wind()
    logs_wind = r_wind.logs["wind 15 m/s from east"]
    analyze_and_plot(logs_wind, target[[0, 1, 5]], "Simulation 1b: wind 15 m/s from east",
                      "sim1b_wind")


if __name__ == "__main__":
    main()