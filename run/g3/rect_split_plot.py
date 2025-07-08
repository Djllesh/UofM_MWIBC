"""
Illia Prykhodko
University of Manitoba
July 7th, 2025
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from scipy.signal import find_peaks

from umbms import get_proj_path, get_script_logger, verify_path
from umbms.loadsave import load_pickle
from umbms.plot.sinogramplot import plt_sino
from umbms.beamform.iczt import iczt
from umbms.beamform.utility import rect
from umbms.boundary.boundary_detection import (
    rho_ToR_from_td,
    time_aligned_kernel,
)

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), "data/umbmid/g3/")

__OUT_DIR = os.path.join(get_proj_path(), "output/g3/msc_thesis/")
verify_path(__OUT_DIR)

__FD_NAME = "fd_data_gen_three_s11.pickle"
__MD_NAME = "metadata_gen_three.pickle"

# The frequency parameters from the scan
__INI_F = 2e9
__FIN_F = 9e9
__N_FS = 1001
__SCAN_FS = np.linspace(__INI_F, __FIN_F, __N_FS)

# The time-domain conversion parameters
__INI_T = 0.5e-9
__FIN_T = 5.5e-9
__N_TS = 700

# The size of the reconstructed image along one dimension
__M_SIZE = 150

__VAC_SPEED = scipy.constants.speed_of_light
# Define propagation speed in vacuum


def load_data():
    """Loads both fd and metadata

    Returns:
    --------
    tuple of two loaded variables
    """
    return load_pickle(os.path.join(__DATA_DIR, __FD_NAME)), load_pickle(
        os.path.join(__DATA_DIR, __MD_NAME)
    )


###############################################################################


if __name__ == "__main__":
    logger = get_script_logger(__file__)

    # Load the frequency domain data and metadata
    fd_data, metadata = load_data()

    n_expts = np.size(fd_data, axis=0)  # The number of individual scans

    ii = 196
    ant_pos = 0

    logger.info("Scan [%3d / %3d]..." % (ii + 1, n_expts))

    # Get the frequency domain data and metadata of this experiment
    tar_fd = fd_data[ii, :, :]

    # Get the unique ID of each experiment / scan
    expt_ids = [md["id"] for md in metadata]

    tar_md = metadata[ii]

    scan_rad = tar_md["ant_rad"] / 100

    # Correct for how the scan radius is measured (from a
    # point on the antenna stand, not at the SMA connection
    # point)
    scan_rad += 0.03618
    # Get the adipose-only and empty reference data
    # for this scan
    adi_fd_emp = fd_data[expt_ids.index(tar_md["emp_ref_id"]), :, :]
    adi_cal_cropped_emp = tar_fd - adi_fd_emp
    adi_cal_cropped_emp = adi_cal_cropped_emp[__SCAN_FS >= 2e9, :]

    sinogram = iczt(
        adi_cal_cropped_emp,
        ini_t=__INI_T,
        fin_t=__FIN_T,
        ini_f=__INI_F,
        fin_f=__FIN_F,
        n_time_pts=__N_TS,
    )

    ts = np.linspace(__INI_T, __FIN_T, __N_TS)
    kernel = time_aligned_kernel(sinogram)

    _, skin_peaks = rho_ToR_from_td(
        td=sinogram,
        ts=ts,
        kernel=kernel,
        ant_rad=scan_rad,
        peak_threshold=10,
        plt_slices=False,
        out_dir="",
    )

    signal_td = np.abs(sinogram[:, ant_pos])
    peaks, _ = find_peaks(-signal_td)

    max_peak = np.argwhere(ts == skin_peaks[ant_pos])[0]

    peaks_to_decide = peaks - max_peak
    t_start_idx = max_peak + peaks_to_decide[peaks_to_decide < 0][-1]
    t_end_idx = max_peak + peaks_to_decide[peaks_to_decide > 0][0]

    t_start = ts[t_start_idx]
    t_end = ts[t_end_idx]

    plt.rc("font", family="Libertinus Serif")
    fig = plt.figure(constrained_layout=True, figsize=(12, 8), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[1, 1])

    titles = ["(a)", "(b)", "(c)", "(d)"]

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    ax4 = fig.add_subplot(gs[:, 1])

    for i, ax in enumerate([ax1, ax2, ax3, ax4], start=0):
        ax.set_title(titles[i], fontsize=18)
        ax.tick_params(labelsize=14)
        ax.grid(linewidth=0.7)

    pre_skin = rect(ts, t0=t_start / 2, a=t_start) * signal_td
    skin = (
        rect(ts, t0=t_start + (t_end - t_start) / 2, a=t_end - t_start)
        * signal_td
    )
    post_skin = (
        rect(ts, t0=t_end + (__FIN_T - t_end) / 2, a=__FIN_T - t_end)
        * signal_td
    )

    ax1.plot(ts * 1e9, pre_skin, "g-")
    ax2.plot(ts * 1e9, skin, "b-")
    ax3.plot(ts * 1e9, post_skin, "k-")
    ax4.plot(
        ts * 1e9,
        pre_skin + skin + post_skin,
        "k-",
        linewidth=1.1,
        label="Combined signal",
    )
    ax4.plot(ts * 1e9, signal_td, "r--", linewidth=1.0, label="True signal")

    ax1.label_outer()
    ax2.label_outer()
    ax3.set_xlabel("Time (ns)", fontsize=18)
    ax4.set_xlabel("Time (ns)", fontsize=18)
    ax4.legend(fontsize=15)

    fig.supylabel("Intensity (a.u.)", fontsize=18)
    plt.savefig(os.path.join(__OUT_DIR, "rect_split_signal.png"), dpi=300)
