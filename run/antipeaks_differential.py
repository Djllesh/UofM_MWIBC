"""
Illia Prykhodko
University of Manitoba
June 24th, 2025
"""

import os
import numpy as np
import pandas
import multiprocessing as mp
import scipy.constants
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle

from umbms.plot.imgplots import plot_fd_img
from umbms.plot.sinogramplot import plt_sino

from umbms.beamform.das import fd_das
from umbms.beamform.time_delay import get_pix_ts_old
from umbms.beamform.utility import apply_ant_t_delay, get_fd_phase_factor

from umbms.boundary.boundary_detection import (
    rho_ToR_from_td,
    time_aligned_kernel,
)
from umbms.beamform.iczt import iczt

from umbms.beamform.propspeed import estimate_speed, get_breast_speed_freq

###############################################################################

__CPU_COUNT = mp.cpu_count()

__DATA_DIR = os.path.join(get_proj_path(), "data/umbmid/g3/")
__OUT_DIR = os.path.join(get_proj_path(), "output/")
verify_path(__OUT_DIR)

__FD_NAME = "g3_s11.pickle"
__MD_NAME = "metadata_gen_three.pickle"

# The frequency parameters from the scan
__INI_F = 2e9
__FIN_F = 9e9
__N_FS = 1001

# The time-domain conversion parameters
__INI_T = 0.5e-9
__FIN_T = 5.5e-9
__N_TS = 1000

# The size of the reconstructed image along one dimension
__M_SIZE = 150

# The approximate radius of each adipose phantom in our array
__ADI_RADS = {
    "A1": 0.05,
    "A2": 0.06,
    "A3": 0.07,
    "A11": 0.06,
    "A12": 0.05,
    "A13": 0.065,
    "A14": 0.06,
    "A15": 0.055,
    "A16": 0.07,
}

__GLASS_CYLINDER_RAD = 0.06
__PHANTOM_RAD = 0.0555
__GLASS_THIKNESS = 0.003
__SPHERE_RAD = 0.0075
__ROD_RAD = 0.002
__ANT_RAD = 0.21

__SPHERE_POS = [
    (np.nan, np.nan),
    (0.0, 4.0),
    (0.0, 3.0),
    (0.0, 2.0),
    (0.0, 1.0),
    (0.0, 0.0),
    (1.0, 0.0),
    (2.0, 0.0),
    (3.0, 0.0),
    (4.0, 0.0),
    (np.nan, np.nan),
    (4.3, 0.0),
    (3.3, 0.0),
    (2.3, 0.0),
    (1.3, 0.0),
    (0.3, 0.0),
    (0.0, 1.3),
    (0.0, 2.3),
    (0.0, 3.3),
    (0.0, 4.3),
    (np.nan, np.nan),
]

__MID_BREAST_RADS = {
    "A1": (0.053, 0.034),
    "A2": (0.055, 0.051),
    "A3": (0.07, 0.049),
    "A11": (0.062, 0.038),
    "A12": (0.051, 0.049),
    "A13": (0.065, 0.042),
    "A14": (0.061, 0.051),
    "A15": (0.06, 0.058),
    "A16": (0.073, 0.05),
}

# Define propagation speed in vacuum
__VAC_SPEED = scipy.constants.speed_of_light


def load_data():
    """Loads both fd_data and metadata

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

    # Get the unique ID of each experiment / scan
    expt_ids = [md["id"] for md in metadata]

    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

    iczt_ts = np.linspace(__INI_T, __FIN_T, __N_TS)

    # Only retain frequencies above 2 GHz due to antenna VSWR
    tar_fs = scan_fs >= 2e9

    # initialize shared worker pool for raytrace/analytical shape
    # time-delay calculations and for reconstruction
    worker_pool = mp.Pool(__CPU_COUNT - 1)

    for ii in [105, 0]:
        logger.info("Scan [%3d / %3d]..." % (ii + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = fd_data[ii, :, :]
        tar_md = metadata[ii]

        if ~np.isnan(tar_md["emp_ref_id"]):
            # for ii in [28]:
            # The output dir, where the reconstructions will be stored
            out_dir = os.path.join(
                __OUT_DIR, "msc_thesis/slices/antipeaks_id%d/" % ii
            )
            verify_path(out_dir)

            for ant_pos in range(np.size(fd_data, axis=2)):
                scan_rad = tar_md["ant_rad"] / 100

                # Correct for how the scan radius is measured (from a
                # point on the antenna stand, not at the SMA connection
                # point)
                scan_rad += 0.03618

                # Get the adipose-only reference data for this scan
                adi_fd = fd_data[expt_ids.index(tar_md["emp_ref_id"]), :, :]

                adi_cal_cropped = (tar_fd - adi_fd)[tar_fs, :]

                sinogram = iczt(
                    adi_cal_cropped,
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
                t_start_idx = (
                    max_peak + peaks_to_decide[peaks_to_decide < 0][-1]
                )
                t_end_idx = max_peak + peaks_to_decide[peaks_to_decide > 0][0]

                t_start = ts[t_start_idx] * 1e9
                t_end = ts[t_end_idx] * 1e9

                start_anti = signal_td[t_start_idx]
                end_anti = signal_td[t_end_idx]

                plt.figure(figsize=(1000 / 150, 800 / 150), dpi=150)
                plt.rc("font", family="Libertinus Serif")

                plt.tick_params(labelsize=14)

                plt.plot(ts * 1e9, signal_td, "k-", lw=1.2)

                plt.plot((t_start, t_end), (start_anti, end_anti), "rx")

                plt.axvline(t_start, color="gray", linestyle="--", lw=1)
                plt.axvline(t_end, color="gray", linestyle="--", lw=1)

                y_arrow = np.max(signal_td) / 2

                plt.annotate(
                    "",
                    xy=(t_start, y_arrow),
                    xytext=(t_end, y_arrow),
                    arrowprops=dict(arrowstyle="<->", lw=1.4),
                )

                plt.text(
                    t_start - 0.02,
                    y_arrow * 1.02,
                    "Skin peak",
                    ha="right",
                    va="bottom",
                    fontsize=16,
                )

                plt.xlabel("Time (ns)", fontsize=16)
                plt.ylabel("Intensity (a.u.)", fontsize=16)
                # plt.legend(fontsize=14)
                plt.grid(linewidth=0.8)
                plt.tight_layout()
                # plt.show()

                plt.savefig(
                    os.path.join(
                        out_dir, "slice_antipeaks_ant%d.png" % ant_pos
                    ),
                    dpi=150,
                    transparent=True,
                )

                plt.close()
