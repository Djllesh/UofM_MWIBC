"""
Illia Prykhodko
University of Manitoba
July 14th, 2025
"""

import matplotlib.pyplot as plt
import os

import numpy as np
import scipy.constants

from scipy.signal import find_peaks, correlate, correlation_lags
from umbms import get_proj_path, get_script_logger, verify_path
from umbms.beamform.das import fd_das
from umbms.beamform.propspeed import (
    estimate_speed,
)
from umbms.beamform.time_delay import (
    get_pix_ts_old,
)
from umbms.beamform.utility import (
    apply_ant_t_delay,
    get_fd_phase_factor,
)

from umbms.boundary.boundary_detection import (
    get_boundary_iczt,
    prepare_fd_data,
    polar_fit_cs,
)

from umbms.loadsave import load_pickle, save_pickle
from umbms.plot.imgplots import plot_fd_img
from umbms.plot.sinogramplot import plt_sino, show_sinogram
from umbms.beamform.iczt import iczt

__DATA_DIR = os.path.join(get_proj_path(), "data/umbmid/g3/")

__OUT_DIR = os.path.join(get_proj_path(), "output/g3/msc_thesis/")
verify_path(__OUT_DIR)

__FD_NAME = "fd_data_gen_three_s11.pickle"
__MD_NAME = "metadata_gen_three.pickle"

# The frequency parameters from the scan
__INI_F = 1e9
__FIN_F = 9e9
__N_FS = 1001

# The time-domain conversion parameters
__INI_T = 0.5e-9
__FIN_T = 5.5e-9
__N_TS = 700

# The size of the reconstructed image along one dimension
__M_SIZE = 150

__SAMPLING = 12

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
__SPHERE_RAD = 0.0075

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


if __name__ == "__main__":
    logger = get_script_logger(__file__)

    # Load the frequency domain data and metadata
    fd_data, metadata = load_data()

    # Get the unique ID of each experiment / scan
    expt_ids = [md["id"] for md in metadata]

    ii = 205

    # Get the frequency domain data and metadata of this experiment
    tar_fd = fd_data[ii, :, :]
    tar_md = metadata[ii]
    scan_rad = tar_md["ant_rad"] / 100

    scan_rad += 0.03618
    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

    ant_rad = apply_ant_t_delay(scan_rad=scan_rad, new_ant=True)

    # Only retain frequencies above 2 GHz due to antenna VSWR
    tar_fs = scan_fs >= 2e9

    adi_fd = fd_data[expt_ids.index(tar_md["adi_ref_id"]), :, :]

    fd_emp = fd_data[expt_ids.index(tar_md["emp_ref_id"]), :, :]

    # Subtract reference and retain the frequencies above 2 GHz
    adi_cal_cropped = (tar_fd - adi_fd)[tar_fs, :]

    # Subtract the empty reference for the boundary detection
    emp_cal_cropped = (tar_fd - fd_emp)[tar_fs, :]

    _, ts, kernel_not_aligned = prepare_fd_data(
        adi_emp_cropped=emp_cal_cropped,
        ini_t=__INI_T,
        fin_t=__FIN_T,
        n_time_pts=__N_TS,
        ini_f=__INI_F,
        fin_f=__FIN_F,
        ant_rad=ant_rad,
        time_algned=False,
    )

    _, _, kernel_aligned = prepare_fd_data(
        adi_emp_cropped=emp_cal_cropped,
        ini_t=__INI_T,
        fin_t=__FIN_T,
        n_time_pts=__N_TS,
        ini_f=__INI_F,
        fin_f=__FIN_F,
        ant_rad=ant_rad,
        time_algned=True,
    )

    plt.figure()
    plt.rc("font", family="Libertinus Serif")
    plt.plot(
        ts * 1e9,
        kernel_aligned,
        "r-",
        linewidth=1.3,
        label="Time-aligned kernel",
    )
    plt.plot(
        ts * 1e9, kernel_not_aligned, "k--", linewidth=1.0, label="Mean signal"
    )
    plt.xlabel("Time (ns)", fontsize=16)
    plt.ylabel("Intensity (a.u.)", fontsize=16)
    plt.tick_params(labelsize=14)
    plt.grid(linewidth=0.7)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.expanduser("~"), "Desktop/aligned_vs_mean_kernel.png"
        ),
        dpi=300,
    )
