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
from scipy.interpolate import interp1d

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle

from umbms.plot.imgplots import plot_fd_img
from umbms.plot.sinogramplot import plt_sino

from umbms.beamform.das import fd_das
from umbms.beamform.time_delay import get_pix_ts_old
from umbms.beamform.utility import apply_ant_t_delay, get_fd_phase_factor

from umbms.boundary.boundary_detection import (
    find_boundary,
    polar_fit_cs,
    cart_to_polar,
)
from umbms.beamform.iczt import iczt

from umbms.beamform.propspeed import estimate_speed, get_breast_speed_freq

###############################################################################

__CPU_COUNT = mp.cpu_count()

__DATA_DIR = os.path.join(get_proj_path(), "data/umbmid/cyl_phantom/")
__OUT_DIR = os.path.join(get_proj_path(), "output/")
verify_path(__OUT_DIR)
__FITTED_DATA_DIR = os.path.join(get_proj_path(), "data/freq_data/")

__FD_NAME = "cyl_phantom_diag_s11.pickle"
__MD_NAME = "metadata_cyl_phantom_diag.pickle"
__FITTED_NAME = "Fitted Dielectric Measurements Glycerin.csv"

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


def get_middle_td(pix_ts):
    size = np.size(pix_ts, axis=0)
    idx = np.size(pix_ts, axis=1) // 2 - 1
    output = np.zeros(size)
    for i in range(size):
        output[i] = pix_ts[i, idx, idx]
    return output


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

    # Read .csv file of fitted values of permittivity and conductivity
    df = pandas.read_csv(os.path.join(__FITTED_DATA_DIR, __FITTED_NAME))

    # Calculate velocity array
    freqs = np.array(df["Freqs"].values, dtype=float)
    permittivities = np.array(df["Permittivity"].values)
    conductivities = np.array(df["Conductivity"].values)
    # conductivities = np.zeros_like(conductivities)
    velocities = get_breast_speed_freq(freqs, permittivities, conductivities)

    # Determine fibroglandular percentage
    fibr_perc = 0

    # initialize shared worker pool for raytrace/analytical shape
    # time-delay calculations and for reconstruction
    worker_pool = mp.Pool(__CPU_COUNT - 1)

    # for ii in range(n_expts - 1):
    for ii in [28]:
        # The output dir, where the reconstructions will be stored
        out_dir = os.path.join(__OUT_DIR, "msc_thesis/slices/")
        verify_path(out_dir)

        logger.info("Scan [%3d / %3d]..." % (ii + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = fd_data[ii, :, :]
        tar_md = metadata[ii]

        if ~np.isnan(tar_md["emp_ref_id"]):
            expt_adi_out_dir = os.path.join(
                out_dir, "id-%d-adi-%.1f-fibr-perc/" % (ii, fibr_perc)
            )
            verify_path(expt_adi_out_dir)

            # Get metadata for plotting
            scan_rad = tar_md["ant_rad"] / 100
            tum_x = tar_md["tum_x"] / 100
            tum_y = tar_md["tum_y"] / 100
            tum_rad = 0.5 * (tar_md["tum_diam"] / 100)
            adi_rad = __PHANTOM_RAD

            # Correct for how the scan radius is measured (from a
            # point on the antenna stand, not at the SMA connection
            # point)
            scan_rad += 0.03618

            # Define the radius of the region of interest
            roi_rad = adi_rad + 0.01

            # Get the area of each pixel in the image domain
            dv = ((2 * roi_rad) ** 2) / (__M_SIZE**2)

            # Correct for the antenna time delay
            # NOTE: Only the new antenna was used in UM-BMID Gen-3
            ant_rad = apply_ant_t_delay(scan_rad=scan_rad, new_ant=True)

            speed = estimate_speed(
                adi_rad=adi_rad, ant_rad=scan_rad, new_ant=True
            )

            pix_ts = get_pix_ts_old(
                ant_rad=ant_rad, m_size=__M_SIZE, roi_rad=roi_rad, speed=speed
            )

            dist_per_pix = 2 * roi_rad / __M_SIZE
            time_per_pix = dist_per_pix / 3e8

            print(f"Time per pixel = {time_per_pix * 1e9} ns")
            print(f"Time per point in iczt = {np.diff(iczt_ts)[0] * 1e9} ns")

            # Get the phase factor for efficient computation
            phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

            # Get the adipose-only reference data for this scan
            adi_fd = fd_data[expt_ids.index(tar_md["emp_ref_id"]), :, :]

            adi_cal_cropped = (tar_fd - adi_fd)[tar_fs, :]

            # If the scan does include a tumour
            if ~np.isnan(tar_md["tum_diam"]):
                # Set a str for plotting
                plt_str = "%.1f cm rod in\nID: %d" % (tar_md["tum_diam"], ii)
            else:
                plt_str = "Empty cylinder\nID: %d" % ii

            das_adi_recon = fd_das(
                fd_data=adi_cal_cropped,
                phase_fac=phase_fac,
                freqs=scan_fs[tar_fs],
                worker_pool=worker_pool,
            )

            sinogram = iczt(
                adi_cal_cropped,
                ini_t=__INI_T,
                fin_t=__FIN_T,
                ini_f=__INI_F,
                fin_f=__FIN_F,
                n_time_pts=__N_TS,
            )

            plt_sino(
                adi_cal_cropped,
                title="",
                save_str="sinogram.png",
                out_dir=out_dir,
                slices=True,
            )

            # Save that DAS reconstruction to a .pickle file
            save_pickle(
                das_adi_recon, os.path.join(expt_adi_out_dir, "das_adi.pickle")
            )

            slice_das = np.abs(
                das_adi_recon[np.size(das_adi_recon, axis=0) // 2, :]
            )
            slice_sino = np.abs(sinogram[:, 0])

            # Normalization
            slice_das /= np.max(slice_das)
            slice_sino /= np.max(slice_sino)

            x_das = np.linspace(-roi_rad, roi_rad, __M_SIZE)
            x_sino_time = iczt_ts
            x_sino_dist = -(ant_rad - x_sino_time * 3e8 / 2)

            # take the time-domain data (now expressed in metres)
            # and ­interpolate it onto the distance-domain sample positions
            interp_fun = interp1d(
                x_sino_dist, slice_sino, kind="cubic", fill_value="extrapolate"
            )

            slice_sino_interp = interp_fun(x_das)

            x_v1 = -roi_rad  # left vertical line
            x_v2 = -0.0285  # right vertical line
            plt.figure(figsize=(1000 / 150, 800 / 150), dpi=150)
            plt.rc("font", family="Libertinus Serif")
            # Set the size of the axis tick labels
            plt.tick_params(labelsize=14)

            # - roi_rad, -0.0285
            plt.plot(x_das, slice_das, "r-", label="DAS", linewidth=1.2)
            plt.plot(
                x_das, slice_sino_interp, "k-", label="Sinogram", linewidth=1.2
            )

            y_arrow = 0.3

            plt.axvline(x_v1, color="gray", linestyle="--", lw=1)
            plt.axvline(x_v2, color="gray", linestyle="--", lw=1)

            plt.annotate(
                "",
                xy=(x_v1, y_arrow),
                xytext=(x_v2, y_arrow),
                arrowprops=dict(arrowstyle="<->", lw=1.4),
            )

            plt.text(
                (x_v1 + x_v2) / 2,
                y_arrow * 1.02,
                "Skin peak",
                ha="center",
                va="bottom",
                fontsize=16,
            )

            plt.xlabel("Distance (m)", fontsize=16)
            plt.ylabel("Intensity normalized (a.u.)", fontsize=16)
            plt.legend(fontsize=14)
            plt.grid(linewidth=0.8)
            plt.tight_layout()
            # plt.show()
            plt.savefig(
                os.path.join(out_dir, "slice_comparison_%d.png" % ii),
                dpi=150,
                transparent=True,
            )

            break
