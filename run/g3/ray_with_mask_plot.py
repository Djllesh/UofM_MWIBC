"""
Illia Prykhodko
University of Manitoba
July 11th, 2025
"""

import multiprocessing as mp
import os
import time

import numpy as np
import pandas
import scipy.constants
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from umbms import get_proj_path, get_script_logger, verify_path
from umbms.analysis.stats import ccc
import umbms.beamform.breastmodels as breastmodels
from umbms.beamform.das import fd_das
from umbms.beamform.propspeed import estimate_speed, get_breast_speed_freq
from umbms.beamform.time_delay import get_pix_ts_old
from umbms.beamform.utility import (
    apply_ant_t_delay,
    get_fd_phase_factor,
    get_xy_arrs,
    get_ant_scan_xys,
)
from umbms.boundary.boundary_detection import (
    cart_to_polar,
    find_boundary,
    get_binary_mask,
    get_boundary_iczt,
    polar_fit_cs,
)
from umbms.loadsave import load_pickle, save_pickle

###############################################################################
from umbms.plot.imgplots import plot_fd_img
from umbms.plot.stlplot import get_shell_xy_for_z

__CPU_COUNT = mp.cpu_count()

__DATA_DIR = os.path.join(get_proj_path(), "data/umbmid/g3/")

__OUT_DIR = os.path.join(get_proj_path(), "output/g3/")
verify_path(__OUT_DIR)
__FITTED_DATA_DIR = os.path.join(get_proj_path(), "data/freq_data/")

__FD_NAME = "fd_data_gen_three_s11.pickle"
__MD_NAME = "metadata_gen_three.pickle"

# The frequency parameters from the scan
__INI_F = 1e9
__FIN_F = 9e9
__N_FS = 1001
__SAMPLING = 12

# The time-domain conversion parameters
__INI_T = 0.5e-9
__FIN_T = 5.5e-9
__N_TS = 700

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

__BEST_SLICE = {"A2": 4.99, "A3": 2.27, "A14": 2.60, "A16": 3.70}

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

    n_expts = len(metadata)  # The number of individual scans

    # Get the unique ID of each experiment / scan
    expt_ids = [md["id"] for md in metadata]

    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

    # Only retain frequencies above 2 GHz due to antenna VSWR
    tar_fs = scan_fs >= 2e9

    # The output dir, where the reconstructions will be stored
    out_dir = os.path.join(__OUT_DIR, "msc_thesis/ray_plots/")
    verify_path(out_dir)

    stats_dict_best = {
        "A2": {"pcc": np.array([]), "ccc": np.array([])},
        "A3": {"pcc": np.array([]), "ccc": np.array([])},
        "A14": {"pcc": np.array([]), "ccc": np.array([])},
        "A16": {"pcc": np.array([]), "ccc": np.array([])},
    }

    stats_dict_real = {
        "A2": {"pcc": np.array([]), "ccc": np.array([])},
        "A3": {"pcc": np.array([]), "ccc": np.array([])},
        "A14": {"pcc": np.array([]), "ccc": np.array([])},
        "A16": {"pcc": np.array([]), "ccc": np.array([])},
    }

    # initialize shared worker pool for raytrace/analytical shape
    # time-delay calculations and for reconstruction
    worker_pool = mp.Pool(__CPU_COUNT - 1)

    # Determine fibroglandular percentage
    # fibr_perc = 2
    # for ii in range(n_expts):  # For each scan / experiment
    for ii in [0, 51, 99, 131]:
        # for ii in [131]:
        logger.info("Scan [%3d / %3d]..." % (ii + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = fd_data[ii, :, :]
        tar_md = metadata[ii]

        # If the scan had a fibroglandular shell (indicating it was of
        # a complete tumour-containing or healthy phantom)
        if "F" in tar_md["phant_id"]:
            # Create the output directory for the adipose-only
            # reference reconstructions
            expt_adi_out_dir = os.path.join(
                out_dir, "id-%d-adi/" % (tar_md["id"])
            )
            verify_path(expt_adi_out_dir)

            # Get metadata for plotting
            scan_rad = tar_md["ant_rad"] / 100
            tum_x = tar_md["tum_x"] / 100
            tum_y = tar_md["tum_y"] / 100
            tum_rad = 0.5 * (tar_md["tum_diam"] / 100)
            adi_rad = __ADI_RADS[tar_md["phant_id"].split("F")[0]]
            mid_breast_max, mid_breast_min = (
                __MID_BREAST_RADS[tar_md["phant_id"].split("F")[0]][0],
                __MID_BREAST_RADS[tar_md["phant_id"].split("F")[0]][1],
            )
            adi_shell_type = tar_md["phant_id"].split("F")[0]
            logger.info(f"Shell type {adi_shell_type}")

            # Correct for how the scan radius is measured (from a
            # point on the antenna stand, not at the SMA connection
            # point)
            scan_rad += 0.03618

            # Define the radius of the region of interest
            roi_rad = adi_rad + 0.0135

            # Get the area of each pixel in the image domain
            dv = ((2 * roi_rad) ** 2) / (__M_SIZE**2)

            # Correct for the antenna time delay
            # NOTE: Only the new antenna was used in UM-BMID Gen-3
            ant_rad = apply_ant_t_delay(scan_rad=scan_rad, new_ant=True)

            # Estimate the propagation speed in the imaging domain
            speed = estimate_speed(
                adi_rad=adi_rad, ant_rad=scan_rad, new_ant=True
            )
            pix_ts = get_pix_ts_old(
                ant_rad=ant_rad, m_size=__M_SIZE, roi_rad=roi_rad, speed=speed
            )

            # Get the phase factor for efficient computation
            phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

            # Get the adipose-only reference data for this scan
            adi_fd = fd_data[expt_ids.index(tar_md["adi_ref_id"]), :, :]
            empt_fd = fd_data[expt_ids.index(tar_md["emp_ref_id"]), :, :]

            # Subtract reference and retain the frequencies above 2 GHz
            adi_cal_cropped = (tar_fd - empt_fd)[tar_fs, :]
            adi_cal_cropped_non_emp = (tar_fd - adi_fd)[tar_fs, :]

            cs, x_cm, y_cm = get_boundary_iczt(
                adi_cal_cropped, ant_rad, ini_ant_ang=-136.0
            )
            # Reconstruct a DAS image
            das_adi_recon = fd_das(
                fd_data=adi_cal_cropped_non_emp[::__SAMPLING, :],
                phase_fac=phase_fac,
                freqs=scan_fs[tar_fs][::__SAMPLING],
                worker_pool=worker_pool,
            )

            ant_pos_xs, ant_pos_ys = get_ant_scan_xys(
                ant_rad=ant_rad * 100, n_ant_pos=72
            )

            ant_pos_x, ant_pos_y = ant_pos_xs[0], ant_pos_ys[0]
            pix_x, pix_y = 5.5, 5.5

            img = np.abs(das_adi_recon)
            cs = cs
            tum_x = tum_x
            tum_y = tum_y
            tum_rad = tum_rad
            ox = x_cm
            oy = y_cm
            ant_rad = ant_rad
            roi_rad = roi_rad
            img_rad = roi_rad
            phantom_id = adi_shell_type
            title = ""
            save_fig = True
            save_str = os.path.join(
                expt_adi_out_dir, "id_%d_ray.png" % tar_md["id"]
            )
            save_close = True
            cbar_fmt = "%.1f"

            pix_xs, pix_ys = get_xy_arrs(np.size(img, axis=0), roi_rad)
            pix_xs *= 100
            pix_ys *= 100
            img_to_plt = np.abs(img)
            img_to_plt = img * np.ones_like(img)
            # img_to_rot = img_to_plt
            # ant_rad *= 100  # Convert from m to cm to facilitate plot
            img_rad *= 100
            # adi_rad *= 100
            roi_rad *= 100

            roi = breastmodels.get_roi(
                roi_rad, np.size(img_to_plt, axis=0), img_rad
            )

            # Set the pixels outside of the antenna trajectory to nan
            img_to_plt[np.logical_not(roi)] = np.nan

            # Define angles for plot the tissue geometry
            draw_angs = np.linspace(0, 2 * np.pi, 1000)

            # Define the x/y coordinates of the approximate tumor outline
            tum_xs, tum_ys = (
                tum_rad * 100 * np.cos(draw_angs) + tum_x * 100,
                tum_rad * 100 * np.sin(draw_angs) + tum_y * 100,
            )

            img_extent = [-img_rad, img_rad, -img_rad, img_rad]

            plt.rc("font", family="Libertinus Serif")
            plt.figure()  # Make the figure window

            plt.imshow(
                img_to_plt, cmap="inferno", extent=img_extent, aspect="equal"
            )

            # Set the size of the axis tick labels
            plt.tick_params(labelsize=14)

            # Set the x/y-ticks at multiples of 5 cm
            plt.gca().set_xticks([-6, -4, -2, 0, 2, 4, 6])
            plt.gca().set_yticks([-6, -4, -2, 0, 2, 4, 6])

            # Specify the colorbar tick format and size
            plt.colorbar(format=cbar_fmt).ax.tick_params(labelsize=14)
            plt.plot((ant_pos_x, pix_x), (ant_pos_y, pix_y), "k-")
            plt.plot(pix_x, pix_y, "bX")

            # Set the x/y axes limits
            plt.xlim([-roi_rad, roi_rad])
            plt.ylim([-roi_rad, roi_rad])

            # Plot the approximate tumor boundary
            plt.plot(
                tum_xs, tum_ys, "g", label="Observed position", linewidth=1.5
            )

            plt.xlabel("x-axis (cm)", fontsize=16)  # Make the x-axis label
            plt.ylabel("y-axis (cm)", fontsize=16)  # Make the y-axis label

            phi = np.deg2rad(np.arange(0, 360, 0.1))
            rho = cs(phi)
            xs = rho * np.cos(phi) * 100
            ys = rho * np.sin(phi) * 100
            plt.plot(xs, ys, "r-", label="Cubic spline")
            # plt.show()
            plt.tight_layout()

            plt.savefig(
                save_str, transparent=False, dpi=300, bbox_inches="tight"
            )
            roi_rad /= 100
            plt.close()

            plt.rc("font", family="Libertinus Serif")
            plt.figure()  # Make the figure window

            mask = get_binary_mask(cs, m_size=__M_SIZE * 5, roi_rad=roi_rad)
            img = np.zeros((__M_SIZE * 5, __M_SIZE * 5))

            img[mask] = 0.5

            img_extent = [-roi_rad, roi_rad, -roi_rad, roi_rad]

            plt.imshow(img, extent=img_extent, cmap="viridis")
            plt.plot(
                (ant_pos_x / 100, pix_x / 100),
                (ant_pos_y / 100, pix_y / 100),
                "k-",
            )
            plt.plot(pix_x / 100, pix_y / 100, "bX")
            plt.xlim([-roi_rad, roi_rad])
            plt.ylim([-roi_rad, roi_rad])

            plt.xlabel("x-axis (cm)", fontsize=16)
            plt.ylabel("y-axis (cm)", fontsize=16)
            plt.tick_params(labelsize=14)
            plt.tight_layout()
            # plt.show()
            save_str = os.path.join(
                expt_adi_out_dir, "id_%d_simplified.png" % tar_md["id"]
            )
            plt.savefig(
                save_str, transparent=False, dpi=300, bbox_inches="tight"
            )
            plt.close()

            plt.rc("font", family="Libertinus Serif")
            plt.figure()  # Make the figure window

            ray = np.zeros(200)

            ray[40:180] = 1

            plt.plot(ray, "k-", linewidth=1.2)
            plt.xlabel("m index", fontsize=16)
            plt.ylabel(r"The value of $\mathcal{M}$", fontsize=16)
            plt.tick_params(labelsize=14)
            plt.grid(linewidth=0.7)
            plt.gca().set_yticks([1, 0])
            plt.tight_layout()
            # plt.show()
            save_str = os.path.join(
                expt_adi_out_dir, "id_%d_ray_unraveled.png" % tar_md["id"]
            )
            plt.savefig(
                save_str, transparent=False, dpi=300, bbox_inches="tight"
            )
