"""
Illia Prykhodko
University of Manitoba
July 10th, 2025
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
from umbms.beamform.das import fd_das
from umbms.beamform.propspeed import estimate_speed, get_breast_speed_freq
from umbms.beamform.time_delay import get_pix_ts_old
from umbms.beamform.utility import apply_ant_t_delay, get_fd_phase_factor
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


def measure_stats(
    stl_z,
    cs,
    x_cm,
    y_cm,
    adi_shell_type,
    id,
    tum_x,
    tum_y,
    ant_rad,
    roi_rad,
    expt_adi_out_dir,
):
    stl_vert_time_start = time.time()
    phantom_xs, phantom_ys = get_shell_xy_for_z(
        adi_shell_type.swapcase(),
        stl_z * 10,
        slice_thickness=1.0,
    )
    logger.info(f"STL vertices = {time.time() - stl_vert_time_start} s")

    phantom_xs /= 10
    phantom_ys /= 10
    phantom_xs += x_cm * 100
    phantom_ys += y_cm * 100

    phi_cs = np.linspace(0, 2 * np.pi, 100)
    rho_stl, phi_stl = cart_to_polar(phantom_xs, phantom_ys)
    cs_rho = cs(phi_stl) * 100

    pcc = pearsonr(cs_rho, rho_stl)[0]
    _ccc = ccc(cs_rho, rho_stl)

    phi_rho_plot_time_start = time.time()
    plt.rc("font", family="Libertinus Serif")
    plt.plot(phi_stl, rho_stl, "b--", label="STL file data")
    plt.plot(phi_cs, cs(phi_cs) * 100, "r-", label="Cubic spline")
    plt.xlabel("Angle of the antenna (rad)", fontsize=16)
    plt.ylabel("Polar radius of the skin response (cm)", fontsize=16)
    plt.grid(linewidth=0.7)
    plt.legend()
    plt.savefig(
        os.path.join(
            expt_adi_out_dir,
            "id_%d_rho-phi_diagram_%.2f_z_slice.png" % (id, stl_z),
        ),
        dpi=300,
    )
    plt.close()
    logger.info(f"Rho-phi plotting = {time.time() - phi_rho_plot_time_start} s")

    das_recon_plot_time = time.time()
    # # Plot the DAS reconstruction
    plot_fd_img(
        img=np.abs(das_adi_recon),
        cs=cs,
        tum_x=tum_x,
        tum_y=tum_y,
        tum_rad=tum_rad,
        ox=x_cm,
        oy=y_cm,
        ant_rad=ant_rad,
        roi_rad=roi_rad,
        img_rad=roi_rad,
        phantom_id=adi_shell_type,
        plot_stl=True,
        stl_z=stl_z,
        title=f"Shell {adi_shell_type}",
        save_fig=True,
        save_str=os.path.join(
            expt_adi_out_dir,
            "id_%d_adi_cal_das_%.2f_z_slice.png" % (id, stl_z),
        ),
        save_close=True,
    )
    logger.info(f"DAS recon plot = {time.time() - das_recon_plot_time} s")

    return pcc, _ccc


__CPU_COUNT = mp.cpu_count()

__DATA_DIR = os.path.join(get_proj_path(), "data/umbmid/g3/")

__OUT_DIR = os.path.join(get_proj_path(), "output/g3/")
verify_path(__OUT_DIR)
__FITTED_DATA_DIR = os.path.join(get_proj_path(), "data/freq_data/")

__FD_NAME = "fd_data_gen_three_s11.pickle"
__MD_NAME = "metadata_gen_three.pickle"
__FITTED_NAME = "Fitted Dielectric Measurements Glycerin.csv"

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

    # Read .csv file of fitted values of permittivity and conductivity
    df = pandas.read_csv(os.path.join(__FITTED_DATA_DIR, __FITTED_NAME))

    # Calculate velocity array
    freqs = np.array(df["Freqs"].values, dtype=float)
    permittivities = np.array(df["Permittivity"].values)
    conductivities = np.array(df["Conductivity"].values)
    velocities = get_breast_speed_freq(freqs, permittivities, conductivities)

    # The output dir, where the reconstructions will be stored
    out_dir = os.path.join(__OUT_DIR, "msc_thesis/stl_comparison/")
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
    # for ii in [0, 51, 99, 131]:
    # for ii in [131]:
    for ii in range(10):
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

            # correction for a phantom
            # mid_breast_max -= 0.00832
            # mid_breast_min -= 0.0071231

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

            # breast_speed = get_breast_speed(2)
            #
            # # Get the one-way propagation times for each pixel,
            # # for each antenna position
            # pix_ts, int_f_xs, int_f_ys, int_b_xs, int_b_ys = \
            #     get_pix_ts(ant_rad=ant_rad, m_size=__M_SIZE,
            #                roi_rad=roi_rad, air_speed=__VAC_SPEED,
            #                breast_speed=breast_speed, adi_rad=adi_rad,
            #                mid_breast_max=mid_breast_max,
            #                mid_breast_min=mid_breast_min)

            # Get the phase factor for efficient computation
            phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

            # Get the adipose-only reference data for this scan
            adi_fd = fd_data[expt_ids.index(tar_md["adi_ref_id"]), :, :]
            empt_fd = fd_data[expt_ids.index(tar_md["emp_ref_id"]), :, :]

            # Subtract reference and retain the frequencies above 2 GHz
            adi_cal_cropped = (tar_fd - empt_fd)[tar_fs, :]
            adi_cal_cropped_non_emp = (tar_fd - adi_fd)[tar_fs, :]

            # plt_sino(fd=adi_cal_cropped_non_emp, title='ID: %d' % (ii+1),
            #     save_str='id_%d_sino.png' % (ii+1), close=True,
            #          out_dir=out_dir, transparent=False)

            cs, x_cm, y_cm = get_boundary_iczt(
                adi_cal_cropped, ant_rad, ini_ant_ang=-136.0
            )

            # If the scan does include a tumour
            if ~np.isnan(tar_md["tum_diam"]):
                # Set a str for plotting
                plt_str = "%.1f cm tum in Class %d %s, ID: %d" % (
                    tar_md["tum_diam"],
                    tar_md["birads"],
                    tar_md["phant_id"],
                    tar_md["id"],
                )

            else:  # If the scan does NOT include a tumour
                plt_str = "Class %d %s, ID: %d" % (
                    tar_md["birads"],
                    tar_md["phant_id"],
                    tar_md["id"],
                )

            # Reconstruct a DAS image
            das_adi_recon = fd_das(
                fd_data=adi_cal_cropped_non_emp[::__SAMPLING, :],
                phase_fac=phase_fac,
                freqs=scan_fs[tar_fs][::__SAMPLING],
                worker_pool=worker_pool,
            )

            # Save that DAS reconstruction to a .pickle file
            save_pickle(
                das_adi_recon, os.path.join(expt_adi_out_dir, "das_adi.pickle")
            )

            # Plot the DAS reconstruction
            plot_fd_img(
                img=np.abs(das_adi_recon),
                cs=cs,
                tum_x=tum_x,
                tum_y=tum_y,
                tum_rad=tum_rad,
                # mid_breast_max=mid_breast_max,
                # mid_breast_min=mid_breast_min,
                ox=x_cm,
                oy=y_cm,
                ant_rad=ant_rad,
                roi_rad=roi_rad,
                img_rad=roi_rad,
                phantom_id=adi_shell_type,
                title="",
                save_fig=True,
                save_str=os.path.join(
                    expt_adi_out_dir, "id_%d_adi_cal_das.png" % tar_md["id"]
                ),
                save_close=True,
            )

            logger.info(f"ID: {ii + 1}, shell: {adi_shell_type}")

            best_pcc, best_ccc = measure_stats(
                stl_z=__BEST_SLICE[adi_shell_type],
                cs=cs,
                x_cm=x_cm,
                y_cm=y_cm,
                adi_shell_type=adi_shell_type,
                id=tar_md["id"],
                tum_x=tum_x,
                tum_y=tum_y,
                ant_rad=ant_rad,
                roi_rad=roi_rad,
                expt_adi_out_dir=expt_adi_out_dir,
            )

            stats_dict_best[adi_shell_type]["pcc"] = np.append(
                stats_dict_best[adi_shell_type]["pcc"], best_pcc
            )
            stats_dict_best[adi_shell_type]["ccc"] = np.append(
                stats_dict_best[adi_shell_type]["ccc"], best_ccc
            )

            real_pcc, real_ccc = measure_stats(
                stl_z=np.abs(tar_md["ant_z"]),
                cs=cs,
                x_cm=x_cm,
                y_cm=y_cm,
                adi_shell_type=adi_shell_type,
                id=tar_md["id"],
                tum_x=tum_x,
                tum_y=tum_y,
                ant_rad=ant_rad,
                roi_rad=roi_rad,
                expt_adi_out_dir=expt_adi_out_dir,
            )

            stats_dict_real[adi_shell_type]["pcc"] = np.append(
                stats_dict_real[adi_shell_type]["pcc"], real_pcc
            )
            stats_dict_real[adi_shell_type]["ccc"] = np.append(
                stats_dict_real[adi_shell_type]["ccc"], real_ccc
            )

    for shell in ["A2", "A3", "A14", "A16"]:
        print(
            f"Average PCC for shell {shell} at best slice = {np.average(stats_dict_best[shell]['pcc'])}"
        )
        print(
            f"Average CCC for shell {shell} at best slice = {np.average(stats_dict_best[shell]['ccc'])}"
        )

        print(
            f"Average PCC for shell {shell} at real slice = {np.average(stats_dict_real[shell]['pcc'])}"
        )
        print(
            f"Average CCC for shell {shell} at real slice = {np.average(stats_dict_real[shell]['ccc'])}"
        )
