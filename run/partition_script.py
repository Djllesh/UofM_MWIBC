"""
Illia Prykhodko
University of Manitoba
April 14th, 2025
"""

import os
import numpy as np
import pandas
import scipy.constants
from time import perf_counter
import multiprocessing as mp

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle

from umbms.beamform.das import fd_das, fd_das_freq_dep
from umbms.hardware.antenna import to_phase_center
from umbms.beamform.time_delay import get_pix_ts_old, get_pix_ts
from umbms.beamform.utility import apply_ant_t_delay, get_fd_phase_factor

from umbms.boundary.boundary_detection import get_binary_mask, get_boundary_iczt

from umbms.beamform.propspeed import estimate_speed, get_breast_speed_freq
from umbms.boundary.raytrace import find_boundary_rt

###############################################################################

__CPU_COUNT = mp.cpu_count()
__PRECISION_SCALING_FACTOR = 1

assert isinstance(__PRECISION_SCALING_FACTOR, int), (
    f"Scaling factor is not int, got: {__PRECISION_SCALING_FACTOR}"
)

__DATA_DIR = os.path.join(get_proj_path(), "data/umbmid/cyl_phantom/")
__OUT_DIR = os.path.join(get_proj_path(), "output/cyl_phantom/")
verify_path(__OUT_DIR)
__FITTED_DATA_DIR = os.path.join(get_proj_path(), "data/freq_data/")

__FD_NAME = "cyl_phantom_immediate_reference_s11_rescan.pickle"
__MD_NAME = "metadata_cyl_phantom_immediate_reference_rescan.pickle"
__FITTED_NAME = "Dielectric Measurements Glycerin.csv"

# The frequency parameters from the scan
__INI_F = 2e9
__FIN_F = 9e9
__N_FS = 1001

# The time-domain conversion parameters
__INI_T = 0.5e-9
__FIN_T = 5.5e-9
__N_TS = 700

# The size of the reconstructed image along one dimension
__M_SIZE = 150
# Number of antenna positions
__N_ANT_POS = 72

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

    rhos = []
    for tum_x, tum_y in zip(
        [md["tum_x"] / 100 for md in metadata],
        [md["tum_y"] / 100 for md in metadata],
    ):
        if not np.isnan(tum_x):
            rho = np.sqrt(tum_x**2 + tum_y**2)
            rhos.append(rho)

    rhos = np.unique(rhos)

    das_regular = [[] for i in range(6)]
    das_part_binary = [[] for i in range(6)]
    das_freq_dep_zero_cond = [[] for i in range(6)]
    das_freq_dep = [[] for i in range(6)]
    das_rt = [[] for i in range(6)]

    n_expts = np.size(fd_data, axis=0)  # The number of individual scans

    # Get the unique ID of each experiment / scan
    expt_ids = [md["id"] for md in metadata]

    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

    # tar_fs = (scan_fs >= 6e9) | (scan_fs <= 5e9)

    # Only retain frequencies above 2 GHz due to antenna VSWR
    tar_fs = scan_fs >= 2e9

    # Read .csv file of fitted values of permittivity and conductivity
    df = pandas.read_csv(os.path.join(__FITTED_DATA_DIR, __FITTED_NAME))

    # Calculate velocity array
    freqs = np.array(df["Freqs"].values, dtype=float)
    permittivities = np.array(df["Permittivity"].values)
    conductivities = np.array(df["Conductivity"].values)
    zero_conductivities = np.zeros_like(conductivities)
    velocities_zero_cond = get_breast_speed_freq(
        freqs, permittivities, zero_conductivities
    )
    velocities = get_breast_speed_freq(freqs, permittivities, conductivities)

    # Determine fibroglandular percentage
    fibr_perc = 0
    mse_array = np.array([])

    # initialize shared worker pool for raytrace/analytical shape
    # time-delay calculations and for reconstruction
    worker_pool = mp.Pool(__CPU_COUNT - 1)

    intersection_times = np.array([], dtype=float)
    recon_times = np.array([], dtype=float)
    full_times = np.array([], dtype=float)

    # The output dir, where the reconstructions will be stored
    out_dir = os.path.join(__OUT_DIR, "recons/Immediate reference/Gen 2/")
    verify_path(out_dir)

    for ii in range(n_expts):
        # Initialize the indices
        idx_to_work = np.arange(__N_ANT_POS)

        # WARN: This is the manual part. You have to make a choice,
        # where to partition your antennas, if you are even interested
        # in this functionality. You essentially select the antenna
        # positions that are going to be taken into account in the
        # reconstruction

        # Antenna position IDs for recon at specific positions
        # ID 4: [22, 47)
        # condition = np.logical_and(idx_to_work >= 22, idx_to_work < 47)

        # ID 8: [17, 38)
        # condition = np.logical_and(idx_to_work >= 17, idx_to_work < 38)

        # ID 10: [10, 44)
        # condition = np.logical_and(idx_to_work >= 10, idx_to_work < 44)

        # ID 12: [31, 58)
        # condition = np.logical_and(idx_to_work >= 31, idx_to_work < 58)

        # ID 19: [0, 8) U (58, 71]
        # condition = np.logical_or(idx_to_work < 8, idx_to_work > 58)

        # ID 21: [0, 11) U (56, 71]
        # condition = np.logical_or(idx_to_work < 11, idx_to_work > 56)

        # ID 25: [49, 71]
        # condition = np.logical_and(idx_to_work >= 49, idx_to_work <= 71)

        # ID 29: [0, 20)
        # condition = np.logical_and(idx_to_work >= 0, idx_to_work < 20)

        # ID 34: [5, 34)
        # condition = np.logical_and(idx_to_work >= 5, idx_to_work < 34)

        # ID 38: [39, 66)
        # condition = np.logical_and(idx_to_work >= 39, idx_to_work < 66)

        # ID 42: [0, 21) U (66, 71]
        # condition = np.logical_or(idx_to_work < 21, idx_to_work > 66)

        # ID 44: [28, 59)
        # condition = np.logical_and(idx_to_work >= 28, idx_to_work < 59)

        # ID 46: [11, 41)
        # condition = np.logical_and(idx_to_work >= 11, idx_to_work < 41)

        # ID 48: [47, 71]
        # condition = np.logical_and(idx_to_work >= 47, idx_to_work <= 71)

        partial_ant_idx = np.logical_and(idx_to_work >= 22, idx_to_work < 47)

        # For inversion, uncomment if you want to use the antennas from
        # "behind"
        # partial_ant_idx = ~partial_ant_idx

        # For a full reconstruction - all antenna idxs are True
        # partial_ant_idx = np.ones([__N_ANT_POS], dtype=bool)

        logger.info("Scan [%3d / %3d]..." % (ii + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = fd_data[ii, :, :]
        tar_md = metadata[ii]

        if ~np.isnan(tar_md["emp_ref_id"]) and ~np.isnan(tar_md["adi_ref_id2"]):
            # If the scan does include a tumour
            if ~np.isnan(tar_md["tum_diam"]):
                # Set a str for plotting
                plt_str = "%.1f cm rod in\nID: %d" % (tar_md["tum_diam"], ii)
            else:
                plt_str = "Empty cylinder\nID: %d" % ii

            full_time_start = perf_counter()
            expt_adi_out_dir = os.path.join(out_dir, "id-%d/" % ii)
            verify_path(expt_adi_out_dir)

            # Get metadata for plotting
            scan_rad = tar_md["ant_rad"] / 100
            tum_x = tar_md["tum_x"] / 100
            tum_y = tar_md["tum_y"] / 100
            tum_rad = 0.5 * (tar_md["tum_diam"] / 100)
            adi_rad = __PHANTOM_RAD
            rho = np.sqrt(tum_x**2 + tum_y**2)
            rho_idx = np.searchsorted(rhos, rho)

            # Correct for how the scan radius is measured (from a
            # point on the antenna stand, not at the SMA connection
            # point)
            # scan_rad += 0.03618

            # Define the radius of the region of interest
            roi_rad = adi_rad + 0.01

            # Get the area of each pixel in the image domain
            dv = ((2 * roi_rad) ** 2) / (__M_SIZE**2)

            # Correct for the antenna time delay
            ant_rad = to_phase_center(meas_rho=scan_rad)

            # Get the adipose-only and empty reference data
            # for this scan
            adi_fd_emp = fd_data[expt_ids.index(tar_md["emp_ref_id"]), :, :]
            adi_fd = fd_data[expt_ids.index(tar_md["adi_ref_id2"]), :, :]

            adi_cal_cropped_emp = tar_fd - adi_fd_emp
            adi_cal_cropped = (tar_fd - adi_fd)[tar_fs, :]

            # plt_sino(fd=adi_cal_cropped_emp, title='ID: %d' % (ii+1),
            #           save_str='id_%d_sino.png' % (ii+1), close=True,
            #           out_dir=out_dir, transparent=False)

            cs, x_cm, y_cm = get_boundary_iczt(adi_cal_cropped_emp, ant_rad)

            mask = get_binary_mask(
                cs,
                m_size=__M_SIZE,
                roi_rad=roi_rad,
                precision_scaling_factor=__PRECISION_SCALING_FACTOR,
            )

            # breast_speed = get_breast_speed(fibr_perc)
            # # # breast_speed = get_speed_from_perm(perm)

            ###########################################################
            # 5 DIFFERENT RECONSTRUCTIONS
            ###########################################################

            # ============== 1. Homogeneous DAS (regular) =============

            speed = estimate_speed(adi_rad=adi_rad, ant_rad=__ANT_RAD)

            pix_ts = get_pix_ts_old(
                ant_rad=ant_rad, m_size=__M_SIZE, roi_rad=roi_rad, speed=speed
            )

            phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

            das_regular_recon = fd_das(
                fd_data=adi_cal_cropped,
                phase_fac=phase_fac,
                freqs=scan_fs[tar_fs],
                worker_pool=worker_pool,
                partial_ant_idx=partial_ant_idx,
            )

            save_pickle(
                das_regular_recon,
                path=os.path.join(out_dir, "regular-das/id%d.pickle" % ii),
            )

            # plot_fd_img(img=np.abs(das_regular_recon), tum_x=tum_x,
            #             tum_y=tum_y, tum_rad=tum_rad, adi_rad=adi_rad, ox=x_cm,
            #             oy=y_cm, ant_rad=ant_rad, roi_rad=roi_rad,
            #             img_rad=roi_rad, title=plt_str, save_fig=True,
            #             save_str=os.path.join(out_dir,
            #                                   'id_%d_das_regular.png'
            #                                   % ii), save_close=True)

            # ========== 2. Binary DAS (domain partitioning) ==========
            breast_speed = np.average(velocities)
            pix_ts, int_f_xs, int_f_ys, int_b_xs, int_b_ys = get_pix_ts(
                ant_rad=ant_rad,
                m_size=__M_SIZE,
                roi_rad=roi_rad,
                air_speed=__VAC_SPEED,
                breast_speed=breast_speed,
                adi_rad=adi_rad,
                ox=x_cm,
                oy=y_cm,
                partial_ant_idx=partial_ant_idx,
                worker_pool=worker_pool,
            )
            phase_fac = get_fd_phase_factor(pix_ts=pix_ts)
            das_binary_recon = fd_das(
                fd_data=adi_cal_cropped,
                phase_fac=phase_fac,
                freqs=scan_fs[tar_fs],
                partial_ant_idx=partial_ant_idx,
                worker_pool=worker_pool,
            )

            save_pickle(
                das_binary_recon,
                path=os.path.join(out_dir, "binary-das/id%d.pickle" % ii),
            )

            # plot_fd_img(img=np.abs(das_binary_recon), tum_x=tum_x,
            #             tum_y=tum_y, tum_rad=tum_rad, adi_rad=adi_rad, ox=x_cm,
            #             oy=y_cm, ant_rad=ant_rad, roi_rad=roi_rad,
            #             img_rad=roi_rad, title=plt_str, save_fig=True,
            #             save_str=os.path.join(out_dir,
            #                                   'id_%d_das_binary.png'
            #                                   % ii), save_close=True)

            # 3. ======== Frequency-dependent DAS (zero cond) =========
            das_freq_dep_zero_cond_recon = fd_das_freq_dep(
                fd_data=adi_cal_cropped,
                int_f_xs=int_f_xs,
                int_f_ys=int_f_ys,
                int_b_xs=int_b_xs,
                int_b_ys=int_b_ys,
                velocities=velocities_zero_cond,
                ant_rad=ant_rad,
                freqs=scan_fs[tar_fs],
                adi_rad=adi_rad,
                m_size=__M_SIZE,
                roi_rad=roi_rad,
                air_speed=__VAC_SPEED,
                partial_ant_idx=partial_ant_idx,
                worker_pool=worker_pool,
            )

            save_pickle(
                das_freq_dep_zero_cond_recon,
                path=os.path.join(
                    out_dir, "freq_dep_non_cond-das/id%d.pickle" % ii
                ),
            )

            # plot_fd_img(img=np.abs(das_freq_dep_zero_cond_recon), tum_x=tum_x,
            #             tum_y=tum_y, tum_rad=tum_rad, adi_rad=adi_rad, ox=x_cm,
            #             oy=y_cm, ant_rad=ant_rad, roi_rad=roi_rad,
            #             img_rad=roi_rad, title=plt_str, save_fig=True,
            #             save_str=os.path.join(out_dir,
            #                                   'id_%d_das'
            #                                   '_freq_dep_zero_cond.png'
            #                                   % ii), save_close=True)

            # 4. ================= Frequency-dependent DAS ============
            das_freq_dep_recon = fd_das_freq_dep(
                fd_data=adi_cal_cropped,
                int_f_xs=int_f_xs,
                int_f_ys=int_f_ys,
                int_b_xs=int_b_xs,
                int_b_ys=int_b_ys,
                velocities=velocities,
                ant_rad=ant_rad,
                freqs=scan_fs[tar_fs],
                adi_rad=adi_rad,
                m_size=__M_SIZE,
                roi_rad=roi_rad,
                air_speed=__VAC_SPEED,
                partial_ant_idx=partial_ant_idx,
                worker_pool=worker_pool,
            )
            save_pickle(
                das_freq_dep_recon,
                path=os.path.join(out_dir, "freq_dep-das/id%d.pickle" % ii),
            )

            # plot_fd_img(img=np.abs(das_freq_dep_recon), tum_x=tum_x,
            #             tum_y=tum_y, tum_rad=tum_rad, adi_rad=adi_rad, ox=x_cm,
            #             oy=y_cm, ant_rad=ant_rad, roi_rad=roi_rad,
            #             img_rad=roi_rad, title=plt_str, save_fig=True,
            #             save_str=os.path.join(out_dir,
            #                                   'id_%d_das_freq_dep.png'
            #                                   % ii), save_close=True)

            # 5. ==================== Ray-tracing =====================
            int_f_xs, int_f_ys, int_b_xs, int_b_ys = find_boundary_rt(
                mask, ant_rad, roi_rad, worker_pool=worker_pool
            )
            das_rt_recon = fd_das_freq_dep(
                fd_data=adi_cal_cropped,
                int_f_xs=int_f_xs,
                int_f_ys=int_f_ys,
                int_b_xs=int_b_xs,
                int_b_ys=int_b_ys,
                velocities=velocities,
                ant_rad=ant_rad,
                freqs=scan_fs[tar_fs],
                adi_rad=adi_rad,
                m_size=__M_SIZE,
                roi_rad=roi_rad,
                air_speed=__VAC_SPEED,
                worker_pool=worker_pool,
            )

            save_pickle(
                das_rt_recon,
                path=os.path.join(out_dir, "rt-das/id%d.pickle" % ii),
            )

            # plot_fd_img(img=np.abs(das_rt_recon), cs=cs, tum_x=tum_x,
            #             tum_y=tum_y, tum_rad=tum_rad, adi_rad=adi_rad, ox=x_cm,
            #             oy=y_cm, ant_rad=ant_rad, roi_rad=roi_rad,
            #             img_rad=roi_rad, title=plt_str, save_fig=True,
            #             save_str=os.path.join(out_dir,
            #                                   'id_%d_das_rt.png'
            #                                   % ii), save_close=True)

            ############################################################

    worker_pool.close()
    worker_pool.join()
