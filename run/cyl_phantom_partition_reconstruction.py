"""
Illia Prykhodko
University of Manitoba
October 11th, 2022
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
from umbms.beamform.time_delay import get_pix_ts_old
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
        # for ii in [18]:

        # Antenna position IDs for recon at specific positions
        # ID 4: [22, 47)
        # ID 8: [17, 38)
        # ID 10: [10, 44)
        # ID 12: [31, 58)
        # ID 19: [0, 8) U (58, 71]
        # ID 21: [0, 11) U (56, 71]
        # ID 25: [49, 71]
        # ID 29: [0, 20)
        # ID 34: [5, 34)
        # ID 38: [39, 66)
        # ID 42: [0, 21) U (66, 71]
        # ID 44: [28, 59)
        # ID 46: [11, 41)
        # ID 48: [47, 71]

        # idx_to_work = np.arange(__N_ANT_POS)
        # partial_ant_idx = np.logical_and(idx_to_work >= 22,
        #                                  idx_to_work < 47)

        # partial_ant_idx = ~partial_ant_idx

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
            scan_rad += 0.03618

            # Define the radius of the region of interest
            roi_rad = adi_rad + 0.01

            # Get the area of each pixel in the image domain
            dv = ((2 * roi_rad) ** 2) / (__M_SIZE**2)

            # Correct for the antenna time delay
            # NOTE: Only the new antenna was used in UM-BMID Gen-3
            ant_rad = apply_ant_t_delay(scan_rad=scan_rad, new_ant=True)

            # speed = estimate_speed(adi_rad=adi_rad, ant_rad=scan_rad,
            #                        new_ant=True)
            # pix_ts = get_pix_ts_old(ant_rad=ant_rad, m_size=__M_SIZE,
            #                         roi_rad=roi_rad, speed=speed)

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

            # tup = (cs, x_cm, y_cm)
            # save_pickle(tup, os.path.join(out_dir, 'spln_pars/'
            #                                        'id%d_pars.pickle' % ii))

            # start = perf_counter()
            #
            # end = perf_counter()
            # intersection_times = np.append(intersection_times, (end - start))
            # logger.info('Parallel pix_ts (raytrace): %f s' % (end - start))

            # angles = np.linspace(0, 2 * np.pi, 1000)
            # x_circ = adi_rad * np.cos(angles) + x_cm
            # y_circ = adi_rad * np.sin(angles) + y_cm
            #
            # rho, _ = cart_to_polar(x_circ, y_circ)
            # rho_cs = cs(angles) * 100
            #
            # mse = np.sum((rho_cs - rho * 100)**2) / np.size(angles)
            # mse_array = np.append(mse_array, mse)
            #
            # print('MSE = %f' %mse)
            #
            # plt.plot(angles, rho, 'b--', label='Circle')
            # plt.plot( angles, rho_cs, 'r-', label='Cubic spline')
            # plt.legend()
            # plt.savefig(os.path.join(out_dir, 'rho-phi_plot_id_%d.png' % ii))
            # plt.close()

            # breast_speed = get_breast_speed(fibr_perc)
            # # # breast_speed = get_speed_from_perm(perm)

            ############################################################
            # 5 DIFFERENT RECONSTRUCTIONS

            # 1. Homogeneous DAS (regular)
            #
            # logger.info('Rho = %.3f, rho_idx = %d' % (rho, rho_idx))
            #
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
            )
            #
            # save_pickle(das_regular_recon,
            #             path=os.path.join(out_dir,
            #                               'regular-das/id%d.pickle' % ii))
            #
            # # das_regular_intensity = np.max(np.abs(das_regular_recon))
            # # das_regular[rho_idx].append(das_regular_intensity)
            # #
            # # logger.info('Regular DAS. Intensity = %.3f.'
            # #             ' Size of the intensity sub-array = %d'
            # #             % (das_regular_intensity,
            # #                len(das_regular[rho_idx])))
            # #
            # # plot_fd_img(img=np.abs(das_regular_recon), tum_x=tum_x,
            # #             tum_y=tum_y, tum_rad=tum_rad, adi_rad=adi_rad, ox=x_cm,
            # #             oy=y_cm, ant_rad=ant_rad, roi_rad=roi_rad,
            # #             img_rad=roi_rad, title=plt_str, save_fig=True,
            # #             save_str=os.path.join(out_dir,
            # #                                   'id_%d_das_regular.png'
            # #                                   % ii), save_close=True)
            #
            # # 2. Binary DAS (domain partitioning)
            # breast_speed = np.average(velocities)
            # pix_ts, int_f_xs, int_f_ys, int_b_xs, int_b_ys = \
            #     get_pix_ts(ant_rad=ant_rad, m_size=__M_SIZE,
            #                roi_rad=roi_rad, air_speed=__VAC_SPEED,
            #                breast_speed=breast_speed, adi_rad=adi_rad,
            #                ox=x_cm, oy=y_cm, worker_pool=worker_pool)
            # phase_fac = get_fd_phase_factor(pix_ts=pix_ts)
            # das_binary_recon = fd_das(fd_data=adi_cal_cropped,
            #                           phase_fac=phase_fac,
            #                           freqs=scan_fs[tar_fs],
            #                           worker_pool=worker_pool)
            #
            # save_pickle(das_binary_recon,
            #             path=os.path.join(out_dir,
            #                               'binary-das/id%d.pickle' % ii))
            # # das_binary_intensity = np.max(np.abs(das_binary_recon))
            # # das_part_binary[rho_idx].append(das_binary_intensity)
            # #
            # # logger.info('Binary DAS (domain partitioning). Intensity = %.3f.'
            # #             ' Size of the intensity sub-array = %d'
            # #             % (das_binary_intensity,
            # #                len(das_part_binary[rho_idx])))
            # #
            # # plot_fd_img(img=np.abs(das_binary_recon), tum_x=tum_x,
            # #             tum_y=tum_y, tum_rad=tum_rad, adi_rad=adi_rad, ox=x_cm,
            # #             oy=y_cm, ant_rad=ant_rad, roi_rad=roi_rad,
            # #             img_rad=roi_rad, title=plt_str, save_fig=True,
            # #             save_str=os.path.join(out_dir,
            # #                                   'id_%d_das_binary.png'
            # #                                   % ii), save_close=True)
            #
            # # 3. Frequency-dependent DAS (zero cond)
            # das_freq_dep_zero_cond_recon = fd_das_vel_freq(
            #     fd_data=adi_cal_cropped,
            #     int_f_xs=int_f_xs,
            #     int_f_ys=int_f_ys,
            #     int_b_xs=int_b_xs,
            #     int_b_ys=int_b_ys,
            #     velocities=
            #     velocities_zero_cond,
            #     ant_rad=ant_rad,
            #     freqs=scan_fs[tar_fs],
            #     adi_rad=adi_rad,
            #     m_size=__M_SIZE,
            #     roi_rad=roi_rad,
            #     air_speed=__VAC_SPEED,
            #     worker_pool=worker_pool)
            #
            # save_pickle(das_freq_dep_zero_cond_recon,
            #             path=os.path.join(out_dir,
            #                               'freq_dep_non_cond-das/id%d.pickle'
            #                               % ii))
            #
            # # das_freq_dep_zero_cond_intensity = \
            # #     np.max(np.abs(das_freq_dep_zero_cond_recon))
            # # das_freq_dep_zero_cond[rho_idx]. \
            # #     append(das_freq_dep_zero_cond_intensity)
            # #
            # # logger.info('Frequency-dependent DAS (zero cond).'
            # #             ' Intensity = %.3f.'
            # #             ' Size of the intensity sub-array = %d'
            # #             % (das_freq_dep_zero_cond_intensity,
            # #                len(das_freq_dep_zero_cond[rho_idx])))
            # #
            # # plot_fd_img(img=np.abs(das_freq_dep_zero_cond_recon), tum_x=tum_x,
            # #             tum_y=tum_y, tum_rad=tum_rad, adi_rad=adi_rad, ox=x_cm,
            # #             oy=y_cm, ant_rad=ant_rad, roi_rad=roi_rad,
            # #             img_rad=roi_rad, title=plt_str, save_fig=True,
            # #             save_str=os.path.join(out_dir,
            # #                                   'id_%d_das'
            # #                                   '_freq_dep_zero_cond.png'
            # #                                   % ii), save_close=True)
            #
            # # 4. Frequency-dependent DAS
            # das_freq_dep_recon = fd_das_vel_freq(fd_data=adi_cal_cropped,
            #                                      int_f_xs=int_f_xs,
            #                                      int_f_ys=int_f_ys,
            #                                      int_b_xs=int_b_xs,
            #                                      int_b_ys=int_b_ys,
            #                                      velocities=velocities,
            #                                      ant_rad=ant_rad,
            #                                      freqs=scan_fs[tar_fs],
            #                                      adi_rad=adi_rad,
            #                                      m_size=__M_SIZE,
            #                                      roi_rad=roi_rad,
            #                                      air_speed=__VAC_SPEED,
            #                                      worker_pool=worker_pool)
            # save_pickle(das_freq_dep_recon,
            #             path=os.path.join(out_dir,
            #                               'freq_dep-das/id%d.pickle' % ii))
            #
            # # das_freq_dep_intensity = np.max(np.abs(das_freq_dep_recon))
            # # das_freq_dep[rho_idx].append(das_freq_dep_intensity)
            # #
            # # logger.info('Frequency-dependent DAS.'
            # #             ' Intensity = %.3f.'
            # #             ' Size of the intensity sub-array = %d'
            # #             % (das_freq_dep_intensity,
            # #                len(das_freq_dep[rho_idx])))
            # #
            # # plot_fd_img(img=np.abs(das_freq_dep_recon), tum_x=tum_x,
            # #             tum_y=tum_y, tum_rad=tum_rad, adi_rad=adi_rad, ox=x_cm,
            # #             oy=y_cm, ant_rad=ant_rad, roi_rad=roi_rad,
            # #             img_rad=roi_rad, title=plt_str, save_fig=True,
            # #             save_str=os.path.join(out_dir,
            # #                                   'id_%d_das_freq_dep.png'
            # #                                   % ii), save_close=True)
            #
            # 5. Ray-tracing
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

            # das_rt_intensity = np.max(np.abs(das_rt_recon))
            # das_rt[rho_idx].append(das_rt_intensity)
            #
            # logger.info('Frequency-dependent DAS, ray-tracing'
            #             ' Intensity = %.3f.'
            #             ' Size of the intensity sub-array = %d'
            #             % (das_rt_intensity,
            #                len(das_rt[rho_idx])))
            #
            # plot_fd_img(img=np.abs(das_rt_recon), cs=cs, tum_x=tum_x,
            #             tum_y=tum_y, tum_rad=tum_rad, adi_rad=adi_rad, ox=x_cm,
            #             oy=y_cm, ant_rad=ant_rad, roi_rad=roi_rad,
            #             img_rad=roi_rad, title=plt_str, save_fig=True,
            #             save_str=os.path.join(out_dir,
            #                                   'id_%d_das_rt.png'
            #                                   % ii), save_close=True)

            ############################################################

            # Get the one-way propagation times for each pixel,
            # for each antenna position and intersection points
            # start = perf_counter()
            # pix_ts, int_f_xs, int_f_ys, int_b_xs, int_b_ys = \
            #     get_pix_ts(ant_rad=ant_rad, m_size=__M_SIZE,
            #                roi_rad=roi_rad, air_speed=__VAC_SPEED,
            #                breast_speed=breast_speed, adi_rad=adi_rad,
            #                ox=x_cm, oy=y_cm, worker_pool=worker_pool)
            # end = perf_counter()
            # intersection_times = np.append(intersection_times, (end - start))
            # logger.info('Parallel pix_ts (circle): %f s' % (end - start))

            # Get the phase factor for efficient computation
            # phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

            # td = iczt(adi_cal_cropped, ini_t=0.5e-9,
            #           fin_t=5.5e-9, n_time_pts=700, ini_f=2e9, fin_f=9e9)
            # ts = np.linspace(0.5, 5.5, 700)
            # angles = np.linspace(0,
            #                     (355/360)*2*np.pi, 72) - np.deg2rad(136.0)
            # rho = np.array([])
            #
            #
            # out_dir_pks = os.path.join(out_dir, 'Peaks/')
            # verify_path(out_dir_pks)
            #
            # for ant_pos in range(np.size(td, axis=1)):
            #     col = np.abs(td[:, ant_pos])
            #     max_int = np.max(col)
            #     peaks, _ = find_peaks(col, height=max_int * 2./3.)
            #     rad = ant_rad - ts[peaks] * 1e-9 * 3e8 / 2
            #     rho = np.append(rho, rad)
            #     phi = angles[ant_pos]
            #     x = rad * np.cos(phi)
            #     y = rad * np.sin(phi)
            #     plt.plot(col, 'b-')
            #     plt.plot(peaks, col[peaks], 'rx',
            #              label="Time response: %f ns\nRadius: %f cm\n"
            #              "x = %f cm, y=%f cm"
            #              % (ts[peaks], rad * 100, x*100, y*100))
            #     plt.xlabel('Indices of time of response array')
            #     plt.ylabel('Intensity')
            #     plt.gca().set_ylim(bottom=0, top=max_int + 2e-3)
            #     plt.plot((peaks[0], peaks[0]), (col[peaks[0]], 0), 'b--')
            #     plt.plot(peaks, 0, 'ro')
            #     plt.text(peaks + 2, 0.0005, 'Index', fontsize=12)
            #     plt.tight_layout()
            #     plt.legend()
            #     plt.savefig(os.path.join(out_dir_pks,
            #                              'peaks_%d.png' % ant_pos))
            #     plt.close()
            #     # plt.show()

            # rec_start = perf_counter()
            # das_adi_recon = fd_das_vel_freq(fd_data=adi_cal_cropped,
            #                                 int_f_xs=int_f_xs,
            #                                 int_f_ys=int_f_ys,
            #                                 int_b_xs=int_b_xs,
            #                                 int_b_ys=int_b_ys,
            #                                 velocities=velocities,
            #                                 ant_rad=ant_rad,
            #                                 freqs=scan_fs[tar_fs],
            #                                 adi_rad=adi_rad,
            #                                 m_size=__M_SIZE, roi_rad=roi_rad,
            #                                 air_speed=__VAC_SPEED,
            #                                 worker_pool=worker_pool)
            # rec_end = perf_counter()
            # recon_times = np.append(recon_times, (rec_end - rec_start))
            # logger.info('Reconstruction: %f s' % (rec_end - rec_start))

            # rec_start = perf_counter()
            # das_adi_recon = fd_das(fd_data=adi_cal_cropped,
            #                        phase_fac=phase_fac,
            #                        freqs=scan_fs[tar_fs],
            #                        worker_pool=worker_pool)
            # rec_end = perf_counter()
            # logger.info('Reconstruction: %f s' % (rec_end - rec_start))

            # save_pickle(das_adi_recon,
            #             os.path.join(expt_adi_out_dir, 'das_adi.pickle'))

            # bound_x, bound_y = \
            # find_boundary(np.abs(das_adi_recon), roi_rad, n_slices=120)
            # rho, phi = cart_to_polar(bound_x, bound_y)
            # cs = polar_fit_cs(rho, angles)

            # ant_pos_x, ant_pos_y = get_ant_scan_xys(ant_rad, 72, -136.0)
            # pix_xs, pix_ys = get_xy_arrs(__M_SIZE, roi_rad)
            # int_f_xs, int_f_ys, int_b_xs, int_b_ys = \
            #     find_xy_ant_bound_circle(ant_xs=ant_pos_x, ant_ys=ant_pos_y,
            #                              n_ant_pos=72, pix_xs=pix_xs[0, :],
            #                             pix_ys=pix_ys[:, 0], adi_rad=adi_rad)
            #
            # plot_fd_img_with_intersections(img=np.abs(das_adi_recon), cs=cs,
            #                                ant_pos_x=ant_pos_x[0],
            #                                ant_pos_y=ant_pos_y[0],
            #                                pix_xs=pix_xs[0, :],
            #                                pix_ys=pix_ys[:, 0],
            #                                int_f_xs=int_f_xs,
            #                                int_f_ys=int_f_ys,
            #                                int_b_xs=int_b_xs,
            #                                int_b_ys=int_b_ys,
            #                                tum_x=tum_x, tum_y=tum_y,
            #                                tum_rad=tum_rad,
            #                                adi_rad=adi_rad, ant_rad=ant_rad,
            #                                roi_rad=roi_rad, img_rad=roi_rad,
            #                                title=plt_str)

            # Plot the DAS reconstruction
            # plot_fd_img(img=np.abs(das_adi_recon), cs=cs, tum_x=tum_x,
            #             tum_y=tum_y, tum_rad=tum_rad, adi_rad=adi_rad, ox=x_cm,
            #             oy=y_cm, ant_rad=ant_rad, roi_rad=roi_rad,
            #             img_rad=roi_rad, title=plt_str, save_fig=True,
            #             save_str=os.path.join(out_dir,
            #                                   'id_%d_das_circle.png'
            #                                   % ii), save_close=True)

            # full_time_end = perf_counter()
            # logger.info('Full time for the scan: %f s' %
            #             (full_time_end-full_time_start))
            # full_times = np.append(full_times, (full_time_end-full_time_start))

    # intensity_dict = {
    #     'rhos': rhos,
    #     'das_regular': das_regular,
    #     'das_part_binary': das_part_binary,
    #     'das_freq_dep_zero_cond': das_freq_dep_zero_cond,
    #     'das_freq_dep': das_freq_dep,
    #     'das_rt': das_rt,
    # }

    worker_pool.close()
    worker_pool.join()
    # save_pickle(intensity_dict, os.path.join(out_dir, 'intensity_dict.pickle'))
    # print('Average time for intersection calculation (ray-tracing): %f s'
    #       % np.average(intersection_times))
    # print('Average time for frequency dependent DAS reconstruction: %f s'
    #       % np.average(recon_times))
    # print('Average time per scan: %f s'
    #       % np.average(full_times))
