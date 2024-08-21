"""
Illia Prykhodko

Univerity of Manitoba,
January 22nd, 2024
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas
from time import perf_counter
import multiprocessing as mp
import scipy.constants

from umbms import get_proj_path, verify_path, get_script_logger
from umbms.beamform.iczt import iczt
from umbms.loadsave import load_pickle, save_pickle
from umbms.hardware.antenna import apply_ant_pix_delay, to_phase_center
from umbms.beamform.das import fd_das, fd_das_freq_dep
from umbms.beamform.time_delay import get_pix_ts, get_pix_ts_old
from umbms.beamform.utility import apply_ant_t_delay, get_fd_phase_factor
from umbms.boundary.boundary_detection import get_boundary_iczt, \
    get_binary_mask
from umbms.beamform.propspeed import estimate_speed, get_breast_speed_freq
from umbms.boundary.raytrace import find_boundary_rt
from umbms.plot.imgplots import plot_fd_img, plot_arc_map, calculate_arc_map, \
    plot_known_arc_map, calculate_arc_map_window
from umbms.plot.sinogramplot import plt_fd_sino

__CPU_COUNT = mp.cpu_count()

# SPECIFY CORRECT DATA AND OUTPUT PATHS
########################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/cyl_phantom/')
__OUT_DIR = os.path.join(get_proj_path(), 'output/cyl_phantom/')
verify_path(__OUT_DIR)
__DIEL_DATA_DIR = os.path.join(get_proj_path(), 'data/freq_data/')

__FD_NAME = '20240109_s11_data.pickle'
__MD_NAME = '20240109_metadata.pickle'
__DIEL_NAME = '20240109_DGBE90.csv'

########################################################################

# SPECIFY CORRECT SCAN PARAMETERS
########################################################################

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

########################################################################

# Define propagation speed in vacuum
__VAC_SPEED = scipy.constants.speed_of_light
__PHANTOM_RAD = 0.0555


def load_data():
    """Loads both fd_data and metadata

    Returns:
    --------
    tuple of two loaded variables
    """
    return load_pickle(os.path.join(__DATA_DIR,
                                    __FD_NAME)), \
        load_pickle(os.path.join(__DATA_DIR,
                                 __MD_NAME))


# For multiprocessing purposes
if __name__ == "__main__":

    # initialize shared worker pool for raytrace/analytical shape
    # time-delay calculations and for reconstruction
    worker_pool = mp.Pool(__CPU_COUNT - 1)

    logger = get_script_logger(__file__)

    # Load the frequency domain data and metadata
    fd_data, metadata = load_data()

    n_expts = np.size(fd_data, axis=0)  # The number of individual scans

    # Get the unique ID of each experiment / scan
    expt_ids = [md['id'] for md in metadata]

    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

    # ICZT times
    iczt_time = np.linspace(__INI_T, __FIN_T, __N_TS)

    # Read .csv file of permittivity and conductivity values
    df = pandas.read_csv(os.path.join(__DIEL_DATA_DIR, __DIEL_NAME))

    # Calculate velocity array
    freqs = np.array(df["Freqs"].values, dtype=float) * 1e6
    permittivities = np.array(df["Permittivity"].values)
    conductivities = np.array(df["Conductivity"].values)
    zero_conductivities = np.zeros_like(conductivities)
    velocities_zero_cond = get_breast_speed_freq(freqs, permittivities,
                                                 zero_conductivities)
    velocities = get_breast_speed_freq(freqs, permittivities, conductivities)
    # plt.rc('font', family='Times New Roman')
    # fig = plt.figure()
    # ax = plt.axes()
    # plt.plot(freqs * 1e-9, velocities * 1e-8)
    # plt.xlim(freqs[0] * 1e-9, freqs[-1] * 1e-9)
    # plt.ylim(velocities[0] * 1e-8, velocities[-1] * 1e-8)
    # x_ticks_nums = np.arange(2, 10)
    # y_ticks_nums = velocities[::142] * 1e-8
    # y_ticks = [r'%.2f $\cdot$ 10$^8$' % y_ticks_nums[i] for i in range(8)]
    # ax.set_xticks(x_ticks_nums)
    # ax.set_yticks(y_ticks_nums)
    # ax.set_yticklabels(y_ticks)
    # plt.xlabel('Frequency (GHz)', fontsize=16)
    # plt.ylabel('Propagation speed m/s', fontsize=16)
    # plt.grid()
    # plt.show()
    # The output dir, where the reconstructions will be stored
    out_dir = os.path.join(__OUT_DIR, 'recons/Immediate reference/'
                                      '20240109_glass_rod/'
                                      'arc_investigation/full_recon')
    verify_path(out_dir)

    for expt in range(n_expts):  # for all scans
        # for expt in [4]:
        logger.info('Scan [%3d / %3d]...' % (expt + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = fd_data[expt, :, :]
        tar_md = metadata[expt]

        # if the scan has both empty and adipose references and is not
        # a rod reference
        if ~np.isnan(tar_md['emp_ref_id']) and \
                ~np.isnan(tar_md['adi_ref_id2']) and \
                tar_md['type'] != "rod reference":

            # If the scan does include a tumour
            if ~np.isnan(tar_md['tum_diam']):

                # Set a str for plotting
                plt_str = "%.1f cm rod in\n" \
                          "ID: %d" % (tar_md['tum_diam'], expt)
            else:
                plt_str = "Empty phantom\n" \
                          "ID: %d" % expt
            # TEMPORARY
            plt_str = ''

            # Create a directory for storing .pickle files
            pickle_dir = os.path.join(out_dir, 'pickles/')
            verify_path(pickle_dir)

            # Get metadata for plotting
            scan_rad = tar_md['ant_rad'] / 100
            tum_x = tar_md['tum_x'] / 100
            tum_y = tar_md['tum_y'] / 100
            tum_rad = 0.5 * (tar_md['tum_diam'] / 100)
            # Cylindrical phantom metadata doesn't have such a field,
            # its radius is hard-coded in the scan parameters section
            # adi_rad = tar_md['adi_rad']
            adi_rad = __PHANTOM_RAD

            # Correct for how the scan radius is measured (from a
            # point on the antenna stand, not at the SMA connection
            # point)
            # scan_rad += 0.03618

            # Obtain the true rho of the phase center of the antenna
            ant_rad = to_phase_center(meas_rho=scan_rad)

            # Define the radius of the region of interest
            roi_rad = adi_rad + 0.01

            # Get the area of each pixel in the image domain
            dv = ((2 * roi_rad) ** 2) / (__M_SIZE ** 2)

            # Correct for the antenna time delay
            # NOTE: Only the new antenna was used in UM-BMID Gen-3
            # ant_rad = apply_ant_t_delay(scan_rad=scan_rad, new_ant=True)

            # Get the adipose-only and empty reference data
            # for this scan
            adi_fd_emp = fd_data[expt_ids.index(tar_md['emp_ref_id']), :, :]
            adi_fd = fd_data[expt_ids.index(tar_md['rod_ref_id']), :, :]
            adi_cal_cropped_emp = (tar_fd - adi_fd_emp)
            adi_cal_cropped = (tar_fd - adi_fd)

            td_data = iczt(fd_data=adi_cal_cropped, ini_t=__INI_T,
                           fin_t=__FIN_T,
                           n_time_pts=__N_TS, ini_f=__INI_F, fin_f=__FIN_F)

            # 5 DIFFERENT RECONSTRUCTIONS
            ############################################################

            # 1. Homogeneous DAS (regular)

            plt_str_regular_das = 'Homogeneous DAS\n%s' % plt_str

            logger.info('\tHomogeneous DAS...')

            # Estimate the average speed for the whole imaging domain
            # Assume homogeneous media and straight line propagation
            speed = estimate_speed(adi_rad=adi_rad, ant_rad=scan_rad,
                                   new_ant=True)

            logger.info('\tTime-delay calculation...')

            pix_ts = get_pix_ts_old(ant_rad=ant_rad, m_size=__M_SIZE,
                                    roi_rad=roi_rad, speed=speed)

            # Account for antenna time delay
            pix_ts = apply_ant_pix_delay(pix_ts=pix_ts)

            arc_map = calculate_arc_map_window(pix_ts=pix_ts,
                                               td_data=td_data,
                                               iczt_time=iczt_time)

            plot_known_arc_map(arc_map=arc_map, img_roi=roi_rad * 100,
                               save_str=os.path.join(out_dir, f'arc_recon_hom_'
                                                              f'{expt}.png'),
                               tar_x=tum_x * 100,
                               tar_y=tum_y * 100, tar_rad=tum_rad * 100)

            ############################################################
            # 2. Binary DAS (domain partitioning)

            plt_str_binary_das = 'Binary DAS\n%s' % plt_str

            logger.info('\tBinary DAS...')
            # Assume average propagation through the adipose layer
            breast_speed = np.average(velocities)

            logger.info('\tTime-delay calculation...')
            time_delay_tp_start = perf_counter()
            pix_ts, int_f_xs, int_f_ys, int_b_xs, int_b_ys = \
                get_pix_ts(ant_rad=ant_rad, m_size=__M_SIZE,
                           roi_rad=roi_rad, air_speed=__VAC_SPEED,
                           breast_speed=breast_speed, adi_rad=adi_rad,
                           worker_pool=worker_pool)

            # Account for antenna time delay
            pix_ts = apply_ant_pix_delay(pix_ts=pix_ts)

            arc_map = calculate_arc_map_window(pix_ts=pix_ts,
                                               td_data=td_data,
                                               iczt_time=iczt_time)

            plot_known_arc_map(arc_map=arc_map, img_roi=roi_rad * 100,
                               save_str=os.path.join(out_dir, f'arc_recon_bin_'
                                                              f'{expt}.png'),
                               tar_x=tum_x * 100,
                               tar_y=tum_y * 100, tar_rad=tum_rad * 100)

            ############################################################
            # 3. Frequency-dependent DAS (zero cond, short - FDNC)

            logger.info('\tFrequency-dependent DAS (zero conductivity)...')
            arc_map = np.zeros(shape=(150, 150))

            for v in velocities_zero_cond[::50]:
                pix_ts, _, _, _, _ = get_pix_ts(ant_rad=ant_rad,
                                                m_size=__M_SIZE,
                                                roi_rad=roi_rad,
                                                air_speed=__VAC_SPEED,
                                                breast_speed=v,
                                                adi_rad=adi_rad,
                                                int_f_xs=int_f_xs,
                                                int_f_ys=int_f_ys,
                                                int_b_xs=int_b_xs,
                                                int_b_ys=int_b_ys)

                arc_map += calculate_arc_map_window(pix_ts=pix_ts,
                                                    td_data=td_data,
                                                    iczt_time=iczt_time)

            plot_known_arc_map(arc_map=arc_map, img_roi=roi_rad * 100,
                               save_str=os.path.join(out_dir,
                                                     f'arc_recon_fdnc_'
                                                     f'{expt}.png'),
                               tar_x=tum_x * 100,
                               tar_y=tum_y * 100, tar_rad=tum_rad * 100)

            ############################################################
            # 4. Frequency-dependent DAS (short - FD)

            logger.info('\tFrequency-dependent DAS...')
            arc_map = np.zeros(shape=(150, 150))

            for v in velocities[::50]:
                pix_ts, _, _, _, _ = get_pix_ts(ant_rad=ant_rad,
                                                m_size=__M_SIZE,
                                                roi_rad=roi_rad,
                                                air_speed=__VAC_SPEED,
                                                breast_speed=v,
                                                adi_rad=adi_rad,
                                                int_f_xs=int_f_xs,
                                                int_f_ys=int_f_ys,
                                                int_b_xs=int_b_xs,
                                                int_b_ys=int_b_ys)

                arc_map += calculate_arc_map_window(pix_ts=pix_ts,
                                                    td_data=td_data,
                                                    iczt_time=iczt_time)

            plot_known_arc_map(arc_map=arc_map, img_roi=roi_rad * 100,
                               save_str=os.path.join(out_dir, f'arc_recon_fd_'
                                                              f'{expt}.png'),
                               tar_x=tum_x * 100,
                               tar_y=tum_y * 100, tar_rad=tum_rad * 100)

            ############################################################
            # 5. Ray-tracing

            logger.info('\tDAS with raytracing...')

            # TEMPORARY:
            # In order to account for straight-line antenna time-delay
            # apply the old correction

            scan_rad += 0.03618
            ant_rad_bound = apply_ant_t_delay(scan_rad=scan_rad, new_ant=True)

            # Routine start: extract the boundary from the sinogram
            cs, x_cm, y_cm = \
                get_boundary_iczt(adi_cal_cropped_emp, ant_rad_bound)

            # Apply the cubic spline onto the grid
            mask = get_binary_mask(cs, m_size=__M_SIZE, roi_rad=roi_rad)

            # Recalculate intersection points according to Siddon's algorithm
            int_f_xs, int_f_ys, int_b_xs, int_b_ys = \
                find_boundary_rt(mask, ant_rad, roi_rad,
                                 worker_pool=worker_pool)

            arc_map = np.zeros(shape=(150, 150))

            for v in velocities[::50]:
                pix_ts, _, _, _, _ = get_pix_ts(ant_rad=ant_rad,
                                                m_size=__M_SIZE,
                                                roi_rad=roi_rad,
                                                air_speed=__VAC_SPEED,
                                                breast_speed=v,
                                                adi_rad=adi_rad,
                                                int_f_xs=int_f_xs,
                                                int_f_ys=int_f_ys,
                                                int_b_xs=int_b_xs,
                                                int_b_ys=int_b_ys)

                arc_map += calculate_arc_map_window(pix_ts=pix_ts,
                                                    td_data=td_data,
                                                    iczt_time=iczt_time)

            plot_known_arc_map(arc_map=arc_map, img_roi=roi_rad * 100,
                               save_str=os.path.join(out_dir, f'arc_recon_rt_'
                                                              f'{expt}.png'),
                               tar_x=tum_x * 100,
                               tar_y=tum_y * 100, tar_rad=tum_rad * 100)
    worker_pool.close()
