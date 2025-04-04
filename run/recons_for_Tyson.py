"""
Illia Prykhodko

Univerity of Manitoba,
May 31st, 2023
"""

# This script is a layout of all the novel algorithms developed during
# GRI-GRA internships
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas
from time import perf_counter
import multiprocessing as mp
import scipy.constants

from umbms import get_proj_path, verify_path, get_script_logger
from umbms.loadsave import load_pickle, save_pickle
from umbms.hardware.antenna import apply_ant_pix_delay, to_phase_center
from umbms.beamform.das import fd_das, fd_das_freq_dep
from umbms.beamform.time_delay import get_pix_ts, get_pix_ts_old
from umbms.beamform.utility import apply_ant_t_delay, get_fd_phase_factor
from umbms.boundary.boundary_detection import get_boundary_iczt, \
    get_binary_mask
from umbms.beamform.propspeed import estimate_speed, get_breast_speed_freq
from umbms.boundary.raytrace import find_boundary_rt
from umbms.plot.imgplots import plot_fd_img

__CPU_COUNT = mp.cpu_count()

# SPECIFY CORRECT DATA AND OUTPUT PATHS
########################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/cyl_phantom/')
__OUT_DIR = os.path.join(get_proj_path(), 'output/cyl_phantom/')
verify_path(__OUT_DIR)
__DIEL_DATA_DIR = os.path.join(get_proj_path(), 'data/freq_data/')

__FD_NAME = 'cal_fd.pickle'
__MD_NAME = 'metadata_cyl_phantom_immediate_reference_rescan.pickle'
__DIEL_NAME = 'dgbe90.csv'

########################################################################

# SPECIFY CORRECT SCAN PARAMETERS
########################################################################

# The frequency parameters from the scan
__INI_F = 2e9
__FIN_F = 9e9
__N_FS = 1001

# The size of the reconstructed image along one dimension
__M_SIZE = 150
# __M_SIZE = int(8.0 * 2 / 0.1)
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
    fd_data,_ = load_data()

    n_expts = np.size(fd_data, axis=0)  # The number of individual scans

    # Get the unique ID of each experiment / scan
    expt_ids = np.arange(n_expts)

    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

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
    out_dir = os.path.join(__OUT_DIR, 'recons/Immediate reference/Recons '
                                      'for Tyson/')
    verify_path(out_dir)

    for expt in range(n_expts):  # for all scans

        logger.info('Scan [%3d / %3d]...' % (expt + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = fd_data[expt, :, :]

        # Create a directory for storing .pickle files
        pickle_dir = os.path.join(out_dir, 'pickles/')
        verify_path(pickle_dir)

        # Get metadata for plotting
        scan_rad = 21. / 100
        # tum_x = tar_md['tum_x'] / 100
        # tum_y = tar_md['tum_y'] / 100
        # tum_rad = 0.5 * (tar_md['tum_diam'] / 100)
        #
        # Cylindrical phantom metadata doesn't have such a field,
        # its radius is hard-coded in the scan parameters section
        # adi_rad = tar_md['adi_rad']
        adi_rad = 0.056

        # Correct for how the scan radius is measured (from a
        # point on the antenna stand, not at the SMA connection
        # point)
        # scan_rad += 0.03618

        # Obtain the true rho of the phase center of the antenna
        ant_rad = to_phase_center(meas_rho=scan_rad)

        # Define the radius of the region of interest
        roi_rad = 0.08

        # Get the area of each pixel in the image domain
        dv = ((2 * roi_rad) ** 2) / (__M_SIZE ** 2)

        # Correct for the antenna time delay
        # NOTE: Only the new antenna was used in UM-BMID Gen-3
        # ant_rad = apply_ant_t_delay(scan_rad=scan_rad, new_ant=True)

        # Get the adipose-only and empty reference data
        # for this scan
        adi_fd = fd_data[expt]


        # 5 DIFFERENT RECONSTRUCTIONS
        ############################################################

        # 1. Homogeneous DAS (regular)

        plt_str_regular_das = 'Homogeneous DAS\n'

        logger.info('\tHomogeneous DAS...')
        # To time the method
        reg_das_start = perf_counter()

        # Estimate the average speed for the whole imaging domain
        # Assume homogeneous media and straight line propagation
        speed = estimate_speed(adi_rad=adi_rad, ant_rad=scan_rad,
                               new_ant=True)

        logger.info('\tTime-delay calculation...')

        # Timing time-delay calculation
        time_delay_tp_start = perf_counter()
        pix_ts = get_pix_ts_old(ant_rad=ant_rad, m_size=__M_SIZE,
                                roi_rad=roi_rad, speed=speed)

        # Account for antenna time delay
        pix_ts = apply_ant_pix_delay(pix_ts=pix_ts)
        time_delay_tp_end = perf_counter()
        logger.info('\t\tTime: %.3f s' %
                    (time_delay_tp_end - time_delay_tp_start))
        phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

        logger.info('\tReconstruction...')

        # Time the reconstruction
        recon_start = perf_counter()
        das_regular_recon = fd_das(fd_data=adi_fd,
                                   phase_fac=phase_fac,
                                   freqs=scan_fs,
                                   worker_pool=worker_pool)
        recon_end = perf_counter()
        logger.info('\t\tTime: %.3f s' %
                    (recon_end - recon_start))

        save_pickle(das_regular_recon,
                    path=os.path.join(pickle_dir,
                                      'id%d_regular.pickle' % expt))

        plot_fd_img(img=np.abs(das_regular_recon), roi_rad=roi_rad,  
                    img_rad=roi_rad,
                    title=plt_str_regular_das,
                    save_fig=True,
                    save_str=os.path.join(out_dir,
                                          'id_%d_das_regular.png'
                                          % expt), save_close=True)

        reg_das_end = perf_counter()

        logger.info('\tThe whole reconstruction time: %.3f s' %
                    (reg_das_end - reg_das_start))

        ############################################################
        # 2. Binary DAS (domain partitioning)

        plt_str_binary_das = 'Binary DAS\n'

        logger.info('\tBinary DAS...')
        bin_das_start = perf_counter()
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
        time_delay_tp_end = perf_counter()
        logger.info('\t\tTime: %.3f s' %
                    (time_delay_tp_end - time_delay_tp_start))

        phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

        logger.info('\tReconstruction...')

        recon_start = perf_counter()
        das_binary_recon = fd_das(fd_data=adi_fd,
                                  phase_fac=phase_fac,
                                  freqs=scan_fs,
                                  worker_pool=worker_pool)
        recon_end = perf_counter()
        logger.info('\t\tTime: %.3f' % (recon_end - recon_start))

        save_pickle(das_binary_recon,
                    path=os.path.join(pickle_dir,
                                      'id%d_binary.pickle' % expt))

        plot_fd_img(img=np.abs(das_binary_recon), roi_rad=roi_rad,
                    img_rad=roi_rad, title=plt_str_binary_das,
                    save_fig=True,
                    save_str=os.path.join(out_dir,
                                          'id_%d_das_binary.png'
                                          % expt), save_close=True)

        bin_das_end = perf_counter()

        logger.info('\tThe whole reconstruction time: %.3f s' %
                    (bin_das_end - bin_das_start))

        ############################################################
        # 3. Frequency-dependent DAS (zero cond, short - FDNC)

        plt_str_fdnc_das = 'Frequency-dependent DAS' \
                           ' (zero conductivity)\n'

        logger.info('\tFrequency-dependent DAS (zero conductivity)...')

        fdnc_das_start = perf_counter()

        recon_start = perf_counter()
        logger.info('\tReconstruction...')
        das_freq_dep_zero_cond_recon = fd_das_freq_dep(
            fd_data=adi_fd,
            int_f_xs=int_f_xs,
            int_f_ys=int_f_ys,
            int_b_xs=int_b_xs,
            int_b_ys=int_b_ys,
            velocities=velocities_zero_cond, ant_rad=ant_rad,
            freqs=scan_fs, adi_rad=adi_rad, m_size=__M_SIZE,
            roi_rad=roi_rad, air_speed=__VAC_SPEED,
            worker_pool=worker_pool)
        recon_end = perf_counter()
        logger.info('\t\tTime: %.3f s' % (recon_end - recon_start))

        save_pickle(das_freq_dep_zero_cond_recon,
                    path=os.path.join(pickle_dir,
                                      'id%d_fdnc.pickle'
                                      % expt))

        plot_fd_img(img=np.abs(das_freq_dep_zero_cond_recon),
                    roi_rad=roi_rad, img_rad=roi_rad,
                    title=plt_str_fdnc_das, save_fig=True,
                    save_str=os.path.join(out_dir,
                                          'id_%d_das'
                                          '_freq_dep_zero_cond.png'
                                          % expt), save_close=True)

        fdnc_das_end = perf_counter()
        logger.info('\tThe whole reconstruction time: %.3f s' %
                    (fdnc_das_end - fdnc_das_start))

        ############################################################
        # 4. Frequency-dependent DAS (short - FD)

        plt_str_fd_das = 'Frequency-dependent DAS\n'

        logger.info('\tFrequency-dependent DAS...')

        fd_das_start = perf_counter()

        recon_start = perf_counter()
        logger.info('\tReconstruction...')
        das_freq_dep_zero_cond_recon = fd_das_freq_dep(
            fd_data=adi_fd,
            int_f_xs=int_f_xs,
            int_f_ys=int_f_ys,
            int_b_xs=int_b_xs,
            int_b_ys=int_b_ys,
            velocities=velocities, ant_rad=ant_rad,
            freqs=scan_fs, adi_rad=adi_rad, m_size=__M_SIZE,
            roi_rad=roi_rad, air_speed=__VAC_SPEED,
            worker_pool=worker_pool)
        recon_end = perf_counter()
        logger.info('\t\tTime: %.3f s' % (recon_end - recon_start))

        save_pickle(das_freq_dep_zero_cond_recon,
                    path=os.path.join(pickle_dir,
                                      'id%d_fd.pickle'
                                      % expt))

        plot_fd_img(img=np.abs(das_freq_dep_zero_cond_recon), roi_rad=roi_rad,
                    img_rad=roi_rad,
                    title=plt_str_fd_das, save_fig=True,
                    save_str=os.path.join(out_dir,
                                          'id_%d_das_freq_dep.png' % expt),
                    save_close=True)

        fd_das_end = perf_counter()
        logger.info('\tThe whole reconstruction time: %.3f s' %
                    (fd_das_end - fd_das_start))

        ############################################################
        # 5. Ray-tracing

        plt_str_rt_das = 'DAS with raytracing, frequency-dependent\n'

        logger.info('\tDAS with raytracing...')

        rt_das_start = perf_counter()

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

        logger.info('\tTime-delay calculation...')
        time_delay_tp_start = perf_counter()
        # Recalculate intersection points according to Siddon's algorithm
        int_f_xs, int_f_ys, int_b_xs, int_b_ys = \
            find_boundary_rt(mask, ant_rad, roi_rad,
                             worker_pool=worker_pool)
        time_delay_tp_end = perf_counter()
        logger.info('\t\tTime: %.3f s' %
                    (time_delay_tp_end - time_delay_tp_start))

        logger.info('\tReconstruction...')
        recon_start = perf_counter()
        das_rt_recon = fd_das_freq_dep(fd_data=adi_fd,
                                       int_f_xs=int_f_xs,
                                       int_f_ys=int_f_ys,
                                       int_b_xs=int_b_xs,
                                       int_b_ys=int_b_ys,
                                       velocities=velocities,
                                       ant_rad=ant_rad,
                                       freqs=scan_fs,
                                       adi_rad=adi_rad,
                                       m_size=__M_SIZE, roi_rad=roi_rad,
                                       air_speed=__VAC_SPEED,
                                       worker_pool=worker_pool)
        recon_end = perf_counter()
        logger.info('\t\tTime: %.3f s' % (recon_end - recon_start))

        save_pickle(das_rt_recon,
                    path=os.path.join(pickle_dir,
                                      'id%d_rt.pickle' % expt))

        plot_fd_img(img=np.abs(das_rt_recon), cs=cs,
                    ant_rad=ant_rad, roi_rad=roi_rad,
                    img_rad=roi_rad, title=plt_str_rt_das, save_fig=True,
                    save_str=os.path.join(out_dir,
                                          'id_%d_das_rt.png'
                                          % expt), save_close=True)

        rt_das_end = perf_counter()
        logger.info('\tThe whole reconstruction time: %.3f s' %
                    (rt_das_end - rt_das_start))

    worker_pool.close()