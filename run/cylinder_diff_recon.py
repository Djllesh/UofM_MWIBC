"""
Illia Prykhodko
University of Manitoba
September 23rd, 2022
"""

import os
import numpy as np
import multiprocessing as mp

import scipy.constants
from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle

from umbms.plot.imgplots import plot_fd_img, plot_fd_img_with_intersections
from umbms.plot import plt_sino, plt_fd_sino

from umbms.beamform.orr import orr_recon
from umbms.beamform.dmas import fd_dmas
from umbms.beamform.das import fd_das
from umbms.beamform.time_delay import (get_pix_ts,
                                       find_xy_ant_bound_circle,
                                       get_pix_ts_old,
                                       find_xy_ant_bound_ellipse)
from umbms.beamform.utility import get_xy_arrs, apply_ant_t_delay, \
    get_ant_scan_xys, get_fd_phase_factor

from umbms.beamform.propspeed import estimate_speed, get_breast_speed, \
                                     get_speed_from_perm
from umbms.beamform.optimfuncs import td_velocity_deriv

###############################################################################

__CPU_COUNT = mp.cpu_count()

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/cylinder/')

__OUT_DIR = os.path.join(get_proj_path(), 'output/cylinder/')
verify_path(__OUT_DIR)

__FD_NAME = 'fd_data_gen_three_s11.pickle'
__MD_NAME = 'metadata_gen_three.pickle'

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

# The approximate radius of each adipose phantom in our array
__ADI_RADS = {
    'A1': 0.05,
    'A2': 0.06,
    'A3': 0.07,
    'A11': 0.06,
    'A12': 0.05,
    'A13': 0.065,
    'A14': 0.06,
    'A15': 0.055,
    'A16': 0.07
}

__GLASS_CYLINDER_RAD = 0.06
__GLASS_THIKNESS = 0.3 / 100
__SPHERE_RAD = 0.0075
__ANT_RAD = 0.21

__SPHERE_POS = [(np.nan, np.nan),
                (0.0, 5.3),
                (0.0, 4.0),
                (0.0, 3.0),
                (0.0, 2.0),
                (0.0, 1.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (2.0, 0.0),
                (3.0, 0.0),
                (4.0, 0.0),
                (5.0, 0.0),
                (np.nan, np.nan)]

__MID_BREAST_RADS = {
    'A1': (0.053, 0.034),
    'A2': (0.055, 0.051),
    'A3': (0.07, 0.049),
    'A11': (0.062, 0.038),
    'A12': (0.051, 0.049),
    'A13': (0.065, 0.042),
    'A14': (0.061, 0.051),
    'A15': (0.06, 0.058),
    'A16': (0.073, 0.05),
}

__VAC_SPEED = scipy.constants.speed_of_light
# Define propagation speed in vacuum


def load_data():
    """Loads both fd and metadata

    Returns:
    --------
    tuple of two loaded variables
    """
    return load_pickle(os.path.join(__DATA_DIR,
                                    __FD_NAME)), \
           load_pickle(os.path.join(__DATA_DIR,
                                    __MD_NAME))


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Load the frequency domain data and metadata

    fd_data = load_pickle(os.path.join(__DATA_DIR, 's11_fd.pickle'))

    n_expts = np.size(fd_data, axis=0)  # The number of individual scans

    # Get the unique IDs of the adipose-only and adipose-fibroglandular
    # (healthy) reference scans for each experiment/scan
    adi_ref_id_first = 0
    adi_ref_id_second = 12

    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

    # Only retain frequencies above 2 GHz due to antenna VSWR
    tar_fs = scan_fs >= 2e9

    # The output dir, where the reconstructions will be stored
    out_dir = os.path.join(__OUT_DIR, 'recons/')
    verify_path(out_dir)

    # initialize shared worker pool for raytrace/analytical shape
    # time-delay calculations and for reconstruction
    worker_pool = mp.Pool(__CPU_COUNT - 1)

    # Determine fibroglandular percentage
    fibr_perc = 0
    # ii = 48
    # for ii in range(n_expts):  # For each scan / experiment
    for ii in range(n_expts - 1):

        if ii == adi_ref_id_first or ii == adi_ref_id_second:
            continue

        logger.info('Scan [%3d / %3d]...' % (ii + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = fd_data[ii, :, :]

        # Create the output directory for the adipose-only
        # reference reconstructions
        expt_adi_out_dir = os.path.join(out_dir,
                                'id-%d-adi-%.1f-fibr-perc/' % (ii, fibr_perc))
        verify_path(expt_adi_out_dir)

        # Create the output directory for the adipose-fibroglandular
        # reference reconstructions
        expt_fib_out_dir = os.path.join(out_dir, 'id-%d-fib-%.1f-fibr-perc/'
                                        % (ii, fibr_perc))
        verify_path(expt_fib_out_dir)

        # Get metadata for plotting
        scan_rad = __ANT_RAD
        tum_x = __SPHERE_POS[ii][0] / 100
        tum_y = __SPHERE_POS[ii][1] / 100
        tum_rad = __SPHERE_RAD
        adi_rad = __GLASS_CYLINDER_RAD

        # Correct for how the scan radius is measured (from a
        # point on the antenna stand, not at the SMA connection
        # point)
        scan_rad += 0.03618

        # Define the radius of the region of interest
        roi_rad = adi_rad + 0.01

        # Get the area of each pixel in the image domain
        dv = ((2 * roi_rad) ** 2) / (__M_SIZE ** 2)

        # Correct for the antenna time delay
        # NOTE: Only the new antenna was used in UM-BMID Gen-3
        ant_rad = apply_ant_t_delay(scan_rad=scan_rad, new_ant=True)

        # Estimate the propagation speed in the imaging domain
        # speed = estimate_speed(adi_rad=adi_rad, ant_rad=scan_rad,
        #                        new_ant=True)

        breast_speed = get_breast_speed(fibr_perc)

        # Get the one-way propagation times for each pixel,
        # for each antenna position and intersection points
        pix_ts, int_f_xs, int_f_ys, int_b_xs, int_b_ys = \
            get_pix_ts(ant_rad=ant_rad, m_size=__M_SIZE,
                       roi_rad=roi_rad, air_speed=__VAC_SPEED,
                       breast_speed=breast_speed, adi_rad=adi_rad)

        # speed = estimate_speed(adi_rad=adi_rad, ant_rad=scan_rad,
        #                        new_ant=True)

        # pix_ts = get_pix_ts_old(ant_rad=ant_rad, m_size=__M_SIZE,
        #                         roi_rad=roi_rad, speed=speed)

        # Get the phase factor for efficient computation
        phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

        # Get the adipose-only reference data for this scan
        adi_fd_first = fd_data[adi_ref_id_first, :, :]
        adi_fd_second = fd_data[adi_ref_id_second, :, :]

        # reference fd data subtraction
        adi_cal_cropped_first = tar_fd - adi_fd_first
        adi_cal_cropped_second = tar_fd - adi_fd_second

        # If the scan does include a tumour
        if ~np.isnan(__SPHERE_POS[ii][0]):

            # Set a str for plotting
            plt_str = "%.1f cm sphere in " \
                      "ID: %d" % (__SPHERE_RAD * 100 * 2, ii)

        else:  # If the scan does NOT include a tumour
            plt_str = "ID: %d" % ii

        # Reconstruct a DAS image
        das_adi_recon_first = fd_das(fd_data=adi_cal_cropped_first,
                                     phase_fac=phase_fac,
                                     freqs=scan_fs[tar_fs],
                                     worker_pool=worker_pool)

        das_adi_recon_second = fd_das(fd_data=adi_cal_cropped_second,
                                      phase_fac=phase_fac,
                                      freqs=scan_fs[tar_fs],
                                      worker_pool=worker_pool)

        # Plot the DAS substr reconstruction
        plot_fd_img(img=np.abs(das_adi_recon_second - das_adi_recon_first),
                    tum_x=tum_x, tum_y=tum_y, tum_rad=tum_rad, adi_rad=adi_rad,
                    ant_rad=ant_rad, roi_rad=roi_rad, img_rad=roi_rad,
                    title='%s\nAdi Cal' % plt_str, save_fig=True,
                    save_str=os.path.join(out_dir,
                                          'id_%d_diff_das_%.1f_fibr_perc.png'
                                          % (ii, fibr_perc)), save_close=True)

        # Reconstruct a DMAS image
        dmas_adi_recon_first = fd_dmas(fd_data=adi_cal_cropped_first,
                               pix_ts=pix_ts, freqs=scan_fs[tar_fs])

        dmas_adi_recon_second = fd_dmas(fd_data=adi_cal_cropped_second,
                                pix_ts=pix_ts, freqs=scan_fs[tar_fs])

        # Plot the DMAS reconstruction
        plot_fd_img(img=np.abs(dmas_adi_recon_second - dmas_adi_recon_first),
                    tum_x=tum_x, tum_y=tum_y, tum_rad=tum_rad, adi_rad=adi_rad,
                    ant_rad=ant_rad, roi_rad=roi_rad, img_rad=roi_rad,
                    title='%s\nAdi Cal' % plt_str,
                    save_fig=True,
                    save_str=os.path.join(out_dir,
                                          'id_%d_diff_dmas_%.1f_fibr_perc.png'
                                          % (ii, fibr_perc)), save_close=True)
