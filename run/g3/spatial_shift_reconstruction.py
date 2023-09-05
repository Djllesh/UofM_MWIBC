"""
Illia Prykhodko

University of Manitoba
August 22nd, 2023
"""

import multiprocessing as mp
import os
import time

import numpy as np

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle
from umbms.beamform.das import fd_das

from umbms.beamform.utility import get_fd_phase_factor, apply_ant_t_delay
from umbms.beamform.time_delay import get_pix_ts_old

from umbms.beamform.propspeed import estimate_speed

from umbms.hardware.antenna import apply_ant_pix_delay, to_phase_center

from umbms.plot.imgplots import plot_fd_img, plot_fd_img_differential

from umbms.boundary.boundary_detection import get_boundary_iczt, \
    fd_differential_align, cart_to_polar, time_aligned_kernel, \
    rho_ToR_from_td, shift_cs, extract_delta_t_from_boundary, \
    prepare_fd_data, phase_shift_aligned_boundaries, shift_rot_cs, \
    window_skin_alignment

from umbms.boundary.differential_minimization import \
    minimize_differential_shift, minimize_differential_shift_rot

__CPU_COUNT = mp.cpu_count()
###############################################################################

__D_DIR = os.path.join(get_proj_path(), 'data/umbmid/g3/differential/')

__O_DIR = os.path.join(get_proj_path(),
                       'output/differential-alignment/spatial_shift'
                       '/Validation/')
verify_path(__O_DIR)

# The frequency parameters from the scan
__INI_F = 2e9
__FIN_F = 9e9
__N_FS = 1001
__SCAN_FS = np.linspace(__INI_F, __FIN_F, __N_FS)

__M_SIZE = 150  # Number of pixels along 1-dimension for reconstruction
__ROI_RAD = 0.08  # ROI radius, in [m]

###############################################################################


def get_breast_pair_s11_diffs(s11_data, idx_pairs):
    """

    Parameters
    ----------
    s11_data : array_like
        S11 dataset
    id_pairs : array_like
        The IDs of the left/right 'breast' scans for each pair

    Returns
    -------
    s11_pair_diffs : array_like
        The differences in the S11 of the left/right breasts
    idx_pairs : array_like
        Indices of the left/right breasts in the S11 data
    """

    s11_pair_diffs = np.empty([np.size(idx_pairs), np.size(s11_data, axis=1),
                               np.size(s11_data, axis=2)])

    for ii in range(np.size(s11_data, axis=0)):
        # Get breast data, pre-cal
        left_uncal = s11_data[idx_pairs[ii, 0], :, :]
        right_uncal = s11_data[idx_pairs[ii, 1], :, :]

        # Get the empty chamber reference data
        left_ref = s11_data[md[idx_pairs[ii, 0]]['emp_ref_id'], :, :]
        right_ref = s11_data[md[idx_pairs[ii, 1]]['emp_ref_id'], :, :]

        # Calibrate the left/right data by empty-chamber subtraction
        left_cal = left_uncal - left_ref
        right_cal = right_uncal - right_ref

        # Obtain the differential S11 data in the frequency domain
        s11_pair_diffs[ii, :, :] = left_cal - right_cal

    return s11_pair_diffs


if __name__ == '__main__':
    # initialize shared worker pool for raytrace/analytical shape
    # time-delay calculations and for reconstruction
    worker_pool = mp.Pool(__CPU_COUNT - 1)

    logger = get_script_logger(__file__)

    fd_data = load_pickle('C:/Users/prikh/Desktop/s11_data.pickle')
    md_set = load_pickle('C:/Users/prikh/Desktop/metadata_new.pickle')

    idx_pairs = np.array([[0, ii] for ii in range(1, 4)])

    tum_y = md_set[0]['tum_y']
    tum_x = md_set[0]['tum_x']
    tum_rad = md_set[0]['tum_diam'] / 2

    uncal_unhealthy = fd_data[0, :, :]
    emp_ref = fd_data[md_set[0]['emp_ref_id'], :, :]
    calibrated_unhealthy = uncal_unhealthy - emp_ref

    for md in md_set[1:-1]:

        right_uncal = fd_data[md['id'], :, :]

        right_cal = right_uncal - emp_ref

        # Radius for boundary detection
        ant_rad_bd = md['ant_rad'] / 100 + 0.03618 + 0.0449

        logger.info('Working on shift %.1f mm' % (md['phant_x_shift'] * 10))

        cs_left, x_cm_left, y_cm_left = \
            get_boundary_iczt(adi_emp_cropped=calibrated_unhealthy,
                              ant_rad=ant_rad_bd,
                              out_dir='',
                              ini_f=2e9, fin_f=9e9,
                              ini_t=1e-9, fin_t=2e-9,
                              n_time_pts=1000)

        cs_right, x_cm_right, y_cm_right = \
            get_boundary_iczt(adi_emp_cropped=right_cal,
                              ant_rad=ant_rad_bd,
                              out_dir='',
                              ini_f=2e9, fin_f=9e9,
                              ini_t=1e-9, fin_t=2e-9,
                              n_time_pts=1000)

        shift = minimize_differential_shift_rot(cs_left=cs_left,
                                                cs_right=cs_right)
        logger.info('Estimated shift parameters:\n\tdelta_x = %.3f mm,'
                    '\n\tdelta_y = %.3f mm\n\tdelta_phi = %.3f rad' %
                    (shift[0]*1000, shift[1]*1000, shift[2]))
        cs_right_shifted = shift_rot_cs(cs=cs_right, delta_x=shift[0],
                                        delta_y=shift[1],
                                        delta_phi=shift[2])

        right_cal = phase_shift_aligned_boundaries(
            fd_emp_ref_right=right_cal,
            ant_rad=ant_rad_bd,
            cs_right_shifted=cs_right_shifted, ini_t=1e-9, fin_t=2e-9,
            n_time_pts=1000, ini_f=__INI_F, fin_f=__FIN_F, n_fs=__N_FS,
            scan_ini_f=__INI_F, scan_fin_f=None)

        right_cal = window_skin_alignment(right_cal,
                                          uncal_unhealthy,
                                          ant_rad=ant_rad_bd,
                                          scan_ini_f=__INI_F,
                                          scan_fin_f=__FIN_F)

        fd = right_cal - calibrated_unhealthy


        ant_rad = md['ant_rad'] / 100
        adi_rad = 6
        ant_rho = to_phase_center(meas_rho=ant_rad)
        # Estimate the propagation speed in the imaging domain
        speed = estimate_speed(adi_rad=adi_rad / 100, ant_rad=ant_rho)
        # Get the approximate pixel time delays
        pix_ts = get_pix_ts_old(ant_rad=ant_rho,
                                m_size=__M_SIZE,
                                roi_rad=__ROI_RAD,
                                speed=speed,
                                ini_ant_ang=-136.0)
        phase_fac = get_fd_phase_factor(pix_ts=pix_ts)
        das_recon = fd_das(fd_data=fd, phase_fac=phase_fac,
                           freqs=__SCAN_FS, worker_pool=worker_pool)

        title_str = 'Spatial shift = %.1f cm' % md['phant_x_shift']
        save_str = 'id%d-0_spatial_shift_%.1f.png' % (md['id'],
                                                      md['phant_x_shift'])

        plot_fd_img(img=abs(das_recon), tum_x=tum_x / 100, tum_y=tum_y / 100,
                    tum_rad=tum_rad / 100, adi_rad=adi_rad / 100,
                    roi_rad=__ROI_RAD, img_rad=__ROI_RAD,
                    save_str=os.path.join(__O_DIR, save_str),
                    title=title_str, save_fig=True)

        logger.info('SUCCESS!')

    worker_pool.close()
