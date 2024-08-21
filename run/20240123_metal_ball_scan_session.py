"""Illia Prykhodko

University of Manitoba,
January 23rd, 2024
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
    plot_known_arc_map

__CPU_COUNT = mp.cpu_count()

# SPECIFy CORRECT DATA AND OUTPUT PATHS
########################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/cyl_phantom/')
__OUT_DIR = os.path.join(get_proj_path(), 'output/cyl_phantom/')
verify_path(__OUT_DIR)

__FD_NAME = '20240123_s11_data.pickle'

########################################################################

# SPECIFy CORRECT SCAN PARAMETERS
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


if __name__ == '__main__':

    # initialize shared worker pool for raytrace/analytical shape
    # time-delay calculations and for reconstruction
    worker_pool = mp.Pool(__CPU_COUNT - 1)

    logger = get_script_logger(__file__)

    # Load the frequency domain data and metadata
    fd_data = load_pickle(os.path.join(__DATA_DIR, __FD_NAME))

    n_expts = np.size(fd_data, axis=0)  # The number of individual scans

    # Get the unique ID of each experiment / scan
    expt_ids = [i for i in range(2)]

    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

    # Calculate the time delay for a target according to different enhs.
    # Assume signal attenuates with 1/r^2
    # Plot
    out_dir = os.path.join(__OUT_DIR, 'recons/Immediate reference/'
                                      '20240109_glass_rod/arc_investigation/')
    verify_path(out_dir)

    for expt in [0]:  # for all scans

        # for expt in [4]:
        logger.info('Scan [%3d / %3d]...' % (expt + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = fd_data[expt, :, :]


        # Get metadata for plotting
        scan_rad = 0.21


        # Cylindrical phantom metadata doesn't have such a field,
        # its radius is hard-coded in the scan parameters section
        # adi_rad = tar_md['adi_rad']
        adi_rad = __PHANTOM_RAD

        # Obtain the true rho of the phase center of the antenna
        ant_rad = to_phase_center(meas_rho=scan_rad)

        # Define the radius of the region of interest
        roi_rad = adi_rad + 0.01

        # Get the area of each pixel in the image domain
        dv = ((2 * roi_rad) ** 2) / (__M_SIZE ** 2)

        # Get the adipose-only and empty reference data
        # for this scan
        adi_fd = fd_data[1, :, :]
        adi_cal_cropped = (tar_fd - adi_fd)

        # HOMOGENEOUS

        plt_str_regular_das = ''

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

        phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

        das_regular_recon = fd_das(fd_data=adi_cal_cropped,
                                   phase_fac=phase_fac,
                                   freqs=scan_fs,
                                   worker_pool=worker_pool)


        plot_fd_img(img=np.abs(das_regular_recon),
                    ant_rad=ant_rad, roi_rad=roi_rad, tum_rad=0.025,
                    img_rad=roi_rad, title=plt_str_regular_das,
                    save_fig=True,
                    save_str=os.path.join(out_dir,
                                          'id_%d_das_regular.png'
                                          % expt), save_close=True,
                    transparent=True)

