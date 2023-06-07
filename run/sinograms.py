"""
Illia Prykhodko
University of Manitoba
September 21st, 2021
"""

import os
import numpy as np
import scipy.constants

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle

from umbms.plot.imgplots import plot_fd_img, plot_fd_img_with_intersections
from umbms.plot import plt_sino, plt_fd_sino

from umbms.beamform.recon import fd_das, fd_dmas, orr_recon
from umbms.beamform.extras import (apply_ant_t_delay, get_pix_ts, get_xy_arrs,
                                   find_xy_ant_bound_circle, get_pix_ts_old,
                                   get_fd_phase_factor, get_ant_scan_xys,
                                   find_xy_ant_bound_ellipse)

from umbms.beamform.propspeed import estimate_speed, get_breast_speed,\
                                     get_speed_from_perm
from umbms.beamform.optimfuncs import td_velocity_deriv

###############################################################################

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

    # The output dir, where the reconstructions will be stored
    out_dir = os.path.join(__OUT_DIR, 'recons/')
    verify_path(out_dir)

    for ii in range(n_expts):

        logger.info('Scan [%3d / %3d]...' % (ii + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = fd_data[ii, :, :]

        plt_sino(fd=tar_fd, title="Experimental Data. ID: %d" % (ii + 1),
                 save_str='experimental_data_id_%d.png' % ii,
                 close=True, out_dir=out_dir, transparent=False)

