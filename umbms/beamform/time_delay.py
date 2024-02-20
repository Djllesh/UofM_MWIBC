"""
Tyson Reimer
University of Manitoba
November 7, 2018
"""

import numpy as np
from functools import partial

import scipy.constants
from umbms.beamform.intersections_analytical import (find_xy_ant_bound_circle,
                                                     find_xy_ant_bound_ellipse,
                                                     _parallel_find_bound_circle_pix)

from umbms.beamform.utility import get_xy_arrs, get_ant_scan_xys

###############################################################################

__GHz = 1e9  # Conversion factor from Hz to GHz

# Speed of light in a vacuum
__VACUUM_SPEED = scipy.constants.speed_of_light


###############################################################################

def get_pix_ts(ant_rad, m_size, roi_rad, air_speed, *, n_ant_pos=72,
               ini_ant_ang=-136.0, breast_speed=0.0, adi_rad=0.0,
               ox=0.0, oy=0.0, mid_breast_max=0.0, mid_breast_min=0.0,
               int_f_xs=None, int_f_ys=None, int_b_xs=None, int_b_ys=None,
               worker_pool=None, partial_ant_idx=None):
    """Get one-way pixel response times

    Parameters
    ----------
    ant_rad : float
        Antenna radius, after correcting for the antenna time delay
    m_size : int
        Size of image-space along one dimension
    roi_rad : float
        Radius of the region of interest that will define the spatial
        extent of the image-space
    air_speed : float
        The estimated propagation speed of the signal in air
    n_ant_pos : int
        Number of antenna positions used in the scan
    ini_ant_ang : float
        Polar angle of initial antenna position
    breast_speed : float
        The estimated propagation speed of the signal in phantom
    adi_rad : float
        Approximate radius of a phantom slice
    ox : float
        x_coord of the centre of the circle
    oy : float
        y_coord of the centre of the circle
    mid_breast_max : float
        major semi-axis of a phantom (b)
    mid_breast_min : float
        minor semi-axis of a phantom (a)
    int_f_xs : array_like n_ant_pos x m_size x m_size
        x_coords of front intersection
    int_f_ys : array_like n_ant_pos x m_size x m_size
        y_coords of front intersection
    int_b_xs : array_like n_ant_pos x m_size x m_size
        x_coords of back intersection
    int_b_ys : array_like n_ant_pos x m_size x m_size
        y_coords of back intersection
    worker_pool : multiprocessing.pool.Pool
        Pool of workers for parallel computation
    partial_ant_idx : array_like 1 x n_ant_pos or None
        Binary mask of indices of antenna positions that are being used
        (if reconstructing specific antenna positions)
    Returns
    -------
    pix_ts : array_like, n_ant_pos x m_size x m_size
        One-way response times of all pixels in the NxN image-space.
        M is the number of antenna positions.
    int_f_xs : array-like n_ant_pos x m_size x m_size
        x-coordinates of each front intersection
    int_f_ys : array-like n_ant_pos x m_size x m_size
        y-coordinates of each front intersection
    int_b_xs : array-like n_ant_pos x m_size x m_size
        x-coordinates of each back intersection
    int_b_ys : array-like n_ant_pos x m_size x m_size
        y-coordinates of each back intersection
    """

    # Get antenna x/y positions during scan
    ant_xs, ant_ys = get_ant_scan_xys(ant_rad=ant_rad, n_ant_pos=n_ant_pos,
                                      ini_ant_ang=ini_ant_ang)

    # Create arrays of pixel x/y positions
    pix_xs, pix_ys = get_xy_arrs(m_size=m_size, roi_rad=roi_rad)

    if int_f_xs is None:  # if no intersections provided
        if adi_rad != 0:  # if the shape is not a circle
            if worker_pool is not None:  # if using parallel computation
                int_f_xs, int_f_ys, int_b_xs, int_b_ys = \
                    get_circle_intersections_parallel(n_ant_pos, m_size,
                                                      ant_xs, ant_ys, pix_xs,
                                                      pix_ys,
                                                      adi_rad, ox, oy,
                                                      worker_pool)
            else:
                int_f_xs, int_f_ys, int_b_xs, int_b_ys \
                    = find_xy_ant_bound_circle(ant_xs, ant_ys, n_ant_pos,
                                               pix_xs[0, :], pix_ys[:, 0],
                                               adi_rad, ox=ox, oy=oy)
        else:
            # TODO: parallel version of ellipse intersections
            int_f_xs, int_f_ys, int_b_xs, int_b_ys \
                = find_xy_ant_bound_ellipse(ant_xs, ant_ys, n_ant_pos,
                                            pix_xs[0, :], pix_ys[:, 0],
                                            mid_breast_max, mid_breast_min)

    return calculate_time_delays(n_ant_pos, m_size, ant_xs, ant_ys,
                                 pix_xs, pix_ys, int_f_xs, int_f_ys, int_b_xs,
                                 int_b_ys, air_speed, breast_speed,
                                 partial_ant_idx), \
           int_f_xs, int_f_ys, int_b_xs, int_b_ys


def calculate_time_delays(n_ant_pos, m_size, ant_xs, ant_ys, pix_xs, pix_ys,
                          int_f_xs, int_f_ys, int_b_xs, int_b_ys, air_speed,
                          breast_speed, partial_ant_idx):
    """Calculates time delays for given set of intersections
    for all antenna positions

    Parmeters:
    -------------
    n_ant_pos : int
        Number of antenna positions used in the scan
    m_size : int
        Size of image-space along one dimension
    ant_xs : array_like 1 x n_ant_pos
        The x-positions in meters of each antenna position used in the
        scan
    ant_ys : array_like 1 x n_ant_pos
        The y-positions in meters of each antenna position used in the
        scan
    pix_xs : array_like m_size x m_size
        A 2D arr. Each element in the arr contains the x-position of
        that pixel in the model, in meters
    pix_ys : array_like m_size x m_size
        A 2D arr. Each element in the arr contains the y-position of
        that pixel in the model, in meters
    int_f_xs : array-like n_ant_pos x m_size x m_size
        x-coordinates of each front intersection
    int_f_ys : array-like n_ant_pos x m_size x m_size
        y-coordinates of each front intersection
    int_b_xs : array-like n_ant_pos x m_size x m_size
        x-coordinates of each back intersection
    int_b_ys : array-like n_ant_pos x m_size x m_size
        y-coordinates of each back intersection
    air_speed : float
        The estimated propagation speed of the signal in air
    breast_speed : float
        The estimated propagation speed of the signal in phantom
    partial_ant_idx : array_like 1 x n_ant_pos or None
        Binary mask of indices of antenna positions that are being used
        (if reconstructing specific antenna positions)

    Returns:
    -------------
    pix_ts : array_like n_ant_pos x m_size x m_size
        One-way response times of all pixels in the NxN image-space.
        M is the number of antenna positions.
    """

    if partial_ant_idx is None:
        # Init array for storing pixel time-delays
        pix_ts = np.zeros([n_ant_pos, m_size, m_size])
        ant_pos_idxs = np.arange(n_ant_pos)

    else:
        # Init array for storing pixel time-delays (first axis size is
        # dependent on the quantity of antennas that are being used in
        # the reconstruction)
        pix_ts = np.zeros([np.count_nonzero(partial_ant_idx), m_size, m_size])
        ant_pos_idxs = np.arange(n_ant_pos)[partial_ant_idx]

    # success_count = 0

    # for a_pos in range(n_ant_pos):  # For each antenna position
    for (a_pos, ts_ant_pos) in \
            zip(ant_pos_idxs, range(np.size(pix_ts, axis=0))):
        # Find x/y position differences of each pixel from antenna
        # x_diffs = pix_xs - ant_xs[a_pos]
        # y_diffs = pix_ys - ant_ys[a_pos]

        pix_to_back_xs = pix_xs - int_b_xs[a_pos]
        pix_to_back_ys = pix_ys - int_b_ys[a_pos]

        back_to_front_xs = int_b_xs[a_pos] - int_f_xs[a_pos]
        back_to_front_ys = int_b_ys[a_pos] - int_f_ys[a_pos]

        front_to_ant_xs = int_f_xs[a_pos] - ant_xs[a_pos]
        front_to_ant_ys = int_f_ys[a_pos] - ant_ys[a_pos]

        # all_x_dists = pix_to_back_xs + back_to_front_xs + front_to_ant_xs
        # all_y_dists = pix_to_back_ys + back_to_front_ys + front_to_ant_ys
        #
        # is_dist = np.allclose(x_diffs, all_x_dists, atol=1e-15, rtol=0) and \
        #           np.allclose(y_diffs, all_y_dists, atol=1e-15, rtol=0)

        air_td_back = np.sqrt(pix_to_back_xs ** 2 + pix_to_back_ys ** 2) \
                      / air_speed
        breast_td = np.sqrt(back_to_front_xs ** 2 + back_to_front_ys ** 2) \
                    / breast_speed
        air_td_front = np.sqrt(front_to_ant_xs ** 2 + front_to_ant_ys ** 2) \
                       / air_speed

        all_td = air_td_front + breast_td + air_td_back

        # all_old_td = np.sqrt(x_diffs ** 2 + y_diffs ** 2) / air_speed
        #
        # is_td = np.allclose(all_td, all_old_td, atol=1e-11, rtol=0.)
        # if is_dist and is_td:
        #     success_count += 1

        # Calculate one-way time-delay of propagation from antenna to
        # each pixel
        pix_ts[ts_ant_pos, :, :] = all_td

    # if success_count == n_ant_pos:
    #     print("Thread ID: %d" % os.getpid())
    #     print("Correct implementation!")
    # else:
    #     print("Thread ID: %d" % os.getpid())
    #     print("ERROR! Something's wrong.")
    #

    return pix_ts


def get_pix_ts_old(ant_rad, m_size, roi_rad, speed, *, n_ant_pos=72,
                   ini_ant_ang=-136.0):
    """Get one-way pixel response times
    Parameters
    ----------
    ant_rad : float
        Antenna radius, after correcting for the antenna time delay
    m_size : int
        Size of image-space along one dimension
    roi_rad : float
        Radius of the region of interest that will define the spatial
        extent of the image-space
    speed : float
        The estimated propagation speed of the signal
    n_ant_pos : int
        Number of antenna positions used in the scan
    ini_ant_ang : float
        Polar angle of initial antenna position
    Returns
    -------
    p_ts : array_like, MxNxN
        One-way response times of all pixels in the NxN image-space.
        M is the number of antenna positions.
    """

    # Get antenna x/y positions during scan
    ant_xs, ant_ys = get_ant_scan_xys(ant_rad=ant_rad, n_ant_pos=n_ant_pos,
                                      ini_ant_ang=ini_ant_ang)

    # Create arrays of pixel x/y positions
    pix_xs, pix_ys = get_xy_arrs(m_size=m_size, roi_rad=roi_rad)

    # Init array for storing pixel time-delays
    p_ts = np.zeros([n_ant_pos, m_size, m_size])

    for a_pos in range(n_ant_pos):  # For each antenna position

        # Find x/y position differences of each pixel from antenna
        x_diffs = pix_xs - ant_xs[a_pos]
        y_diffs = pix_ys - ant_ys[a_pos]

        # Calculate one-way time-delay of propagation from antenna to
        # each pixel
        p_ts[a_pos, :, :] = np.sqrt(x_diffs ** 2 + y_diffs ** 2) / speed

    return p_ts


def get_circle_intersections_parallel(n_ant_pos, m_size, ant_xs, ant_ys,
                                      pix_xs, pix_ys, adi_rad, ox, oy,
                                      worker_pool):
    """Finds breast boundary intersection coordinates
    with propagation trajectory from antenna position
    to corresponding pixel (for parallel calculation)

    Parameters
    ----------
    n_ant_pos : int
        Number of antenna positions
    m_size : int
        Size of image-space along one dimension
    ant_xs : array_like Mx1
        Antenna x-coordinates
    ant_ys : array_like Mx1
        Antenna y-coordinates
    pix_xs : array_like m_size x m_size
        Positions of x-coordinates of each pixel
    pix_ys : array_like m_size x m_size
        Positions of y-coordinates of each pixel
    adi_rad : float
        Approximate radius of a phantom
    ox : float
        x_coord of the centre of the circle
    oy : float
        y_coord of the centre of the circle
    worker_pool : multiprocessing.pool.Pool
        Pool of workers for parallel computation
    Returns
    ----------
    int_f_xs : array_like n_ant_pos x m_size x m_size
        x-coordinates of each front intersection
    int_f_ys : array_like n_ant_pos x m_size x m_size
        y-coordinates of each front intersection
    int_b_xs : array_like n_ant_pos x m_size x m_size
        x-coordinates of each back intersection
    int_b_ys : array_like n_ant_pos x m_size x m_size
        y-coordinates of each back intersection
    """

    iterable_idx = range(n_ant_pos * m_size * m_size)
    parallel_func = partial(_parallel_find_bound_circle_pix, ant_xs, ant_ys,
                            n_ant_pos, pix_xs[0, :], pix_ys[:, 0], adi_rad,
                            ox, oy)

    # asynchronously find all the intersections
    intersections = np.array(worker_pool.map(parallel_func, iterable_idx))

    # the shape of the output is [n_ant_pos * m_size**2, 4]
    # the order is - [front x, front y, back x, back y]
    # C-style reshape each column to [n_ant_pos, m_size, m_size]
    int_f_xs = np.reshape(intersections[:, 0], [n_ant_pos, m_size, m_size])
    int_f_ys = np.reshape(intersections[:, 1], [n_ant_pos, m_size, m_size])
    int_b_xs = np.reshape(intersections[:, 2], [n_ant_pos, m_size, m_size])
    int_b_ys = np.reshape(intersections[:, 3], [n_ant_pos, m_size, m_size])

    return int_f_xs, int_f_ys, int_b_xs, int_b_ys


def apply_syst_cor(xs, ys, x_err, y_err, phi_err):
    """Apply systematic error correction

    Parameters
    ----------
    xs : array_like
        Observed x positions of target, in [cm]
    ys : array_like
        Observed y positions of target, in [cm]
    x_err : float
        Systematic x error in observed target position, in [cm]
    y_err : float
        Systematic y error in observed target position, in [cm]
    phi_err
        Systematic phi error in observed target position, in [deg]
    Returns
    -------
    cor_xs2 : array_like
        Corrected observed x positions of target, after applying
        correction for systematic errors, in [cm]
    cor_ys2 : array_like
        Corrected observed x positions of target, after applying
        correction for systematic errors, in [cm]
    """
    cor_xs = xs - x_err
    cor_ys = ys - y_err
    cor_xs2 = (cor_xs * np.cos(np.deg2rad(phi_err))
               - cor_ys * np.sin(np.deg2rad(phi_err)))
    cor_ys2 = (cor_xs * np.sin(np.deg2rad(phi_err))
               + cor_ys * np.cos(np.deg2rad(phi_err)))

    return cor_xs2, cor_ys2


def time_signal_per_antenna_modelled(tar_x, tar_y, tar_rad, ant_rad,
                                     speed, *, int_f_xs=None, int_f_ys=None,
                                     x_idx=None, y_idx=None,
                                     breast_speed=None, air_speed=None,
                                     n_ant_pos=72, ant_ini_ang=-136.):
    """Returns a scalar value of the signal based on the theoretical
    time-delay and 1/r^2 attenuation

    Parameters
    ----------
    tar_x : float
        Target x-coordinate (centre)
    tar_y : float
        Target y-coordinate (centre)
    tar_rad : float
        Target radius
    ant_rad : float
        Radius of antenna trajectory (m)
    speed : float
        Homogeneous speed (m/s)
    int_f_xs : array-like n_ant_pos x m_size x m_size
        x-coordinates of each front intersection
    int_f_ys : array-like n_ant_pos x m_size x m_size
        y-coordinates of each front intersection
    breast_speed : float
        The estimated propagation speed of the signal in phantom
    air_speed : float
        The estimated propagation speed of the signal in air
    n_ant_pos : int
        Number of discrete antenna positions
    ant_ini_ang : float
        Initial antenna angle

    Returns
    -------
    times_signals : array_like
        An array where the first column is times of the theoretical
        response and second is the calculated signal value
    """

    ant_angs = np.flip(np.deg2rad(np.linspace(0, 355, n_ant_pos) +
                                  ant_ini_ang))
    ant_xs = ant_rad * np.cos(ant_angs)
    ant_ys = ant_rad * np.sin(ant_angs)

    # Initialize
    times_signals = np.zeros(shape=(n_ant_pos, 2))

    if int_f_xs is None:

        for ant_pos in range(n_ant_pos):  # For every antenna

            distance = np.sqrt((tar_x - ant_xs[ant_pos]) ** 2 + \
                               (tar_y - ant_ys[ant_pos]) ** 2) - tar_rad

            # Time is 2d/v
            time = 2 * distance / speed + 2 * 0.19e-9
            # Signal is attenuated
            signal = 1 / distance**2

            times_signals[ant_pos, 0] = time
            times_signals[ant_pos, 1] = signal
    else: # Binary

        for ant_pos in range(n_ant_pos):  # For every antenna

            distance = np.sqrt((tar_x - ant_xs[ant_pos]) ** 2 + \
                               (tar_y - ant_ys[ant_pos]) ** 2) - tar_rad

            # Time is 2d/v
            air_to_plastic = np.sqrt(
                (int_f_xs[ant_pos, y_idx, x_idx] - ant_xs[ant_pos])**2 +
                (int_f_ys[ant_pos, y_idx, x_idx] - ant_ys[ant_pos])**2)
            plastic_to_target = np.sqrt(
                (int_f_xs[ant_pos, y_idx, x_idx] - tar_x)**2 +
                (int_f_ys[ant_pos, y_idx, x_idx] - tar_y)**2) - tar_rad
            time = 2 * air_to_plastic / air_speed + \
                   2 * plastic_to_target / breast_speed + 2 * 0.19e-9

            # Signal is attenuated
            signal = 1 / distance ** 2

            # signal = 1
            times_signals[ant_pos, 0] = time
            times_signals[ant_pos, 1] = signal

    return times_signals
