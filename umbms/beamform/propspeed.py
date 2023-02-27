"""
Tyson Reimer
University of Manitoba
November 8 2018
"""

import numpy as np

from umbms.beamform.breastmodels import get_breast, get_roi
from umbms.beamform.extras import (apply_ant_t_delay, get_ant_scan_xys,
                                   get_pixdist_ratio, get_ant_xy_idxs,
                                   get_xy_arrs)
from umbms.beamform.raytrace import parallel_time_raytrace, get_tof

###############################################################################

__VAC_SPEED = 3e8  # Define propagation speed in vacuum
__VAC_PERMITTIVITY = 8.85e-12
__VAC_PERMEABILITY = 1.25e-6

# Permittivity of the breast tissue analogs used in the lab at the
# central frequency (glycerin for fat, 30% Triton X-100 solution for
# fibroglandular, and saline solution for tumor)
measured_air_perm = 1
measured_adi_perm = 7.08
measured_fib_perm = 44.94
measured_tum_perm = 77.11


###############################################################################


def estimate_speed(adi_rad, ant_rad, m_size=500, new_ant=True):
    """Estimates the propagation speed of the signal in the scan

    Estimates the propagation speed of the microwave signal for
    *all* antenna positions in the scan. Estimates using the average
    propagation speed.

    Parameters
    ----------
    adi_rad : float
        The approximate radius of the breast, in m
    ant_rad : float
        The radius of the antenna trajectory in the scan, as measured
        from the black line on the antenna holder, in m
    m_size : int
        The number of pixels along one dimension used to model the 2D
        imaging chamber
    new_ant : bool
        If True, indicates the 'new' antenna (from 2021) was used
    Returns
    -------
    speed : float
        The estimated propagation speed of the signal at all antenna
        positions, in m/s
    """

    # Correct for antenna phase-delay
    ant_rad = apply_ant_t_delay(ant_rad, new_ant=new_ant)

    # Model the breast as a homogeneous adipose circle, in air
    breast_model = get_breast(m_size=m_size, adi_rad=adi_rad, ant_rad=ant_rad)

    # Get the region of the scan area within the antenna trajectory
    roi = get_roi(ant_rad, m_size, ant_rad)

    # Estimate the speed
    speed = np.mean(__VAC_SPEED / np.sqrt(breast_model[roi]))

    return speed


def get_breast_speed(fibr_perc):
    """ Calculates average breast propagation velocity
    for a given fibroglandular tissue percentage

    Parameters
    -----------
    fibr_perc : integer
        Percentage of fibroglandular tissue in a breast

    Returns
    -----------
    breast_speed : float
        Estimated average propagation velocity
    """

    # calculate average permittivity for a given percentage
    avg_perm = ((100 - fibr_perc) / 100.) * measured_adi_perm +\
               (fibr_perc / 100.) * measured_fib_perm

    breast_speed = __VAC_SPEED / np.sqrt(avg_perm)

    if breast_speed < __VAC_SPEED:
        return breast_speed


def get_speed_from_perm(permittivity):
    """
    Parameters
    -----------
    permittivity : float
        Relative permittivity

    Returns
    -----------
    breast_speed : float
        Propagation velocity in a space with given perm
    """

    breast_speed = __VAC_SPEED / np.sqrt(permittivity)
    return breast_speed


def get_breast_speed_freq(freqs, permittivities, conductivities):
    """ Calculates propagation speed wrt every frequency and
    corresponding permittivities and conductivities

    Parameters
    -----------
    freqs : array_like 1001 X 1
        Linearly spaced frequencies
    permittivities : array_like 1001 X 1
        Permittivities fitted to a Cole-Cole model
    conductivities : array_like 1001 X 1
        Conductivities fitted to a Cole-Coel model

    Returns
    -----------
    breast_speed : array_like 1001 X 1
        Estimated propagation velocities
    """

    beta = 2 * np.pi * freqs * np.sqrt((__VAC_PERMEABILITY * __VAC_PERMITTIVITY
                                        * permittivities / 2) * (np.sqrt(1
                                        + (conductivities / ( 2 * np.pi * freqs
                                        * permittivities * __VAC_PERMITTIVITY))
                                                                    ** 2) + 1))

    return 2 * np.pi * freqs / beta

def get_rt_pix_ts(pix_angs, speed_map, ant_rad, roi_rad=0.13,
                  ini_ant_ang=-136.0, worker_pool=None):
    """Get the response-times for each pixel for each antenna position

    Gets the map of the time-of-propagation of the microwave signal,
    from the antenna to each pixel in the image-space, for each antenna
    position used in the scan.

    Parameters
    ----------
    pix_angs : array_like
        Array of the angle of each pixel off of the central axis of the
        antenna, for every antenna position in the scan, in degrees
    speed_map : array_like
        Map of the estimated propagation speed of the microwave signal
        in each pixel, in m/s
    ant_rad : float
        The radius of the antennas trajectory during the scan, in meters
    roi_rad : float
        The radius of the central region-of-interest used to limit the
        scope of the propagation-time-map computation; the
        time-of-flight will only be computed for pixels within this
        central region of interest
    ini_ant_ang : float
        The initial angular offset of the antenna off of the negative
        x-axis of the image-space (typically 75 deg), in degrees
    worker_pool : multiprocessing.pool.Pool
        Pool of workers for parallel computation

    Returns
    -------
    propagation-time_map : array_like
        The time-of-flight of the signal from the antenna, to each pixel
        in the image-space, for every antenna position, in seconds
    """

    # Find the number of antenna positions used, and the number of pixels
    # along one dimension to define the image-space
    n_ant_pos, m_size, _ = pix_angs.shape

    # Get the x/y-positions of the antenna during the scan
    ant_xs, ant_ys = get_ant_scan_xys(ant_rad, n_ant_pos,
                                      ini_ant_ang=ini_ant_ang)

    # Find the physical width of each pixel, in meters
    pix_width = 1 / get_pixdist_ratio(m_size, roi_rad)

    # Get the x/y coordinates of the antenna during the scan
    # (i.e., the pixel coordinates of the x/y-positions)
    ant_x_idxs, ant_y_idxs = get_ant_xy_idxs(ant_rad, roi_rad, n_ant_pos,
                                             m_size, ini_ant_ang=ini_ant_ang)

    # Init arrays for the indices of the possibly-intersected x/y pixels -
    # created once to save computation time
    possible_x_idxs = np.arange(- m_size / 2 + 1, m_size / 2 + 1)
    possible_y_idxs = np.arange(- m_size / 2 + 1, m_size / 2 + 1)

    # Get the x/y positions of each pixel in the image space
    pix_xs, pix_ys = get_xy_arrs(m_size, roi_rad)

    # Get the distance from the center of the image space for each pixel
    # in the image-space
    pix_dists_from_center = np.sqrt(pix_xs**2 + pix_ys**2)

    # Find the physical locations of each of the planes that are used to
    # separate each pixel in the image-space
    plane_dists = np.linspace(-m_size * pix_width / 2, m_size * pix_width / 2,
                              m_size + 1)

    first_plane_dist = plane_dists[0]  # Find the location of the zeroth plane

    # Make arr storing the positions of each pixel in the image-space
    pix_dists_one_dimension = np.linspace(-roi_rad, roi_rad, m_size)

    if worker_pool is not None:
        # If parallelizing the ray-tracing operation (recommended)

        # Make iterable object, for iterating over every pixel in the
        # image-space, for each antenna position
        iterable_idxs = range(np.size(pix_angs))

        # Make parallel function for ray-tracing for each antenna position
        parallel_func = partial(parallel_time_raytrace, pix_angs,
                                pix_dists_from_center, speed_map, ant_xs,
                                ant_ys, ant_x_idxs, ant_y_idxs, pix_width,
                                first_plane_dist, possible_x_idxs,
                                possible_y_idxs, roi_rad,
                                pix_dists_one_dimension)

        # Call the parallel function to compute the time map, and store the
        # result as an np arr
        prop_times = np.array(worker_pool.map(parallel_func, iterable_idxs))

        # Reshape the returned propagation time map from a 1D arr to the
        # 3D expected arr
        prop_times = np.reshape(prop_times, np.shape(pix_angs))

    # If *not* using parallelization for the ray-tracing (not recommended)
    else:

        # Init arr for storing the propagation time map
        prop_times = np.ones_like(pix_angs)

        for ant_pos in range(n_ant_pos):  # For every antenna position

            # Find the x/y-positions origin of the 'ray' for this
            # antenna position
            ray_ini_x = ant_xs[ant_pos]
            ray_ini_y = ant_ys[ant_pos]

            # Find the x/y-coordinates origin of the 'ray for this
            # antenna position
            ray_ini_x_idx = ant_x_idxs[ant_pos]
            ray_ini_y_idx = ant_y_idxs[ant_pos]

            # Take the abs-value for quickly determining the if-statement below
            pix_angs_here = np.abs(pix_angs[ant_pos])

            # For every pixel in the image-space
            for x_idx in range(m_size):
                for y_idx in range(m_size):

                    # If the pixel is in front of the antenna
                    if pix_angs_here[x_idx, y_idx] < 90:

                        # Compute the time-of-flight for the signal from this
                        # ant_pos to this pixel and back
                        prop_times[ant_pos, x_idx, y_idx] = \
                            get_tof(speed_map, ray_ini_x_idx,
                                    ray_ini_y_idx, x_idx, y_idx,
                                    ray_ini_x, ray_ini_y,
                                    pix_dists_one_dimension[x_idx],
                                    pix_dists_one_dimension[y_idx],
                                    pix_width, first_plane_dist,
                                    possible_x_idxs,
                                    possible_y_idxs)

    return np.flip(prop_times, axis=2)
