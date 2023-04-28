"""
Tyson Reimer
University of Manitoba

"""

import numpy as np
import multiprocessing as mp
from functools import partial
from umbms.beamform.extras import get_pix_ts, get_fd_phase_factor


###############################################################################


def get_ref_derivs(phase_fac, fd, fwd, freqs, worker_pool):
    """Get the gradient of the loss func wrt the reflectivities

    Parameters
    ----------
    phase_fac : array_like
        The phase factor of the imaging domain
    fd : array_like
        The frequency domain measured data
    fwd : array_like
        The frequency domain forward projection of the current image
        estimate
    freqs : array_like
        The frequencies used in the scan
    worker_pool : multiprocessing.pool.Pool
        Pool of workers for parallel computation

    Returns
    -------
    ref_derives : array_like
        The gradient of the loss function with respect to the
        reflectivities in the image domain
    """

    # Create func for parallel computation
    parallel_func = partial(_parallel_ref_deriv, phase_fac, fwd, fd,
                            freqs)

    iterable_idxs = range(np.size(fd, axis=0))  # Indices to iterate over

    # Store projections from parallel processing
    all_ref_derivs = np.array(worker_pool.map(parallel_func, iterable_idxs))

    # Reshape
    all_ref_derivs = np.reshape(all_ref_derivs, [np.size(fd, axis=0),
                                                 np.size(phase_fac, axis=1),
                                                 np.size(phase_fac, axis=2)])

    # Sum over all frequencies
    ref_derivs = np.sum(all_ref_derivs, axis=0)

    ref_derivs *= -1  # Apply normalization factor

    return ref_derivs


def get_ref_derivs_vel_freq(int_f_xs, int_f_ys, int_b_xs, int_b_ys, velocities,
                            ant_rad, m_size, roi_rad, air_speed, adi_rad,
                            mid_breast_max, mid_breast_min,
                            fd, fwd, freqs, worker_pool):
    """Get the gradient of the loss func wrt the reflectivities

    Parameters
    ----------
    int_f_xs : array_like m x N x N
        x_coords of front intersection
    int_f_ys : array_like m x N x N
        y_coords of front intersection
    int_b_xs : array_like m x N x N
        x_coords of back intersection
    int_b_ys : array_like m x N x N
        y_coords of back intersection
    velocities : array_like 1001 x 1
        Velocities calculated wrt frequency dependent permittivity
    ant_rad : float
        Antenna radius, after correcting for the antenna time delay
    m_size : int
        Size of image-space along one dimension
    roi_rad : float
        Radius of the region of interest that will define the spatial
        extent of the image-space
    air_speed : float
        The estimated propagation speed of the signal in air
    freqs : array_like, Nx1
        The frequencies used in the scan
    adi_rad : float
        Approximate radius of a phantom slice
    mid_breast_max : float
        major semi-axis of a phantom (b)
    mid_breast_min : float
        minor semi-axis of a phantom (a)
    fd : array_like
        The frequency domain measured data
    fwd : array_like
        The frequency domain forward projection of the current image
        estimate
    worker_pool : multiprocessing.pool.Pool
        Pool of workers for parallel computation

    Returns
    -------
    ref_derives : array_like
        The gradient of the loss function with respect to the
        reflectivities in the image domain
    """

    # Create func for parallel computation
    parallel_func = partial(_parallel_ref_deriv_vel_freq, int_f_xs, int_f_ys,
                            int_b_xs, int_b_ys, velocities, ant_rad, m_size,
                            roi_rad, air_speed, adi_rad, mid_breast_max,
                            mid_breast_min, fwd, fd, freqs)

    # Indices to iterate over
    iterable_idxs = range(np.size(fd, axis=0))

    # Store projections from parallel processing
    all_ref_derivs = np.array(worker_pool.map(parallel_func, iterable_idxs))

    # Reshape
    all_ref_derivs = np.reshape(all_ref_derivs,
                                [np.size(fd, axis=0), m_size, m_size])

    # Sum over all frequencies
    ref_derivs = np.sum(all_ref_derivs, axis=0)

    ref_derivs *= -1  # Apply normalization factor

    return ref_derivs


def get_ref_derivs_speed(phase_fac, fd, fwd, freqs, dv, int_f_xs, int_f_ys,
                         int_b_xs, int_b_ys, speed, worker_pool):
    """Get the gradient of the loss func wrt the reflectivities and speed

    Parameters
    ----------
    phase_fac : array_like
        The phase factor of the imaging domain
    fd : array_like
        The frequency domain measured data
    fwd : array_like
        The frequency domain forward projection of the current image
        estimate
    freqs : array_like
        The frequencies used in the scan
    dv : float
        Area of an individual pixel
    int_f_xs : array_like m x N x N
        x_coords of front intersection
    int_f_ys : array_like m x N x N
        y_coords of front intersection
    int_b_xs : array_like m x N x N
        x_coords of back intersection
    int_b_ys : array_like m x N x N
        y_coords of back intersection
    speed : float
        Average propagation speed in  breast
    worker_pool : multiprocessing.pool.Pool
        Pool of workers for parallel computation

    Returns
    -------
    ref_derives : array_like
        The gradient of the loss function with respect to the
        reflectivities in the image domain
    """

    # Create func for parallel computation
    parallel_func = partial(_parallel_ref_deriv_speed, phase_fac, fwd, fd,
                            freqs, dv, int_f_xs, int_f_ys, int_b_xs, int_b_ys,
                            speed)

    iterable_idxs = range(np.size(fd, axis=0))  # Indices to iterate over

    # Store projections from parallel processing
    all_ref_derivs = np.array(worker_pool.map(parallel_func, iterable_idxs))

    # Reshape
    new_shape = [np.size(fd, axis=0),
                 np.size(phase_fac, axis=1) * np.size(phase_fac, axis=2) + 1]

    all_ref_derivs = np.reshape(all_ref_derivs, new_shape)

    all_ref_derivs = np.array([])

    for ff in range(np.size(fd, axis=0)):
        all_ref_derivs = np.concatenate((all_ref_derivs,
                                    _parallel_ref_deriv_speed(phase_fac, fwd,
                                                        fd, freqs, dv,
                                                        int_f_xs, int_f_ys,
                                                        int_b_xs, int_b_ys,
                                                        speed, ff)), axis=0)

    all_ref_derivs = np.reshape(all_ref_derivs, new_shape)

    # Sum over all frequencies
    ref_derivs = np.sum(all_ref_derivs, axis=0)

    # cut off the velocity derivative
    v_in_deriv = ref_derivs[np.size(ref_derivs) - 1]
    ref_derivs = np.delete(ref_derivs, np.size(ref_derivs) - 1)

    # reshape it to phase factor size
    ref_derivs = np.reshape(ref_derivs,
                            [np.size(phase_fac, axis=1),
                             np.size(phase_fac, axis=2)])

    ref_derivs *= -1  # Apply normalization factor
    # v_in_deriv *= -1

    return ref_derivs, v_in_deriv


def _parallel_ref_deriv_speed(phase_fac, fwd, fd, freqs, dv,
                              int_f_xs, int_f_ys, int_b_xs, int_b_ys,
                              speed, ff):
    """Parallelized function for ref_deriv calculation
    (both reflectivity and velocity gradient)

    Parameters
    ----------
    phase_fac : array_like
        The phase factor of the imaging domain
    fd : array_like
        The frequency domain measured data
    fwd : array_like
        The frequency domain forward projection of the current image
        estimate
    freqs : array_like
        The frequencies used in the scan
    ff : int
        The index for the frequency to be used

    Returns
    -------
    ref_deriv : array_like
        The gradient at this frequency
    """
    # calculate the derivatives wrt speed
    td_s_deriv_vel = td_velocity_deriv(int_f_xs, int_f_ys, int_b_xs, int_b_ys,
                                       speed)
    s_deriv_vel = s_velocity_deriv(phase_fac, td_s_deriv_vel, freqs[ff], dv)

    # Calculate the derivative wrt to the S-parameter
    s_deriv_refl = phase_fac ** (2 * freqs[ff])

    # number of antenna positions for flattening
    n_ant_pos = np.size(phase_fac, axis=0)

    # # temporary array with a column of speed derivative value
    # temp = np.zeros([n_ant_pos, 1])
    # temp[:, 0] = s_deriv_vel

    s_deriv = np.zeros([n_ant_pos,
                        np.size(phase_fac, axis=1)
                        * np.size(phase_fac, axis=2)], dtype=complex)

    for i in range(n_ant_pos):  # create a flattened array
        s_deriv[i] = s_deriv_refl[i].flatten()

    # making velocity array a 1-dimensional vector
    s_deriv_vel = np.reshape(s_deriv_vel, [np.size(s_deriv_vel), 1])

    # concatenate speed derivatives with reflectivity derivatives
    s_deriv = np.concatenate((s_deriv, s_deriv_vel), axis=1)

    # Calculate the derivative wrt the reflectivities
    ref_deriv = np.sum((np.conj(s_deriv)
                        * (fd[ff, :, None] - fwd[ff, :, None])
                        + s_deriv * np.conj(fd[ff, :, None]
                                            - fwd[ff, :, None])),
                       axis=0)

    ref_deriv = ref_deriv.flatten()

    return ref_deriv


def _parallel_ref_deriv(phase_fac, fwd, fd, freqs, ff):
    """Parallelized function for ref_deriv calculation

    Parameters
    ----------
    phase_fac : array_like
        The phase factor of the imaging domain
    fd : array_like
        The frequency domain measured data
    fwd : array_like
        The frequency domain forward projection of the current image
        estimate
    freqs : array_like
        The frequencies used in the scan
    ff : int
        The index for the frequency to be used

    Returns
    -------
    ref_deriv : array_like
        The gradient at this frequency
    """
    # Calculate the derivative wrt to the S-parameter

    s_deriv_refl = phase_fac ** (2 * freqs[ff])

    # Calculate the derivative wrt the reflectivities
    ref_deriv = np.sum((np.conj(s_deriv_refl)
                        * (fd[ff, :, None, None] - fwd[ff, :, None, None])
                        + s_deriv_refl * np.conj(fd[ff, :, None, None]
                                                 - fwd[ff, :, None, None])),
                       axis=0)

    return ref_deriv


def _parallel_ref_deriv_vel_freq(int_f_xs, int_f_ys, int_b_xs, int_b_ys,
                                 velocities, ant_rad, m_size, roi_rad,
                                 air_speed, adi_rad, mid_breast_max,
                                 mid_breast_min, fwd, fd, freqs, ff):
    """Parallelized function for ref_deriv calculation

    Parameters
    ----------
    int_f_xs : array_like m x N x N
        x_coords of front intersection
    int_f_ys : array_like m x N x N
        y_coords of front intersection
    int_b_xs : array_like m x N x N
        x_coords of back intersection
    int_b_ys : array_like m x N x N
        y_coords of back intersection
    velocities : array_like 1001 x 1
        Velocities calculated wrt frequency dependent permittivity
    ant_rad : float
        Antenna radius, after correcting for the antenna time delay
    m_size : int
        Size of image-space along one dimension
    roi_rad : float
        Radius of the region of interest that will define the spatial
        extent of the image-space
    air_speed : float
        The estimated propagation speed of the signal in air
    adi_rad : float
        Approximate radius of a phantom slice
    mid_breast_max : float
        major semi-axis of a phantom (b)
    mid_breast_min : float
        minor semi-axis of a phantom (a)
    fd : array_like
        The frequency domain measured data
    fwd : array_like
        The frequency domain forward projection of the current image
        estimate
    freqs : array_like
        The frequencies used in the scan
    ff : int
        The index for the frequency to be used

    Returns
    -------
    ref_deriv : array_like
        The gradient at this frequency
    """
    pix_ts, _, _, _, _ = get_pix_ts(ant_rad=ant_rad, m_size=m_size,
                            roi_rad=roi_rad, air_speed=air_speed,
                            breast_speed=velocities[ff], adi_rad=adi_rad,
                            mid_breast_max=mid_breast_max,
                            mid_breast_min=mid_breast_min, int_f_xs=int_f_xs,
                            int_f_ys=int_f_ys, int_b_xs=int_b_xs,
                            int_b_ys=int_b_ys)

    phase_fac = get_fd_phase_factor(pix_ts)

    # Calculate the derivative wrt to the S-parameter
    s_deriv_refl = phase_fac ** (2 * freqs[ff])

    # Calculate the derivative wrt the reflectivities
    ref_deriv = np.sum((np.conj(s_deriv_refl)
                        * (fd[ff, :, None, None] - fwd[ff, :, None, None])
                        + s_deriv_refl * np.conj(fd[ff, :, None, None]
                                                 - fwd[ff, :, None, None])),
                       axis=0)

    return ref_deriv


def s_velocity_deriv(phase_fac, speed_deriv, dv, ff):
    """Returns s_deriv array wrt
    to propagation speed

    Parameters
    ----------
    phase_fac : array_like MxNxN
        Phase factor
    speed_deriv : array_like MxNxN
        Values of time-delay derivatives wrt speed
    dv : float
        Area of every individual pixel
    ff : float
        Frequency

    Returns
    -------
    s_speed_deriv : array_like Mx1
        Value of s_deriv wrt to speed
    """
    # sum over all antenna positions and all pixels
    s_speed_deriv = -2 * np.pi * 1j * 2 * ff * dv * \
                    np.sum(phase_fac * speed_deriv, axis=(1, 2))
    return s_speed_deriv


def td_velocity_deriv(int_f_xs, int_f_ys, int_b_xs, int_b_ys, speed):
    """Returns a derivative of a time-delay function
    wrt propagation speed

    Parameters
    ----------
    int_f_xs : array_like m x N x N
        x_coords of front intersection
    int_f_ys : array_like m x N x N
        y_coords of front intersection
    int_b_xs : array_like m x N x N
        x_coords of back intersection
    int_b_ys : array_like m x N x N
        y_coords of back intersection
    speed : float
        Propagation speed in breast
    Returns
    -------
    speed_deriv : array_like m x N x N
        Derivatives array
    """

    # initialize an array of derivs
    speed_deriv = np.zeros([np.size(int_f_xs, axis=0),
                            np.size(int_f_xs, axis=1),
                            np.size(int_f_xs, axis=2)])

    # iterate over every antenna position
    for a_pos in range(np.size(int_f_xs, axis=0)):
        speed_deriv[a_pos] = - np.sqrt((int_b_xs[a_pos] - int_f_xs[a_pos]) ** 2
                                       +
                                       (int_b_ys[a_pos] - int_f_ys[a_pos]) ** 2)\
                                        / (speed ** 2)

    return speed_deriv
