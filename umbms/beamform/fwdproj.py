"""
Tyson Reimer
University of Manitoba
June 3rd, 2019
"""

import numpy as np
import multiprocessing as mp
from functools import partial
from umbms.beamform.time_delay import get_pix_ts
from umbms.beamform.utility import get_fd_phase_factor


###############################################################################


def fd_fwd_proj(model, phase_fac, dv, freqs, worker_pool):
    """Forward project in the frequency domain

    Parameters
    ----------
    model : array_like, NxN
        Image-space model to be forward projected. N is number of pixels
        along one dimension.
    phase_fac : array_like, MxNxN
        Phase factor for efficient computation. N is number of pixels
        along one dimension of model, M is number of antenna positions
        used in the scan.
    dv : float
        Volume element, units m^3 (also area element, units m^2)
    freqs : array_like, Fx1
        The frequency vector used in the scan, in Hz, for F frequencies
    worker_pool : multiprocessing.pool.Pool
        Pool of workers for parallel computation

    Returns
    -------
    fwd : array_like, LxM
        Forward projection of primary scatter responses only. L is the
        number of frequencies in the scan, M is the number of antenna
        positions.
    """

    n_fs = np.size(freqs)

    # Create function for parallel processing
    parallel_func = partial(_parallel_fd_fwd_proj, freqs,
                            phase_fac, model, dv)

    iterable_idxs = range(n_fs)  # Get indices to iterate over

    # Collect forward projections
    fwds = np.array(worker_pool.map(parallel_func, iterable_idxs))

    fwds = np.reshape(fwds, [n_fs, 72])

    return fwds


def fd_fwd_proj_vel_freq(model, int_f_xs, int_f_ys, int_b_xs, int_b_ys,
                         velocities, ant_rad, m_size, roi_rad, air_speed, dv,
                         freqs, *, adi_rad=0, mid_breast_max=0.0,
                         mid_breast_min=0.0, worker_pool=None):
    """Forward project in the frequency domain

    Parameters
    ----------
    model : array_like, NxN
        Image-space model to be forward projected. N is number of pixels
        along one dimension.
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
    dv : float
        Volume element, units m^3 (also area element, units m^2)
    freqs : array_like, Fx1
        The frequency vector used in the scan, in Hz, for F frequencies
    worker_pool : multiprocessing.pool.Pool
        Pool of workers for parallel computation

    Returns
    -------
    fwd : array_like, LxM
        Forward projection of primary scatter responses only. L is the
        number of frequencies in the scan, M is the number of antenna
        positions.
    """

    n_fs = np.size(freqs)

    # Create function for parallel processing
    parallel_func = partial(_parallel_fd_fwd_proj_vel_freq, freqs,
                            int_f_xs, int_f_ys, int_b_xs, int_b_ys, velocities,
                            ant_rad, m_size, roi_rad, air_speed, adi_rad,
                            mid_breast_max, mid_breast_min, model, dv)

    iterable_idxs = range(n_fs)  # Get indices to iterate over

    # Collect forward projections
    fwds = np.array(worker_pool.map(parallel_func, iterable_idxs))

    fwds = np.reshape(fwds, [n_fs, 72])

    return fwds


def _parallel_fd_fwd_proj(freqs, phase_fac, model, dv, ff):
    """Parallel processing function to compute projection at freq ff

    Parameters
    ----------
    freqs : array_like, Lx1
        Frequency vector, L is the number of frequencies
    phase_fac : array_like, MxNxN
        Phase factor for efficient computation. N is number of pixels
        along one dimension of model, M is number of antenna positions
        used in the scan.
    model : array_like, NxN
        Image-space model to be forward projected. N is number of pixels
        along one dimension.
    ff : int
        Frequency index

    Returns
    -------
    p_resp : array_like, LxM
        Forward projection at frequency ff
    """

    temp_var = phase_fac ** freqs[ff]

    temp_var2 = model[None, :, :] * temp_var

    # Compute the primary scatter responses
    p_resp = np.sum(temp_var2 * temp_var, axis=(1, 2)) * dv

    return p_resp


def _parallel_fd_fwd_proj_vel_freq(freqs, int_f_xs, int_f_ys, int_b_xs,
                                   int_b_ys, velocities, ant_rad, m_size,
                                   roi_rad, air_speed, adi_rad, mid_breast_max,
                                   mid_breast_min, model, dv, ff):
    """Parallel processing function to compute projection at freq ff

    Parameters
    ----------
    freqs : array_like, Lx1
        Frequency vector, L is the number of frequencies
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
    model : array_like, NxN
        Image-space model to be forward projected. N is number of pixels
        along one dimension.
    ff : int
        Frequency index

    Returns
    -------
    p_resp : array_like, LxM
        Forward projection at frequency ff
    """

    pix_ts, _, _, _, _ = get_pix_ts(ant_rad=ant_rad, m_size=m_size,
                                roi_rad=roi_rad, air_speed=air_speed,
                                breast_speed=velocities[ff], adi_rad=adi_rad,
                                mid_breast_max=mid_breast_max,
                                mid_breast_min=mid_breast_min,
                                int_f_xs=int_f_xs, int_f_ys=int_f_ys,
                                int_b_xs=int_b_xs, int_b_ys=int_b_ys)

    phase_fac = get_fd_phase_factor(pix_ts)

    temp_var = phase_fac ** freqs[ff]

    temp_var2 = model[None, :, :] * temp_var

    # Compute the primary scatter responses
    p_resp = np.sum(temp_var2 * temp_var, axis=(1, 2)) * dv

    return p_resp
