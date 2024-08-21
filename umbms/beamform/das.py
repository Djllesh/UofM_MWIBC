from functools import partial

import numpy as np

from umbms.beamform.time_delay import get_pix_ts
from umbms.beamform.utility import get_fd_phase_factor
from umbms.hardware.antenna import apply_ant_pix_delay


def fd_das(fd_data, phase_fac, freqs, worker_pool=None, partial_ant_idx=None):
    """Compute frequency-domain DAS reconstruction

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension
    freqs : array_like, Nx1
        The frequencies used in the scan
    worker_pool : multiprocessing.pool.Pool
        Pool of workers for parallel computation
    partial_ant_idx : array_like 1 x n_ant_pos
        Binary mask of indices of antenna positions that are being used
        (if reconstructing specific antenna positions)

    Returns
    -------
    img : array_like, KxK
        Reconstructed image, K pixels by K pixels
    """

    n_fs = np.size(freqs)  # Find number of frequencies used

    if worker_pool is None:  # If *not* doing parallel computation

        # Reconstruct the image
        img = np.sum(
            fd_data[:, :, None, None]
            * np.power(phase_fac[None, :, :, :],
                       -2 * freqs[:, None, None, None]),
            axis=(0,1))

    else:
        # Correct for to/from propagation
        new_phase_fac = phase_fac ** (-2)

        if partial_ant_idx is None:
            # Create func for parallel computation
            parallel_func = partial(_parallel_fd_das_func, fd_data, new_phase_fac,
                                    freqs)
        else:
            parallel_func = partial(_parallel_fd_das_func_part_ant, fd_data,
                                    new_phase_fac, partial_ant_idx, freqs)

        iterable_idxs = range(n_fs)  # Indices to iterate over

        # Store projections from parallel processing
        back_projections = np.array(worker_pool.map(parallel_func, iterable_idxs))

        # Reshape
        back_projections = np.reshape(back_projections,
                                      [n_fs, np.size(phase_fac, axis=1),
                                       np.size(phase_fac, axis=2)])

        # Sum over all frequencies
        img = np.sum(back_projections, axis=0)

    return img


def _parallel_fd_das_func(fd_data, new_phase_fac, freqs, ff):
    """Compute projection for given frequency ff

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    new_phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension, corrected for DAS
    ff : int
        Frequency index

    Returns
    -------
    this_projection : array_like, KxK
        Back-projection of this particular frequency-point
    """

    # Get phase factor for this frequency
    # this_phase_fac = new_phase_fac * np.exp(-1j * np.pi * freqs[ff])
    this_phase_fac = new_phase_fac ** freqs[ff]

    # Sum over antenna positions
    this_projection = np.sum(this_phase_fac * fd_data[ff, :, None, None],
                             axis=0)

    return this_projection


def _parallel_fd_das_func_part_ant(fd_data, new_phase_fac, partial_ant_idx,
                                   freqs, ff):
    """Compute projection for given frequency ff and for given antenna
    indices

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    new_phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension, corrected for DAS
    partial_ant_idx : array_like 1 x n_ant_pos
        Binary mask of indices of antenna positions that are being used
        (if reconstructing specific antenna positions)
    ff : int
        Frequency index

    Returns
    -------
    this_projection : array_like, KxK
        Back-projection of this particular frequency-point
    """

    # Get phase factor for this frequency
    this_phase_fac = new_phase_fac ** freqs[ff]

    # Sum over antenna positions
    this_projection = np.sum(this_phase_fac * fd_data[ff, partial_ant_idx,
                                                      None, None], axis=0)

    return this_projection


def fd_das_freq_dep(fd_data, *, int_f_xs, int_f_ys, int_b_xs, int_b_ys,
                    velocities, ant_rad, m_size, roi_rad, air_speed, freqs,
                    worker_pool, adi_rad=0, mid_breast_max=0.0,
                    mid_breast_min=0.0, partial_ant_idx=None):
    """Compute frequency dependent DAS reconstruction

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
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
    freqs : array_like, Nx1
        The frequencies used in the scan
    worker_pool : multiprocessing.pool.Pool
        Pool of workers for parallel computation
    partial_ant_idx : array_like 1 x n_ant_pos
        Binary mask of indices of antenna positions that are being used
        (if reconstructing specific antenna positions)

    Returns
    -------
    img : array_like, KxK
        Reconstructed image, K pixels by K pixels
    """

    n_fs = np.size(freqs)  # Find number of frequencies used
    if partial_ant_idx is None:
        # Create func for parallel computation
        parallel_func = partial(_parallel_fd_das_freq_dep_func, fd_data,
                                int_f_xs, int_f_ys, int_b_xs, int_b_ys,
                                velocities,
                                ant_rad, m_size, roi_rad, air_speed, freqs,
                                adi_rad, mid_breast_max, mid_breast_min)
    else:
        # Create func for parallel computation
        parallel_func = partial(_parallel_fd_das_freq_dep_func_part_ant,
                                fd_data, int_f_xs, int_f_ys, int_b_xs,
                                int_b_ys, velocities, ant_rad, m_size, roi_rad,
                                air_speed, freqs, adi_rad, mid_breast_max,
                                mid_breast_min, partial_ant_idx)

    iterable_idxs = range(n_fs)  # Indices to iterate over

    # Store projections from parallel processing
    back_projections = np.array(worker_pool.map(parallel_func, iterable_idxs))

    # Reshape
    back_projections = np.reshape(back_projections, [n_fs, m_size, m_size])

    # Sum over all frequencies
    img = np.sum(back_projections, axis=0)

    return img


def _parallel_fd_das_freq_dep_func(fd_data, int_f_xs, int_f_ys, int_b_xs,
                                   int_b_ys, velocities, ant_rad, m_size,
                                   roi_rad, air_speed, freqs, adi_rad,
                                   mid_breast_max, mid_breast_min, ff):
    """Compute projection for given frequency ff

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
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
    ff : int
        Frequency index

    Returns
    -------
    this_projection : array_like, KxK
        Back-projection of this particular frequency-point
    """

    # on each parallel iteration (ff) recalculate the time-delay array
    # using different velocity: velocities[ff]
    pix_ts, _, _, _, _ = get_pix_ts(ant_rad=ant_rad, m_size=m_size,
                                    roi_rad=roi_rad, air_speed=air_speed,
                                    breast_speed=velocities[ff],
                                    adi_rad=adi_rad,
                                    mid_breast_max=mid_breast_max,
                                    mid_breast_min=mid_breast_min,
                                    int_f_xs=int_f_xs, int_f_ys=int_f_ys,
                                    int_b_xs=int_b_xs, int_b_ys=int_b_ys)

    pix_ts = apply_ant_pix_delay(pix_ts=pix_ts)

    phase_fac = get_fd_phase_factor(pix_ts)

    # Get phase factor for this frequency
    this_phase_fac = phase_fac ** (freqs[ff] * (-2))

    # Sum over antenna positions
    this_projection = np.sum(this_phase_fac * fd_data[ff, :, None, None],
                             axis=0)

    return this_projection


def _parallel_fd_das_freq_dep_func_part_ant(fd_data, int_f_xs, int_f_ys,
                                            int_b_xs, int_b_ys, velocities,
                                            ant_rad, m_size, roi_rad,
                                            air_speed, freqs, adi_rad,
                                            mid_breast_max, mid_breast_min,
                                            partial_ant_idx,
                                            ff):
    """Compute projection for given frequency ff and for given antenna
    indices

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
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
    ff : int
        Frequency index

    Returns
    -------
    this_projection : array_like, KxK
        Back-projection of this particular frequency-point
    """

    # on each parallel iteration (ff) recalculate the time-delay array
    # using different velocity: velocities[ff]
    pix_ts, _, _, _, _ = get_pix_ts(ant_rad=ant_rad, m_size=m_size,
                                    roi_rad=roi_rad, air_speed=air_speed,
                                    breast_speed=velocities[ff],
                                    adi_rad=adi_rad,
                                    mid_breast_max=mid_breast_max,
                                    mid_breast_min=mid_breast_min,
                                    int_f_xs=int_f_xs, int_f_ys=int_f_ys,
                                    int_b_xs=int_b_xs, int_b_ys=int_b_ys,
                                    partial_ant_idx=partial_ant_idx)

    phase_fac = get_fd_phase_factor(pix_ts)

    # Get phase factor for this frequency
    this_phase_fac = phase_fac ** (freqs[ff] * (-2))

    # Sum over antenna positions
    this_projection = np.sum(this_phase_fac * fd_data[ff, partial_ant_idx,
                                                      None, None], axis=0)

    return this_projection
