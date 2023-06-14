"""
Tyson Reimer
University of Manitoba
June 4th, 2019
"""

import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from functools import partial

from umbms import null_logger

from umbms.loadsave import save_pickle

from umbms.beamform.time_delay import get_pix_ts
from umbms.beamform.utility import get_fd_phase_factor
from umbms.hardware.antenna import apply_ant_pix_delay
from umbms.beamform.fwdproj import fd_fwd_proj, fd_fwd_proj_vel_freq
from umbms.beamform.optimfuncs import get_ref_derivs, get_ref_derivs_speed, \
    get_ref_derivs_vel_freq

from umbms.plot import plt_sino, plt_fd_sino
from umbms.plot.imgplots import plot_fd_img


###############################################################################

def orr_recon(ini_img, freqs, m_size, fd, pos, tum_rad,
              velocities=None, phase_fac=None, breast_speed=0.0, ant_rad=0.0,
              air_speed=0.0, adi_rad=0.0, mid_breast_max=0.0,
              mid_breast_min=0.0, step_size=0.114,
              int_f_xs=None, int_f_ys=None, int_b_xs=None, int_b_ys=None,
              out_dir='', worker_pool=None, logger=null_logger):
    """Perform optimization-based radar reconstruction, via grad desc

    Parameters
    ----------
    ini_img : array_like
        Initial image estimate
    freqs : array_like
        The frequencies used in the scan
    m_size : int
        The number of pixels along one dimension of the reconstructed
        image
    fd : array_like
        The measured frequency domain data
    md : dict
        The metadata dictionary for this scan
    velocities : array_like 1001 x 1
        Velocities calculated wrt frequency dependent permittivity
    adi_rad : float
        Approximate radius of a phantom
    mid_breast_max : float
        major semi-axis of a phantom (b)
    mid_breast_min : float
        minor semi-axis of a phantom (a)
    int_f_xs : array_like m x N x N
        x_coords of front intersection
    int_f_ys : array_like m x N x N
        y_coords of front intersection
    int_b_xs : array_like m x N x N
        x_coords of back intersection
    int_b_ys : array_like m x N x N
        y_coords of back intersecti
    step_size : float
        The step size to use for gradient descent
    out_dir : str
        The output directory, where the figures and image estimates will
        be saved
    worker_pool : multiprocessing.pool.Pool
        Pool of workers for parallel computation
    logger :
        Logging object

    Returns
    -------
    img : array_like
        Reconstructed image
    """

    # Get tumour position and size in m
    tum_x = pos[0] / 100
    tum_y = pos[1] / 100
    tum_rad = tum_rad

    # Get the radius of the region of interest
    roi_rad = adi_rad + 0.01

    # Get the area of each individual pixel
    dv = ((2 * roi_rad) ** 2) / (m_size ** 2)

    # Plot the original data in the time and frequency domains
    plt_sino(fd=fd, title="Experimental Data",
             save_str='experimental_data.png',
             close=True, out_dir=out_dir, transparent=False)
    plt_fd_sino(fd=fd, title='Experimental Data',
                save_str='expt_data_fd.png',
                close=True, out_dir=out_dir, transparent=False)

    img = ini_img  # Initialize the image

    # Plot the initial image estimate
    plot_fd_img(np.real(img), tum_x=tum_x, tum_y=tum_y, tum_rad=tum_rad,
                adi_rad=adi_rad, mid_breast_max=mid_breast_max,
                mid_breast_min=mid_breast_min, roi_rad=roi_rad,
                img_rad=roi_rad,
                title="Image Estimate Step %d" % 0,
                save_str=os.path.join(out_dir,
                                      "imageEstimate_step_%d.png" % 0),
                save_fig=True, save_close=True, cbar_fmt='%.2e',
                transparent=False)

    cost_funcs = []  # Init list for storing cost function values

    # Forward project the current image estimate
    if not (velocities is None):
        fwd = fd_fwd_proj_vel_freq(model=img, int_f_xs=int_f_xs,
                                   int_f_ys=int_f_ys, int_b_xs=int_b_xs,
                                   int_b_ys=int_b_ys, velocities=velocities,
                                   ant_rad=ant_rad, m_size=m_size,
                                   roi_rad=roi_rad,
                                   air_speed=air_speed, dv=dv, adi_rad=adi_rad,
                                   mid_breast_max=mid_breast_max,
                                   mid_breast_min=mid_breast_min, freqs=freqs,
                                   worker_pool=worker_pool)
    else:

        # Forward project the current image estimate
        fwd = fd_fwd_proj(model=img, phase_fac=phase_fac, dv=dv, freqs=freqs,
                          worker_pool=worker_pool)

    img_estimates = []  # Init list for storing image estimates

    # Store the initial cost function value
    cost_funcs.append(float(np.sum(np.abs(fwd - fd) ** 2)))

    # Initialize the number of steps performed in gradient descent
    step = 0

    # Initialize the relative change in the cost function
    cost_rel_change = 1

    # new_speed = breast_speed
    # old_speed = breast_speed

    logger.info('\tInitial cost value:\t%.4f' % cost_funcs[0])

    # Perform gradient descent until the relative change in the cost
    # function is less than 0.1%
    while cost_rel_change > 0.001:

        logger.info('\tStep %d...' % (step + 1))

        # Plot the forward projection of the image estimate
        plt_sino(fd=fwd, title='Forward Projection Step %d' % (step + 1),
                 save_str='fwdProj_step_%d.png' % (step + 1),
                 close=True, out_dir=out_dir, transparent=False)
        plt_fd_sino(fd=fwd, title='Forward Projection Step %d' % (step + 1),
                    save_str='fwdProj_FD_step_%d.png' % (step + 1),
                    close=True, out_dir=out_dir, transparent=False)

        # Plot the diff between forward and expt data
        plt_sino(fd=(fd - fwd), title='Exp - Fwd Step %d' % (step + 1),
                 save_str='fwdExpDiff_step_%d.png' % (step + 1),
                 close=True, out_dir=out_dir, transparent=False)
        plt_fd_sino(fd=(fd - fwd), title='Exp - Fwd Step %d' % (step + 1),
                    save_str='fwdExpDiff_FD_step_%d.png' % (step + 1),
                    close=True, out_dir=out_dir, transparent=False)

        if not (velocities is None):
            ref_derivs = get_ref_derivs_vel_freq(int_f_xs=int_f_xs,
                                                 int_f_ys=int_f_ys,
                                                 int_b_xs=int_b_xs,
                                                 int_b_ys=int_b_ys,
                                                 ant_rad=ant_rad,
                                                 velocities=velocities,
                                                 m_size=m_size,
                                                 roi_rad=roi_rad,
                                                 air_speed=air_speed,
                                                 mid_breast_max=mid_breast_max,
                                                 mid_breast_min=mid_breast_min,
                                                 adi_rad=adi_rad, fd=fd,
                                                 fwd=fwd,
                                                 freqs=freqs,
                                                 worker_pool=worker_pool)
        else:
            # Calculate the gradient of the loss function wrt the
            # reflectivities in the object space
            ref_derivs = get_ref_derivs(phase_fac=phase_fac, fd=fd, fwd=fwd,
                                        freqs=freqs, worker_pool=worker_pool)

        # Update image estimate
        img -= step_size * np.real(ref_derivs)
        # print("Velocity gradient: ", np.real(v_in_deriv))
        # print("Old speed: ", old_speed, " m/s")
        #
        # new_speed -= alpha_v * np.real(v_in_deriv)
        #
        # print("New speed (alpha = %d):" % alpha_v, new_speed, " m/s")
        # old_speed = new_speed

        # Store the updated image estimate
        img_estimates.append(img * np.ones_like(img))

        # Plot the map of the loss function derivative with respect to
        # each reflectivity point
        plot_fd_img(np.real(ref_derivs), tum_x=tum_x, tum_y=tum_y,
                    tum_rad=tum_rad, adi_rad=adi_rad,
                    mid_breast_max=mid_breast_max,
                    mid_breast_min=mid_breast_min, roi_rad=roi_rad,
                    img_rad=roi_rad,
                    title="Full Deriv Step %d" % (step + 1),
                    save_str=os.path.join(out_dir,
                                          "fullDeriv_step_%d.png"
                                          % (step + 1)),
                    save_fig=True,
                    save_close=True,
                    cbar_fmt='%.2e',
                    transparent=False)

        # Plot the new image estimate
        plot_fd_img(np.real(img), tum_x=tum_x, tum_y=tum_y,
                    tum_rad=tum_rad, adi_rad=adi_rad,
                    mid_breast_max=mid_breast_max,
                    mid_breast_min=mid_breast_min, roi_rad=roi_rad,
                    img_rad=roi_rad,
                    title="Image Estimate Step %d" % (step + 1),
                    save_str=os.path.join(out_dir,
                                          "imageEstimate_step_%d.png"
                                          % (step + 1)),
                    save_fig=True,
                    save_close=True,
                    cbar_fmt='%.2e',
                    transparent=False)
        plot_fd_img(np.abs(img), tum_x=tum_x, tum_y=tum_y,
                    tum_rad=tum_rad, adi_rad=adi_rad,
                    mid_breast_max=mid_breast_max,
                    mid_breast_min=mid_breast_min, roi_rad=roi_rad,
                    img_rad=roi_rad,
                    title="Image Estimate Step %d" % (step + 1),
                    save_str=os.path.join(out_dir,
                                          "imageEstimate_step_%d_abs.png"
                                          % (step + 1)),
                    save_fig=True,
                    save_close=True,
                    cbar_fmt='%.2e',
                    transparent=False)
        if not (velocities is None):
            # Forward project the current image estimate
            fwd = fd_fwd_proj_vel_freq(model=img, int_f_xs=int_f_xs,
                                       int_f_ys=int_f_ys, int_b_xs=int_b_xs,
                                       int_b_ys=int_b_ys,
                                       velocities=velocities,
                                       ant_rad=ant_rad, m_size=m_size,
                                       roi_rad=roi_rad, air_speed=air_speed,
                                       dv=dv, adi_rad=adi_rad,
                                       mid_breast_max=mid_breast_max,
                                       mid_breast_min=mid_breast_min,
                                       freqs=freqs,
                                       worker_pool=worker_pool)
        else:
            # Forward project the current image estimate
            fwd = fd_fwd_proj(model=img, phase_fac=phase_fac, dv=dv,
                              freqs=freqs, worker_pool=worker_pool)

        # Normalize the forward projection
        cost_funcs.append(np.sum(np.abs(fwd - fd) ** 2))

        logger.info('\t\tCost func:\t%.4f' % (cost_funcs[step + 1]))

        # Calculate the relative change in the cost function
        cost_rel_change = ((cost_funcs[step] - cost_funcs[step + 1])
                           / cost_funcs[step])
        logger.info('\t\t\tCost Func ratio:\t%.4f%%'
                    % (100 * cost_rel_change))

        if step >= 1:  # For each step after the 0th

            # Plot the value of the cost function vs the number of
            # gradient descent steps performed
            plt.figure(figsize=(12, 6))
            plt.rc('font', family='Times New Roman')
            plt.tick_params(labelsize=18)
            plt.plot(np.arange(1, step + 2), cost_funcs[:step + 1],
                     'ko--')
            plt.xlabel('Iteration Number', fontsize=22)
            plt.ylabel('Cost Function Value', fontsize=22)
            plt.title("Optimization Performance with Gradient Descent",
                      fontsize=24)
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(out_dir,
                                     "costFuncs_step_%d.png" % (
                                             step + 1)),
                        transparent=False,
                        dpi=300)
            plt.close()

        step += 1  # Increment the step counter

    # After completing image reconstruction, plot the learning curve
    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=18)
    plt.plot(np.arange(1, len(cost_funcs) + 1), cost_funcs, 'ko--')
    plt.xlabel('Iteration Number', fontsize=22)
    plt.ylabel('Cost Function Value', fontsize=22)
    plt.title("Optimization Performance with Gradient Descent",
              fontsize=24)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(out_dir, "costFuncs.png"),
                transparent=True,
                dpi=300)
    plt.close()

    # Save the image estimates to a .pickle file
    save_pickle(img_estimates, os.path.join(out_dir, 'img_estimates.pickle'))

    return img


###############################################################################


def fd_das(fd_data, phase_fac, freqs, worker_pool, partial_ant_idx=None):
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


def fd_das_vel_freq(fd_data, *, int_f_xs, int_f_ys, int_b_xs, int_b_ys,
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
        parallel_func = partial(_parallel_fd_das_vel_freq_func, fd_data,
                                int_f_xs, int_f_ys, int_b_xs, int_b_ys,
                                velocities,
                                ant_rad, m_size, roi_rad, air_speed, freqs,
                                adi_rad, mid_breast_max, mid_breast_min)
    else:
        # Create func for parallel computation
        parallel_func = partial(_parallel_fd_das_vel_freq_func_part_ant,
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


def _parallel_fd_das_vel_freq_func(fd_data, int_f_xs, int_f_ys, int_b_xs,
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


def _parallel_fd_das_vel_freq_func_part_ant(fd_data, int_f_xs, int_f_ys,
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


###############################################################################


def fd_dmas(fd_data, pix_ts, freqs):
    """Compute frequency-domain DMAS reconstruction

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    pix_ts : array_like
        One-way response times for each pixel in the domain, for each
        antenna position
    freqs : array_like
        The frequencies used in the scan

    Returns
    -------
    img : array_like, KxK
        Reconstructed image, K pixels by K pixels
    """
    n_ant_pos = np.zise(fd_data, axis=1)

    # Init array for storing the individual back-projections, from
    # each antenna position
    back_projections = np.empty([n_ant_pos, np.size(pix_ts, axis=1),
                                 np.size(pix_ts, axis=2)], dtype=complex)

    # For each antenna position
    for aa in range(n_ant_pos):
        # Get the value to back-project
        back_proj_val = (fd_data[:, aa, None, None]
                         * np.exp(-2j * np.pi * freqs[:, None, None]
                                  * (-2 * pix_ts[aa, :, :])))
        # Sum over all frequencies
        back_proj_val = np.sum(back_proj_val, axis=0)

        # Store the back projection
        back_projections[aa, :, :] = back_proj_val

    # Init image to return
    img = np.empty([np.size(pix_ts, axis=1), np.size(pix_ts, axis=1)],
                   dtype=complex)

    # Loop over each antenna position
    for a_pos_frw in range(n_ant_pos):

        # For each other antenna position
        for a_pos_mult in range(a_pos_frw + 1, n_ant_pos):
            img += (back_projections[a_pos_frw, :, :] *
                    back_projections[a_pos_mult, :, :])

    return img
