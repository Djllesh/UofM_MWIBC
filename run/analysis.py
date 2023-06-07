"""
Illia Prykhodko
University of Manitoba
February 16th, 2023
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle

from umbms.beamform.extras import get_xy_arrs

from umbms.beamform.iqms import get_contrast_for_cyl
from umbms.analysis.acc_poserr import (do_pos_err_analysis, apply_syst_cor,
                                       plt_rand_pos_errs, plot_obs_vs_recon)
from umbms.analysis.acc_size import do_size_analysis, init_plt

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/cyl_phantom/')
__OUT_DIR = os.path.join(get_proj_path(), 'output/')
verify_path(__OUT_DIR)

__FD_NAME = 'cyl_phantom_immediate_reference_s11_rescan.pickle'
__MD_NAME = 'metadata_cyl_phantom_immediate_reference_rescan.pickle'

# The size of the reconstructed image along one dimension
__M_SIZE = 150


__PHANTOM_RAD = 0.0555

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


def make_imgs_arr(dir):
    """Creates a list of reflectivity arrays obtained from a
    reconstruction

    Parameters:
    -----------
    dir : string
        Directory of recon .pickle files

    Returns:
    ----------
    imgs : array_like
        Array of images

    """

    recons = os.listdir(dir)
    recons.sort()
    imgs = np.zeros([16, 150, 150], dtype=complex)

    for recon in recons:
        idx = int(recon.split('.')[0][2:]) - 1
        imgs[idx, :, :] = load_pickle(os.path.join(dir, recon))

    return imgs


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Load the frequency domain data and metadata

    fd_data, metadata = load_data()

    n_expts = np.size(fd_data, axis=0)  # The number of individual scans

    # Get the unique ID of each experiment / scan
    expt_ids = [md['id'] for md in metadata]

    # The output dir, where the reconstructions will be stored
    out_dir = os.path.join(__OUT_DIR, 'iqms-dec/')
    recon_dir = os.path.join(__OUT_DIR, 'cyl_phantom/recons/'
                                        'Immediate reference/Antenna time '
                                        'delay/')
    verify_path(out_dir)

    t_coords = np.zeros([16, 2])
    # Get metadata for plotting
    i = 0
    for md in metadata:
        if not np.isnan(md['tum_x']):
            t_coords[i, :] = [md['tum_x'], md['tum_y']]
            i += 1

    # Apply systematic corrections to the target positions in
    # positioning devise frame of reference to obtain antenna frame of
    # reference coordinates
    xs_antenna, ys_antenna = apply_syst_cor(xs=t_coords[:, 0],
                                            ys=t_coords[:, 1], x_err=-0.028,
                                            y_err=-0.027, phi_err=3.)
    t_coords[:, 0] = xs_antenna
    t_coords[:, 1] = ys_antenna

    adi_rad = __PHANTOM_RAD

    # Define the radius of the region of interest
    roi_rad = adi_rad + 0.01
    xs, _ = get_xy_arrs(__M_SIZE, roi_rad)
    dx = np.abs(xs[0, 1] - xs[0, 0])
    # Get the area of each pixel in the image domain
    dv = ((2 * roi_rad) ** 2) / (__M_SIZE ** 2)

    ####################################################################
    # 5 DIFFERENT RECONSTRUCTIONS
    ####################################################################

    # **************************************************************** #
    # 1. Homogeneous DAS (regular)
    # **************************************************************** #

    # Obtain all the reconstructions as an array
    # (requires base-one indexing)
    imgs_reg = make_imgs_arr(os.path.join(recon_dir, 'pickles/regular/'))

    # Initialize arrays to store the SCR and SMR values
    scr_values_reg = np.zeros([np.size(imgs_reg, axis=0), ])
    smr_values_reg = np.zeros([np.size(imgs_reg, axis=0), ])

    for jj in range(np.size(imgs_reg, 0)):  # For all images

        # Download the cubic spline and CoM parameters
        cs, x_cm, y_cm = load_pickle(os.path.join(recon_dir,
                                                  'spln_pars/id%d_pars.pickle'
                                                  % (jj + 1)))

        # Calculate SCR
        scr, _ = get_contrast_for_cyl(imgs_reg[jj], roi_rad=roi_rad,
                                      adi_rad=adi_rad, thickness=0.002,
                                      x_cm=x_cm, y_cm=y_cm, method='scr')
        # Calculate SMR
        smr, _ = get_contrast_for_cyl(imgs_reg[jj], roi_rad=roi_rad,
                                      adi_rad=adi_rad, thickness=0.002,
                                      x_cm=x_cm, y_cm=y_cm, method='smr')
        # Store it
        scr_values_reg[jj] = scr
        smr_values_reg[jj] = smr

    # Perform positioning error analysis
    c_xs_CoM_r, c_ys_CoM_r, xs_CoM_r, ys_CoM_r = do_pos_err_analysis(
                                                 imgs=imgs_reg,
                                                 tar_xs=t_coords[:, 0],
                                                 tar_ys=t_coords[:, 1],
                                                 roi_rad=roi_rad,
                                                 use_img_maxes=False,
                                                 make_plts=False,
                                                 logger=logger)

    # Find boundary coordinates to obtain images of the same size
    x_lim_lhs_regular = np.min([c_xs_CoM_r, xs_CoM_r, t_coords[:, 0]])
    x_lim_rhs_regular = np.max([c_xs_CoM_r, xs_CoM_r, t_coords[:, 0]])
    y_lim_bot_regular = np.min([c_ys_CoM_r, ys_CoM_r, t_coords[:, 1]])
    y_lim_top_regular = np.max([c_ys_CoM_r, ys_CoM_r, t_coords[:, 1]])

    # Call this func without plotting to print out the pos error values
    plt_rand_pos_errs(c_xs=c_xs_CoM_r, c_ys=c_ys_CoM_r,
                      xs=t_coords[:, 0], ys=t_coords[:, 1], logger=logger,
                      o_dir_str='regular_das/', save_str='', make_plts=False)

    # Initialize arrays for size analysis
    rho_sizes = np.zeros([np.size(imgs_reg, axis=0), ])
    phi_sizes = np.zeros([np.size(imgs_reg, axis=0), ])

    for jj in range(np.size(imgs_reg, axis=0)):  # For every image

        # Perform size analysis
        rho_size, phi_size = \
            do_size_analysis(img_here=np.abs(imgs_reg[jj, :, :]) ** 2, dx=dx,
                             roi_rad=roi_rad, make_plts=False)

        # Store sizes
        rho_sizes[jj] = rho_size
        phi_sizes[jj] = phi_size

    # Print out the results
    logger.info('Regular DAS:'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean rho-extent: %.4f'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean phi-extent: %.4f'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean SCR: %.4f'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean SMR: %.4f'
                % (np.mean(rho_sizes), np.mean(phi_sizes),
                   np.mean(scr_values_reg), np.mean(smr_values_reg)))

    # //////////////////////////////////////////////////////////////// #
    # Plotting part (rho and phi extents wrt polar radius of the target)
    # //////////////////////////////////////////////////////////////// #

    tar_rhos = np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2)
    colors = [
        [0, 0, 0],
        [100 / 255, 100 / 255, 100 / 255],
        [205 / 255, 205 / 255, 205 / 255],
    ]

    init_plt()
    plt.scatter(np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2),
                rho_sizes,
                color=colors[0])

    plt.title(r'$\mathdefault{\hat{\rho}}$-extent', fontsize=24)
    plt.xlabel(r'Target $\mathdefault{\rho}$ Position (cm)',
               fontsize=22)
    plt.ylabel('Target Response Extent (cm)', fontsize=22)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'size/paper/Antenna time delay/'
                                      'reg_das_rho_extent_vs_rho.png'),
                dpi=150,
                transparent=False)
    plt.close()

    init_plt()
    plt.scatter(np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2),
                phi_sizes,
                color=colors[0])
    plt.title(r'$\mathdefault{\hat{\varphi}}$-extent', fontsize=24)
    plt.xlabel(r'Target $\mathdefault{\rho}$ Position (cm)',
               fontsize=22)
    plt.ylabel('Target Response Extent (cm)', fontsize=22)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'size/paper/Antenna time delay/'
                                      'reg_das_phi_extent_vs_rho.png'),
                dpi=150,
                transparent=False)
    plt.close()

    # //////////////////////////////////////////////////////////////// #
    # End of plotting
    # //////////////////////////////////////////////////////////////// #

    # REPEAT FOR ANY OTHER TIME DELAY ESTIMATION TECHNIQUE

    # **************************************************************** #
    # 2. Binary DAS
    # **************************************************************** #

    imgs_b = make_imgs_arr(os.path.join(recon_dir, 'pickles/binary/'))

    scr_values_b = np.zeros([np.size(imgs_b, axis=0), ])
    smr_values_b = np.zeros([np.size(imgs_b, axis=0), ])

    for jj in range(np.size(imgs_b, 0)):
        cs, x_cm, y_cm = load_pickle(os.path.join(recon_dir,
                                                  'spln_pars/id%d_pars.pickle'
                                                  % (jj + 1)))

        scr, _ = get_contrast_for_cyl(imgs_b[jj], roi_rad=roi_rad,
                                      adi_rad=adi_rad, thickness=0.002,
                                      x_cm=x_cm, y_cm=y_cm, method='scr')

        smr, _ = get_contrast_for_cyl(imgs_b[jj], roi_rad=roi_rad,
                                      adi_rad=adi_rad, thickness=0.002,
                                      x_cm=x_cm, y_cm=y_cm, method='smr')

        scr_values_b[jj] = scr
        smr_values_b[jj] = smr

    c_xs_CoM_b, c_ys_CoM_b, xs_CoM_b, ys_CoM_b = do_pos_err_analysis(
                                                 imgs=imgs_b,
                                                 tar_xs=t_coords[:, 0],
                                                 tar_ys=t_coords[:, 1],
                                                 roi_rad=roi_rad,
                                                 use_img_maxes=False,
                                                 make_plts=False,
                                                 logger=logger)

    x_lim_lhs_binary = np.min([c_xs_CoM_b, xs_CoM_b, t_coords[:, 0]])
    x_lim_rhs_binary = np.max([c_xs_CoM_b, xs_CoM_b, t_coords[:, 0]])
    y_lim_bot_binary = np.min([c_ys_CoM_b, ys_CoM_b, t_coords[:, 1]])
    y_lim_top_binary = np.max([c_ys_CoM_b, ys_CoM_b, t_coords[:, 1]])

    plt_rand_pos_errs(c_xs=c_xs_CoM_b, c_ys=c_ys_CoM_b,
                      xs=t_coords[:, 0], ys=t_coords[:, 1], logger=logger,
                      o_dir_str='binary_das/', save_str='', make_plts=False)

    rho_sizes = np.zeros([np.size(imgs_b, axis=0), ])
    phi_sizes = np.zeros([np.size(imgs_b, axis=0), ])

    for jj in range(np.size(imgs_b, axis=0)):
        rho_size, phi_size = \
            do_size_analysis(img_here=np.abs(imgs_b[jj, :, :]) ** 2, dx=dx,
                             roi_rad=roi_rad, make_plts=False)
        rho_sizes[jj] = rho_size
        phi_sizes[jj] = phi_size

    logger.info('Binary DAS:'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean rho-extent: %.4f'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean phi-extent: %.4f'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean SCR: %.4f'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean SMR: %.4f'
                % (np.mean(rho_sizes), np.mean(phi_sizes),
                   np.mean(scr_values_b), np.mean(smr_values_b)))

    tar_rhos = np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2)
    colors = [
        [0, 0, 0],
        [100 / 255, 100 / 255, 100 / 255],
        [205 / 255, 205 / 255, 205 / 255],
    ]

    init_plt()
    plt.scatter(np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2),
                rho_sizes,
                color=colors[0])

    plt.title(r'$\mathdefault{\hat{\rho}}$-extent', fontsize=24)
    plt.xlabel(r'Target $\mathdefault{\rho}$ Position (cm)',
               fontsize=22)
    plt.ylabel('Target Response Extent (cm)', fontsize=22)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'size/paper/Antenna time delay/'
                                      'binary_das_rho_extent_vs_rho.png'),
                dpi=150,
                transparent=False)
    plt.close()

    init_plt()
    plt.scatter(np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2),
                phi_sizes,
                color=colors[0])
    plt.title(r'$\mathdefault{\hat{\varphi}}$-extent', fontsize=24)
    plt.xlabel(r'Target $\mathdefault{\rho}$ Position (cm)',
               fontsize=22)
    plt.ylabel('Target Response Extent (cm)', fontsize=22)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'size/paper/Antenna time delay/'
                                      'binary_das_phi_extent_vs_rho.png'),
                dpi=150,
                transparent=False)
    plt.close()

    # **************************************************************** #
    # 3. Frequency dependent DAS (zero conductivity)
    # **************************************************************** #

    imgs_fdnc = make_imgs_arr(os.path.join(recon_dir, 'pickles/fdnc/'))

    scr_values_fdnc = np.zeros([np.size(imgs_fdnc, axis=0), ])
    smr_values_fdnc = np.zeros([np.size(imgs_fdnc, axis=0), ])

    for jj in range(np.size(imgs_fdnc, 0)):
        cs, x_cm, y_cm = load_pickle(os.path.join(recon_dir,
                                                  'spln_pars/id%d_pars.pickle'
                                                  % (jj + 1)))

        scr, _ = get_contrast_for_cyl(imgs_fdnc[jj], roi_rad=roi_rad,
                                      adi_rad=adi_rad, thickness=0.002,
                                      x_cm=x_cm, y_cm=y_cm, method='scr')

        smr, _ = get_contrast_for_cyl(imgs_fdnc[jj], roi_rad=roi_rad,
                                      adi_rad=adi_rad, thickness=0.002,
                                      x_cm=x_cm, y_cm=y_cm, method='smr')

        scr_values_fdnc[jj] = scr
        smr_values_fdnc[jj] = smr

    c_xs_CoM_fdnc, c_ys_CoM_fdnc, xs_CoM_fdnc, ys_CoM_fdnc = \
        do_pos_err_analysis(
                            imgs=imgs_fdnc,
                            tar_xs=t_coords[:, 0],
                            tar_ys=t_coords[:, 1],
                            roi_rad=roi_rad,
                            use_img_maxes=False,
                            make_plts=False,
                            logger=logger)

    x_lim_lhs_fdnc = np.min([c_xs_CoM_fdnc, ys_CoM_fdnc, t_coords[:, 0]])
    x_lim_rhs_fdnc = np.max([c_xs_CoM_fdnc, ys_CoM_fdnc, t_coords[:, 0]])
    y_lim_bot_fdnc = np.min([c_ys_CoM_fdnc, ys_CoM_fdnc, t_coords[:, 1]])
    y_lim_top_fdnc = np.max([c_ys_CoM_fdnc, ys_CoM_fdnc, t_coords[:, 1]])

    plt_rand_pos_errs(c_xs=c_xs_CoM_fdnc, c_ys=c_ys_CoM_fdnc,
                      xs=t_coords[:, 0], ys=t_coords[:, 1], logger=logger,
                      o_dir_str='freq_dep_non_cond_das/', save_str='',
                      make_plts=False)

    rho_sizes = np.zeros([np.size(imgs_fdnc, axis=0), ])
    phi_sizes = np.zeros([np.size(imgs_fdnc, axis=0), ])

    for jj in range(np.size(imgs_fdnc, axis=0)):
        rho_size, phi_size = \
            do_size_analysis(img_here=np.abs(imgs_fdnc[jj, :, :]) ** 2, dx=dx,
                             roi_rad=roi_rad, make_plts=False)
        rho_sizes[jj] = rho_size
        phi_sizes[jj] = phi_size

    logger.info('FDNC DAS:'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean rho-extent: %.4f'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean phi-extent: %.4f'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean SCR: %.4f'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean SMR: %.4f'
                % (np.mean(rho_sizes), np.mean(phi_sizes),
                   np.mean(scr_values_fdnc), np.mean(smr_values_fdnc)))

    tar_rhos = np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2)
    colors = [
        [0, 0, 0],
        [100 / 255, 100 / 255, 100 / 255],
        [205 / 255, 205 / 255, 205 / 255],
    ]

    init_plt()
    plt.scatter(np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2),
                rho_sizes,
                color=colors[0])

    plt.title(r'$\mathdefault{\hat{\rho}}$-extent', fontsize=24)
    plt.xlabel(r'Target $\mathdefault{\rho}$ Position (cm)',
               fontsize=22)
    plt.ylabel('Target Response Extent (cm)', fontsize=22)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'size/paper/Antenna time delay/'
                                      'fdnc_das_rho_extent_vs_rho.png'),
                dpi=150,
                transparent=False)
    plt.close()

    init_plt()
    plt.scatter(np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2),
                phi_sizes,
                color=colors[0])
    plt.title(r'$\mathdefault{\hat{\varphi}}$-extent', fontsize=24)
    plt.xlabel(r'Target $\mathdefault{\rho}$ Position (cm)',
               fontsize=22)
    plt.ylabel('Target Response Extent (cm)', fontsize=22)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'size/paper/Antenna time delay/'
                                      'fdnc_das_phi_extent_vs_rho.png'),
                dpi=150,
                transparent=False)
    plt.close()

    # **************************************************************** #
    # 4. Frequency dependent DAS
    # **************************************************************** #

    imgs_fd = make_imgs_arr(os.path.join(recon_dir, 'pickles/fd/'))

    scr_values_fd = np.zeros([np.size(imgs_fd, axis=0), ])
    smr_values_fd = np.zeros([np.size(imgs_fd, axis=0), ])

    for jj in range(np.size(imgs_fd, 0)):
        cs, x_cm, y_cm = load_pickle(os.path.join(recon_dir,
                                                  'spln_pars/id%d_pars.pickle'
                                                  % (jj + 1)))

        scr, _ = get_contrast_for_cyl(imgs_fd[jj], roi_rad=roi_rad,
                                      adi_rad=adi_rad, thickness=0.002,
                                      x_cm=x_cm, y_cm=y_cm, method='scr')

        smr, _ = get_contrast_for_cyl(imgs_fd[jj], roi_rad=roi_rad,
                                      adi_rad=adi_rad, thickness=0.002,
                                      x_cm=x_cm, y_cm=y_cm, method='smr')

        scr_values_fd[jj] = scr
        smr_values_fd[jj] = smr

    c_xs_CoM_fd, c_ys_CoM_fd, xs_CoM_fd, ys_CoM_fd = do_pos_err_analysis(
                                                     imgs=imgs_fd,
                                                     tar_xs=t_coords[:, 0],
                                                     tar_ys=t_coords[:, 1],
                                                     roi_rad=roi_rad,
                                                     use_img_maxes=False,
                                                     make_plts=False,
                                                     logger=logger)

    x_lim_lhs_fd = np.min([c_xs_CoM_fd, xs_CoM_fd, t_coords[:, 0]])
    x_lim_rhs_fd = np.max([c_xs_CoM_fd, xs_CoM_fd, t_coords[:, 0]])
    y_lim_bot_fd = np.min([c_ys_CoM_fd, ys_CoM_fd, t_coords[:, 1]])
    y_lim_top_fd = np.max([c_ys_CoM_fd, ys_CoM_fd, t_coords[:, 1]])

    plt_rand_pos_errs(c_xs=c_xs_CoM_fd, c_ys=c_ys_CoM_fd,
                      xs=t_coords[:, 0], ys=t_coords[:, 1], logger=logger,
                      o_dir_str='freq_dep_das/', save_str='', make_plts=False)

    rho_sizes = np.zeros([np.size(imgs_fd, axis=0), ])
    phi_sizes = np.zeros([np.size(imgs_fd, axis=0), ])

    for jj in range(np.size(imgs_fd, axis=0)):
        rho_size, phi_size = \
            do_size_analysis(img_here=np.abs(imgs_fd[jj, :, :]) ** 2, dx=dx,
                             roi_rad=roi_rad, make_plts=False)
        rho_sizes[jj] = rho_size
        phi_sizes[jj] = phi_size

    logger.info('FD DAS:'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean rho-extent: %.4f'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean phi-extent: %.4f'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean SCR: %.4f'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean SMR: %.4f'
                % (np.mean(rho_sizes), np.mean(phi_sizes),
                   np.mean(scr_values_fd), np.mean(smr_values_fd)))

    tar_rhos = np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2)
    colors = [
        [0, 0, 0],
        [100 / 255, 100 / 255, 100 / 255],
        [205 / 255, 205 / 255, 205 / 255],
    ]

    init_plt()
    plt.scatter(np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2),
                rho_sizes,
                color=colors[0])

    plt.title(r'$\mathdefault{\hat{\rho}}$-extent', fontsize=24)
    plt.xlabel(r'Target $\mathdefault{\rho}$ Position (cm)',
               fontsize=22)
    plt.ylabel('Target Response Extent (cm)', fontsize=22)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'size/paper/Antenna time delay/'
                                      'fd_das_rho_extent_vs_rho.png'),
                dpi=150,
                transparent=False)
    plt.close()

    init_plt()
    plt.scatter(np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2),
                phi_sizes,
                color=colors[0])
    plt.title(r'$\mathdefault{\hat{\varphi}}$-extent', fontsize=24)
    plt.xlabel(r'Target $\mathdefault{\rho}$ Position (cm)',
               fontsize=22)
    plt.ylabel('Target Response Extent (cm)', fontsize=22)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'size/paper/Antenna time delay/'
                                      'fd_das_phi_extent_vs_rho.png'),
                dpi=150,
                transparent=False)
    plt.close()

    # **************************************************************** #
    # 5. DAS with ray-tracing
    # **************************************************************** #

    imgs_rt = make_imgs_arr(os.path.join(recon_dir, 'pickles/rt/'))

    scr_values_rt = np.zeros([np.size(imgs_rt, axis=0), ])
    smr_values_rt = np.zeros([np.size(imgs_rt, axis=0), ])

    for jj in range(np.size(imgs_rt, 0)):
        cs, x_cm, y_cm = load_pickle(os.path.join(recon_dir,
                                                  'spln_pars/id%d_pars.pickle'
                                                  % (jj + 1)))

        scr, _ = get_contrast_for_cyl(imgs_rt[jj], roi_rad=roi_rad,
                                      adi_rad=adi_rad, thickness=0.002,
                                      x_cm=x_cm, y_cm=y_cm, method='scr')

        smr, _ = get_contrast_for_cyl(imgs_rt[jj], roi_rad=roi_rad,
                                      adi_rad=adi_rad, thickness=0.002,
                                      x_cm=x_cm, y_cm=y_cm, method='smr')

        scr_values_rt[jj] = scr
        smr_values_rt[jj] = smr

    c_xs_CoM_rt, c_ys_CoM_rt, xs_CoM_rt, ys_CoM_rt = do_pos_err_analysis(
                                                     imgs=imgs_rt,
                                                     tar_xs=t_coords[:, 0],
                                                     tar_ys=t_coords[:, 1],
                                                     roi_rad=roi_rad,
                                                     use_img_maxes=False,
                                                     make_plts=False,
                                                     logger=logger)

    x_lim_lhs_rt = np.min([c_xs_CoM_rt, xs_CoM_rt, t_coords[:, 0]])
    x_lim_rhs_rt = np.max([c_xs_CoM_rt, xs_CoM_rt, t_coords[:, 0]])
    y_lim_bot_rt = np.min([c_ys_CoM_rt, ys_CoM_rt, t_coords[:, 1]])
    y_lim_top_rt = np.max([c_ys_CoM_rt, ys_CoM_rt, t_coords[:, 1]])

    plt_rand_pos_errs(c_xs=c_xs_CoM_rt, c_ys=c_ys_CoM_rt,
                      xs=t_coords[:, 0], ys=t_coords[:, 1], logger=logger,
                      o_dir_str='rt_das/', save_str='', make_plts=False)

    rho_sizes = np.zeros([np.size(imgs_rt, axis=0), ])
    phi_sizes = np.zeros([np.size(imgs_rt, axis=0), ])

    for jj in range(np.size(imgs_rt, axis=0)):
        rho_size, phi_size = \
            do_size_analysis(img_here=np.abs(imgs_rt[jj, :, :]) ** 2, dx=dx,
                             roi_rad=roi_rad, make_plts=False)
        rho_sizes[jj] = rho_size
        phi_sizes[jj] = phi_size

    logger.info('RT DAS:'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean rho-extent: %.4f '
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean phi-extent: %.4f '
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean SCR: %.4f'
                '\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tMean SMR: %.4f'
                % (np.mean(rho_sizes), np.mean(phi_sizes),
                   np.mean(scr_values_rt), np.mean(smr_values_rt)))

    # **************************************************************** #
    # **************************************************************** #

    tar_rhos = np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2)
    colors = [
        [0, 0, 0],
        [100 / 255, 100 / 255, 100 / 255],
        [205 / 255, 205 / 255, 205 / 255],
    ]

    init_plt()
    plt.scatter(np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2),
                rho_sizes,
                color=colors[0])

    plt.title(r'$\mathdefault{\hat{\rho}}$-extent', fontsize=24)
    plt.xlabel(r'Target $\mathdefault{\rho}$ Position (cm)',
               fontsize=22)
    plt.ylabel('Target Response Extent (cm)', fontsize=22)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'size/paper/Antenna time delay/'
                                      'rt_das_rho_extent_vs_rho.png'),
                dpi=150,
                transparent=False)
    plt.close()

    init_plt()
    plt.scatter(np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2),
                phi_sizes,
                color=colors[0])
    plt.title(r'$\mathdefault{\hat{\varphi}}$-extent', fontsize=24)
    plt.xlabel(r'Target $\mathdefault{\rho}$ Position (cm)',
               fontsize=22)
    plt.ylabel('Target Response Extent (cm)', fontsize=22)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'size/paper/Antenna time delay/'
                                      'rt_das_phi_extent_vs_rho.png'),
                dpi=150,
                transparent=False)
    plt.close()

    x_lim_lhs = np.min([x_lim_lhs_regular, x_lim_lhs_binary, x_lim_lhs_fdnc,
                        x_lim_lhs_fd, x_lim_lhs_rt]) - 0.15
    x_lim_rhs = np.max([x_lim_rhs_regular, x_lim_rhs_binary, x_lim_rhs_fdnc,
                        x_lim_rhs_fd, x_lim_rhs_rt]) + 0.15
    y_lim_bot = np.min([y_lim_bot_regular, y_lim_bot_binary, y_lim_bot_fdnc,
                        y_lim_bot_fd, y_lim_bot_rt]) - 0.15
    y_lim_top = np.max([y_lim_top_regular, y_lim_top_binary, y_lim_top_fdnc,
                        y_lim_top_fd, y_lim_top_rt]) + 0.15

    plot_obs_vs_recon(xs=t_coords[:, 0], ys=t_coords[:, 1],
                      compare_xs=c_xs_CoM_r, compare_ys=c_ys_CoM_r,
                      save_str='regular_das', o_dir_str='paper/Antenna time '
                                                        'delay/',
                      x_lim_lhs=x_lim_lhs, x_lim_rhs=x_lim_rhs,
                      y_lim_bot=y_lim_bot, y_lim_top=y_lim_top)

    plot_obs_vs_recon(xs=t_coords[:, 0], ys=t_coords[:, 1],
                      compare_xs=c_xs_CoM_b, compare_ys=c_ys_CoM_b,
                      save_str='binary_das', o_dir_str='paper/Antenna time '
                                                       'delay/',
                      x_lim_lhs=x_lim_lhs, x_lim_rhs=x_lim_rhs,
                      y_lim_bot=y_lim_bot, y_lim_top=y_lim_top)

    plot_obs_vs_recon(xs=t_coords[:, 0], ys=t_coords[:, 1],
                      compare_xs=c_xs_CoM_fdnc, compare_ys=c_ys_CoM_fdnc,
                      save_str='fdnc_das', o_dir_str='paper/Antenna time '
                                                     'delay/',
                      x_lim_lhs=x_lim_lhs, x_lim_rhs=x_lim_rhs,
                      y_lim_bot=y_lim_bot, y_lim_top=y_lim_top)

    plot_obs_vs_recon(xs=t_coords[:, 0], ys=t_coords[:, 1],
                      compare_xs=c_xs_CoM_fd, compare_ys=c_ys_CoM_fd,
                      save_str='fd_das', o_dir_str='paper/Antenna time delay/',
                      x_lim_lhs=x_lim_lhs, x_lim_rhs=x_lim_rhs,
                      y_lim_bot=y_lim_bot, y_lim_top=y_lim_top)

    plot_obs_vs_recon(xs=t_coords[:, 0], ys=t_coords[:, 1],
                      compare_xs=c_xs_CoM_rt, compare_ys=c_ys_CoM_rt,
                      save_str='rt_das', o_dir_str='paper/Antenna time delay/',
                      x_lim_lhs=x_lim_lhs, x_lim_rhs=x_lim_rhs,
                      y_lim_bot=y_lim_bot, y_lim_top=y_lim_top)
