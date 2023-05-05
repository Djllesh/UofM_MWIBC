"""
Illia Prykhodko
University of Manitoba
October 11th, 2022
"""

import os
import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from time import perf_counter
import multiprocessing as mp

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle

from umbms.plot.imgplots import plot_fd_img, \
    plot_fd_img_with_intersections
from umbms.plot import plt_sino, plt_fd_sino

from umbms.beamform.recon import fd_das, fd_dmas, orr_recon, \
    fd_das_vel_freq
from umbms.beamform.extras import (apply_ant_t_delay, get_pix_ts,
                                   get_fd_phase_factor,
                                   find_xy_ant_bound_circle,
                                   find_xy_ant_bound_ellipse,
                                   get_ant_scan_xys, get_xy_arrs,
                                   get_pix_ts_old, get_pix_dists_angs,
                                   get_circle_intersections_parallel)

from umbms.beamform.boundary_detection import (find_boundary, polar_fit_cs,
                                               find_centre_of_mass,
                                               get_binary_mask,
                                               get_boundary_iczt,
                                               cart_to_polar, make_speed_map)

from umbms.beamform.propspeed import (estimate_speed, get_breast_speed,
                                      get_speed_from_perm,
                                      get_breast_speed_freq)
from umbms.beamform.raytrace import find_boundary_rt
from umbms.beamform.optimfuncs import td_velocity_deriv
from umbms.beamform.sigproc import iczt
from umbms.beamform.iqms import get_contrast_for_cyl
from umbms.beamform.acc_poserr import do_pos_err_analysis, plt_rand_pos_errs
from umbms.beamform.acc_size import do_size_analysis, init_plt

###############################################################################

__CPU_COUNT = mp.cpu_count()
__PRECISION_SCALING_FACTOR = 1

assert isinstance(__PRECISION_SCALING_FACTOR, int), \
    f"Scaling factor is not int, got: {__PRECISION_SCALING_FACTOR}"

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/cyl_phantom/')
__OUT_DIR = os.path.join(get_proj_path(), 'output/')
verify_path(__OUT_DIR)
__FITTED_DATA_DIR = os.path.join(get_proj_path(), 'data/freq_data/')

__FD_NAME = 'cyl_phantom_immediate_reference_s11_rescan.pickle'
__MD_NAME = 'metadata_cyl_phantom_immediate_reference_rescan.pickle'
__FITTED_NAME = 'Dielectric Measurements Glycerin.csv'

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
__PHANTOM_RAD = 0.0555
__GLASS_THIKNESS = 0.003
__SPHERE_RAD = 0.0075
__ROD_RAD = 0.002
__ANT_RAD = 0.21

__SPHERE_POS = [(np.nan, np.nan),
                (0.0, 4.0),
                (0.0, 3.0),
                (0.0, 2.0),
                (0.0, 1.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (2.0, 0.0),
                (3.0, 0.0),
                (4.0, 0.0),
                (np.nan, np.nan),
                (4.3, 0.0),
                (3.3, 0.0),
                (2.3, 0.0),
                (1.3, 0.0),
                (0.3, 0.0),
                (0.0, 1.3),
                (0.0, 2.3),
                (0.0, 3.3),
                (0.0, 4.3),
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

__VAC_SPEED = 3e8  # Define propagation speed in vacuum


def load_data():
    """Loads both fd_data and metadata

    Returns:
    --------
    tuple of two loaded variables
    """
    return load_pickle(os.path.join(__DATA_DIR,
                                    __FD_NAME)), \
           load_pickle(os.path.join(__DATA_DIR,
                                    __MD_NAME))


def get_middle_td(pix_ts):
    size = np.size(pix_ts, axis=0)
    idx = np.size(pix_ts, axis=1) // 2 - 1
    output = np.zeros(size)
    for i in range(size):
        output[i] = pix_ts[i, idx, idx]
    return output


def make_imgs_arr(dir):
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
                                        'Immediate reference/Gen 2/')
    verify_path(out_dir)

    t_coords = np.zeros([16, 2])
    # Get metadata for plotting
    i = 0
    for md in metadata:
        if not np.isnan(md['tum_x']):
            t_coords[i, :] = [md['tum_x'], md['tum_y']]
            i += 1

    tar_md = metadata[1]

    scan_rad = tar_md['ant_rad'] / 100
    tum_x = tar_md['tum_x'] / 100
    tum_y = tar_md['tum_y'] / 100
    tum_rad = 0.5 * (tar_md['tum_diam'] / 100)
    adi_rad = __PHANTOM_RAD

    # Correct for how the scan radius is measured (from a
    # point on the antenna stand, not at the SMA connection
    # point)
    scan_rad += 0.03618

    # Define the radius of the region of interest
    roi_rad = adi_rad + 0.01
    xs, _ = get_xy_arrs(__M_SIZE, roi_rad)
    dx = np.abs(xs[0, 1] - xs[0, 0])
    # Get the area of each pixel in the image domain
    dv = ((2 * roi_rad) ** 2) / (__M_SIZE ** 2)

    # Correct for the antenna time delay
    # NOTE: Only the new antenna was used in UM-BMID Gen-3
    ant_rad = apply_ant_t_delay(scan_rad=scan_rad, new_ant=True)

    ############################################################
    # 5 DIFFERENT RECONSTRUCTIONS

    # 1. Homogeneous DAS (regular)

    imgs = make_imgs_arr(os.path.join(recon_dir, 'freq_dep-das/'))

    c_xs_CoM, c_ys_CoM, xs_CoM, ys_CoM = do_pos_err_analysis(imgs=imgs,
                                         tar_xs=t_coords[:, 0],
                                         tar_ys=t_coords[:, 1],
                                         roi_rad=roi_rad,
                                         use_img_maxes=False,
                                         make_plts=False, logger=logger)

    scr_values = np.zeros([np.size(imgs, axis=0), ])

    for jj in range(np.size(imgs, 0)):

        cs, x_cm, y_cm = load_pickle(os.path.join(recon_dir,
                                                  'spln_pars/id%d_pars.pickle'
                                                  % (jj + 1)))

        scr, _ = get_contrast_for_cyl(imgs[jj], roi_rad=roi_rad,
                                      adi_rad=adi_rad, thickness=0.002,
                                      x_cm=x_cm, y_cm=y_cm, method='mmr')

        scr_values[jj] = scr

    init_plt()
    tar_rhos = np.sqrt(c_xs_CoM ** 2 + c_ys_CoM ** 2)
    plt.scatter(tar_rhos, scr_values, color=[0, 0, 0])
    plt.xlabel(r'Target $\mathdefault{\rho}$ Position (cm)', fontsize=22)
    plt.ylabel('Mean-to-mean ratio (dB)', fontsize=22)
    plt.savefig(os.path.join(out_dir, 'contrast/freq_dep_das_mmr.png'),
                dpi=150)

    # rho_sizes = np.zeros([np.size(imgs, axis=0), ])
    # phi_sizes = np.zeros([np.size(imgs, axis=0), ])
    #
    # for jj in range(np.size(imgs, axis=0)):
    #
    #     rho_size, phi_size = \
    #         do_size_analysis(img_here=np.abs(imgs[jj, :, :]) ** 2, dx=dx,
    #                          roi_rad=roi_rad,
    #                          save_dir=os.path.join(out_dir,
    #                          'rt_das/'),
    #                          save_str='rt_das_expt_%d.png'
    #                                   % (jj + 1),
    #                          make_plts=True)
    #     rho_sizes[jj] = rho_size
    #     phi_sizes[jj] = phi_size
    #
    # tar_rhos = np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2)
    # colors = [
    #     [0, 0, 0],
    #     [100 / 255, 100 / 255, 100 / 255],
    #     [205 / 255, 205 / 255, 205 / 255],
    # ]
    #
    # init_plt()
    # plt.scatter(np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2),
    #             rho_sizes,
    #             color=colors[0],
    #             label=r'$\mathdefault{\hat{\rho}}$-extent')
    # plt.scatter(np.sqrt(t_coords[:, 0] ** 2 + t_coords[:, 1] ** 2),
    #             phi_sizes,
    #             color=colors[2],
    #             label=r'$\mathdefault{\hat{\phi}}$-extent')
    # plt.xlabel(r'Target $\mathdefault{\rho}$ Position (cm)',
    #            fontsize=22)
    # plt.ylabel('Target Response Extent (cm)', fontsize=22)
    # plt.legend(fontsize=18)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, 'rt_das/'
    #                                   'rt_das_extent_vs_rho.png'),
    #             dpi=150,
    #             transparent=False)

    #
    # c_xs_max, c_ys_max, xs_max, ys_max = do_pos_err_analysis(imgs=imgs,
    #                                          tar_xs=t_coords[:, 0],
    #                                          tar_ys=t_coords[:, 1],
    #                                          roi_rad=roi_rad,
    #                                          o_dir_str='regular_das/',
    #                                          save_str='%s' %
    #                                             'regular_das_max_4',
    #                                          use_img_maxes=True,
    #                                          make_plts=True,
    #                                          logger=logger)
    #
    # plt_rand_pos_errs(c_xs=c_xs_max, c_ys=c_ys_max,
    #                   xs=t_coords[:, 0], ys=t_coords[:, 1],
    #                   o_dir_str='regular_das/',
    #                   save_str='%s' %
    #                            'regular_das_m_4',
    #                   make_plts=True, logger=logger)
    #
    # c_xs_CoM, c_ys_CoM, xs_CoM, ys_CoM = do_pos_err_analysis(imgs=imgs,
    #                                     tar_xs=t_coords[:, 0],
    #                                     tar_ys=t_coords[:, 1],
    #                                     roi_rad=roi_rad,
    #                                     o_dir_str='regular_das/',
    #                                     save_str='%s' %
    #                                              'regular_das_com_4',
    #                                     use_img_maxes=False,
    #                                     make_plts=True, logger=logger)
    #
    # plt_rand_pos_errs(c_xs=c_xs_CoM, c_ys=c_ys_CoM,
    #                   xs=t_coords[:, 0], ys=t_coords[:, 1],
    #                   o_dir_str='regular_das/',
    #                   save_str='%s' %
    #                            'regular_das_com_4',
    #                   make_plts=True, logger=logger)
    #
    # for (img, idx) in zip(imgs, range(np.size(t_coords, axis=0))):
    #
    #     plot_fd_img(img=np.abs(img), loc_err=True,
    #                 tum_x=t_coords[idx, 0]/100,
    #                 tum_y=t_coords[idx, 1]/100,
    #                 tum_x_uncor=xs_max[idx]/100,
    #                 tum_y_uncor=ys_max[idx]/100,
    #                 tum_x_cor=c_xs_max[idx]/100,
    #                 tum_y_cor=c_ys_max[idx]/100, tum_rad=tum_rad,
    #                 img_rad=roi_rad, adi_rad=adi_rad,
    #                 ant_rad=ant_rad, roi_rad=roi_rad, save_fig=True,
    #                 save_close=True,
    #                 save_str=os.path.join(out_dir, 'regular_das/'
    #                                                'img%d_max_4.png'
    #                                                 % idx))
    #
    #     plot_fd_img(img=np.abs(img), loc_err=True,
    #                 tum_x=t_coords[idx, 0]/100,
    #                 tum_y=t_coords[idx, 1]/100,
    #                 tum_x_uncor=xs_CoM[idx]/100,
    #                 tum_y_uncor=ys_CoM[idx]/100,
    #                 tum_x_cor=c_xs_CoM[idx]/100,
    #                 tum_y_cor=c_ys_CoM[idx]/100, tum_rad=tum_rad,
    #                 adi_rad=adi_rad, ant_rad=ant_rad,
    #                 roi_rad=roi_rad, save_fig=True, img_rad=roi_rad,
    #                 save_str=os.path.join(out_dir, 'regular_das/'
    #                                                'img%d_CoM_4.png'
    #                                       % idx))

    # 2. Binary DAS (domain partitioning)
    # imgs = make_imgs_arr(os.path.join(recon_dir, 'binary-das/'))
    #
    # c_xs_max, c_ys_max, xs_max, ys_max = do_pos_err_analysis(imgs=imgs,
    #                                      tar_xs=t_coords[:, 0],
    #                                      tar_ys=t_coords[:, 1],
    #                                      roi_rad=roi_rad,
    #                                      o_dir_str='binary_das/',
    #                                      save_str='%s' % 'binary_das_max_4',
    #                                      use_img_maxes=True,
    #                                      make_plts=True,
    #                                      logger=logger)
    #
    # plt_rand_pos_errs(c_xs=c_xs_max, c_ys=c_ys_max,
    #                   xs=t_coords[:, 0], ys=t_coords[:, 1],
    #                   o_dir_str='binary_das/',
    #                   save_str='%s' %
    #                            'binary_das_m_4',
    #                   make_plts=True, logger=logger)
    #
    # c_xs_CoM, c_ys_CoM, xs_CoM, ys_CoM = do_pos_err_analysis(imgs=imgs,
    #                                      tar_xs=t_coords[:, 0],
    #                                      tar_ys=t_coords[:, 1],
    #                                      roi_rad=roi_rad,
    #                                      o_dir_str='binary_das/',
    #                                      save_str='%s' % 'binary_das_com_4',
    #                                      use_img_maxes=False,
    #                                      make_plts=True,
    #                                      logger=logger)
    #
    # plt_rand_pos_errs(c_xs=c_xs_CoM, c_ys=c_ys_CoM,
    #                   xs=t_coords[:, 0], ys=t_coords[:, 1],
    #                   o_dir_str='binary_das/',
    #                   save_str='%s' %
    #                            'binary_das_com_4',
    #                   make_plts=True, logger=logger)
    #
    # for (img, idx) in zip(imgs, range(np.size(t_coords, axis=0))):
    #
    #     _, x_cm, y_cm = load_pickle(os.path.join(recon_dir,
    #                                              'spln_pars/'
    #                                              'id%d_pars.pickle'
    #                                              % (idx+1)))
    #
    #     plot_fd_img(img=np.abs(img), loc_err=True,
    #                 tum_x=t_coords[idx, 0] / 100,
    #                 tum_y=t_coords[idx, 1] / 100,
    #                 tum_x_uncor=xs_max[idx] / 100,
    #                 tum_y_uncor=ys_max[idx] / 100,
    #                 tum_x_cor=c_xs_max[idx] / 100,
    #                 tum_y_cor=c_ys_max[idx] / 100, tum_rad=tum_rad,
    #                 img_rad=roi_rad, adi_rad=adi_rad, ox=x_cm, oy=y_cm,
    #                 ant_rad=ant_rad, roi_rad=roi_rad, save_fig=True,
    #                 save_close=True,
    #                 save_str=os.path.join(out_dir, 'binary_das/'
    #                                                'img%d_max_4.png'
    #                                       % idx))
    #
    #     plot_fd_img(img=np.abs(img), loc_err=True,
    #                 tum_x=t_coords[idx, 0] / 100,
    #                 tum_y=t_coords[idx, 1] / 100,
    #                 tum_x_uncor=xs_CoM[idx] / 100,
    #                 tum_y_uncor=ys_CoM[idx] / 100,
    #                 tum_x_cor=c_xs_CoM[idx] / 100,
    #                 tum_y_cor=c_ys_CoM[idx] / 100, tum_rad=tum_rad,
    #                 adi_rad=adi_rad, ox=x_cm, oy=y_cm, ant_rad=ant_rad,
    #                 roi_rad=roi_rad, save_fig=True, img_rad=roi_rad,
    #                 save_str=os.path.join(out_dir, 'binary_das/'
    #                                                'img%d_CoM_4.png'
    #                                       % idx))

    # # 3. Frequency-dependent DAS (zero cond)
    #
    # imgs = make_imgs_arr(os.path.join(recon_dir, 'freq_dep_non_cond-das/'))
    #
    # c_xs_max, c_ys_max, xs_max, ys_max = do_pos_err_analysis(imgs=imgs,
    #                                      tar_xs=t_coords[:, 0],
    #                                      tar_ys=t_coords[:, 1],
    #                                      roi_rad=roi_rad,
    #                                      o_dir_str='freq_dep_non_cond_das/',
    #                                      save_str='%s' % 'fdnc_das_max_4',
    #                                      use_img_maxes=True,
    #                                      make_plts=True,
    #                                      logger=logger)
    #
    # plt_rand_pos_errs(c_xs=c_xs_max, c_ys=c_ys_max,
    #                   xs=t_coords[:, 0], ys=t_coords[:, 1],
    #                   o_dir_str='freq_dep_non_cond_das/',
    #                   save_str='%s' %
    #                            'fdnc_das_m_4',
    #                   make_plts=True, logger=logger)
    #
    # c_xs_CoM, c_ys_CoM, xs_CoM, ys_CoM = do_pos_err_analysis(imgs=imgs,
    #                                      tar_xs=t_coords[:, 0],
    #                                      tar_ys=t_coords[:, 1],
    #                                      roi_rad=roi_rad,
    #                                      o_dir_str='freq_dep_non_cond_das/',
    #                                      save_str='%s' % 'fdnc_das_com_4',
    #                                      use_img_maxes=False,
    #                                      make_plts=True,
    #                                      logger=logger)
    #
    # plt_rand_pos_errs(c_xs=c_xs_CoM, c_ys=c_ys_CoM,
    #                   xs=t_coords[:, 0], ys=t_coords[:, 1],
    #                   o_dir_str='freq_dep_non_cond_das/',
    #                   save_str='%s' %
    #                            'fdnc_das_com_4',
    #                   make_plts=True, logger=logger)
    #
    # for (img, idx) in zip(imgs, range(np.size(t_coords, axis=0))):
    #     _, x_cm, y_cm = load_pickle(os.path.join(recon_dir,
    #                                              'spln_pars/'
    #                                              'id%d_pars.pickle'
    #                                              % (idx + 1)))
    #
    #     plot_fd_img(img=np.abs(img), loc_err=True,
    #                 tum_x=t_coords[idx, 0] / 100,
    #                 tum_y=t_coords[idx, 1] / 100,
    #                 tum_x_uncor=xs_max[idx] / 100,
    #                 tum_y_uncor=ys_max[idx] / 100,
    #                 tum_x_cor=c_xs_max[idx] / 100,
    #                 tum_y_cor=c_ys_max[idx] / 100, tum_rad=tum_rad,
    #                 img_rad=roi_rad, adi_rad=adi_rad, ox=x_cm, oy=y_cm,
    #                 ant_rad=ant_rad, roi_rad=roi_rad, save_fig=True,
    #                 save_close=True,
    #                 save_str=os.path.join(out_dir, 'freq_dep_non_cond_das/'
    #                                                'img%d_max_4.png'
    #                                       % idx))
    #
    #     plot_fd_img(img=np.abs(img), loc_err=True,
    #                 tum_x=t_coords[idx, 0] / 100,
    #                 tum_y=t_coords[idx, 1] / 100,
    #                 tum_x_uncor=xs_CoM[idx] / 100,
    #                 tum_y_uncor=ys_CoM[idx] / 100,
    #                 tum_x_cor=c_xs_CoM[idx] / 100,
    #                 tum_y_cor=c_ys_CoM[idx] / 100, tum_rad=tum_rad,
    #                 adi_rad=adi_rad, ox=x_cm, oy=y_cm, ant_rad=ant_rad,
    #                 roi_rad=roi_rad, save_fig=True, img_rad=roi_rad,
    #                 save_str=os.path.join(out_dir, 'freq_dep_non_cond_das/'
    #                                                'img%d_CoM_4.png'
    #                                       % idx))
    #
    # # 4. Frequency-dependent DAS
    #
    # imgs = make_imgs_arr(os.path.join(recon_dir, 'freq_dep-das/'))
    #
    # c_xs_max, c_ys_max, xs_max, ys_max = do_pos_err_analysis(imgs=imgs,
    #                                      tar_xs=t_coords[:, 0],
    #                                      tar_ys=t_coords[:, 1],
    #                                      roi_rad=roi_rad,
    #                                      o_dir_str='freq_dep_das/',
    #                                      save_str='%s' % 'freq_dep_max_4',
    #                                      use_img_maxes=True,
    #                                      make_plts=True,
    #                                      logger=logger)
    #
    # plt_rand_pos_errs(c_xs=c_xs_max, c_ys=c_ys_max,
    #                   xs=t_coords[:, 0], ys=t_coords[:, 1],
    #                   o_dir_str='freq_dep_das/',
    #                   save_str='%s' %
    #                            'freq_dep_m_4',
    #                   make_plts=True, logger=logger)
    #
    # c_xs_CoM, c_ys_CoM, xs_CoM, ys_CoM = do_pos_err_analysis(imgs=imgs,
    #                                      tar_xs=t_coords[:, 0],
    #                                      tar_ys=t_coords[:, 1],
    #                                      roi_rad=roi_rad,
    #                                      o_dir_str='freq_dep_das/',
    #                                      save_str='%s' % 'freq_dep_das_com_4',
    #                                      use_img_maxes=False,
    #                                      make_plts=True,
    #                                      logger=logger)
    #
    # plt_rand_pos_errs(c_xs=c_xs_CoM, c_ys=c_ys_CoM,
    #                   xs=t_coords[:, 0], ys=t_coords[:, 1],
    #                   o_dir_str='freq_dep_das/',
    #                   save_str='%s' %
    #                            'freq_dep_das_com_4',
    #                   make_plts=True, logger=logger)
    #
    # for (img, idx) in zip(imgs, range(np.size(t_coords, axis=0))):
    #     _, x_cm, y_cm = load_pickle(os.path.join(recon_dir,
    #                                              'spln_pars/'
    #                                              'id%d_pars.pickle'
    #                                              % (idx + 1)))
    #
    #     plot_fd_img(img=np.abs(img), loc_err=True,
    #                 tum_x=t_coords[idx, 0] / 100,
    #                 tum_y=t_coords[idx, 1] / 100,
    #                 tum_x_uncor=xs_max[idx] / 100,
    #                 tum_y_uncor=ys_max[idx] / 100,
    #                 tum_x_cor=c_xs_max[idx] / 100,
    #                 tum_y_cor=c_ys_max[idx] / 100, tum_rad=tum_rad,
    #                 img_rad=roi_rad, adi_rad=adi_rad, ox=x_cm, oy=y_cm,
    #                 ant_rad=ant_rad, roi_rad=roi_rad, save_fig=True,
    #                 save_close=True,
    #                 save_str=os.path.join(out_dir, 'freq_dep_das/'
    #                                                'img%d_max_4.png'
    #                                       % idx))
    #
    #     plot_fd_img(img=np.abs(img), loc_err=True,
    #                 tum_x=t_coords[idx, 0] / 100,
    #                 tum_y=t_coords[idx, 1] / 100,
    #                 tum_x_uncor=xs_CoM[idx] / 100,
    #                 tum_y_uncor=ys_CoM[idx] / 100,
    #                 tum_x_cor=c_xs_CoM[idx] / 100,
    #                 tum_y_cor=c_ys_CoM[idx] / 100, tum_rad=tum_rad,
    #                 adi_rad=adi_rad, ox=x_cm, oy=y_cm, ant_rad=ant_rad,
    #                 roi_rad=roi_rad, save_fig=True, img_rad=roi_rad,
    #                 save_str=os.path.join(out_dir, 'freq_dep_das/'
    #                                                'img%d_CoM_4.png'
    #                                       % idx))

    # 5. Ray-tracing

    # imgs = make_imgs_arr(os.path.join(recon_dir, 'rt-das/'))
    #
    # c_xs_max, c_ys_max, xs_max, ys_max = do_pos_err_analysis(imgs=imgs,
    #                                      tar_xs=t_coords[:, 0],
    #                                      tar_ys=t_coords[:, 1],
    #                                      roi_rad=roi_rad,
    #                                      o_dir_str='rt_das/',
    #                                      save_str='%s' % 'rt_max_4',
    #                                      use_img_maxes=True,
    #                                      make_plts=True,
    #                                      logger=logger)
    #
    # plt_rand_pos_errs(c_xs=c_xs_max, c_ys=c_ys_max,
    #                   xs=t_coords[:, 0], ys=t_coords[:, 1],
    #                   o_dir_str='rt_das/',
    #                   save_str='%s' %
    #                            'rt_m_4',
    #                   make_plts=True, logger=logger)
    #
    # c_xs_CoM, c_ys_CoM, xs_CoM, ys_CoM = do_pos_err_analysis(imgs=imgs,
    #                                      tar_xs=t_coords[:, 0],
    #                                      tar_ys=t_coords[:, 1],
    #                                      roi_rad=roi_rad,
    #                                      o_dir_str='rt_das/',
    #                                      save_str='%s' % 'rt_das_com_4',
    #                                      use_img_maxes=False,
    #                                      make_plts=True,
    #                                      logger=logger)
    #
    # plt_rand_pos_errs(c_xs=c_xs_CoM, c_ys=c_ys_CoM,
    #                   xs=t_coords[:, 0], ys=t_coords[:, 1],
    #                   o_dir_str='rt_das/',
    #                   save_str='%s' %
    #                            'rt_das_com_4',
    #                   make_plts=True, logger=logger)
    #
    # for (img, idx) in zip(imgs, range(np.size(t_coords, axis=0))):
    #     cs, x_cm, y_cm = load_pickle(os.path.join(recon_dir,
    #                                              'spln_pars/'
    #                                              'id%d_pars.pickle'
    #                                              % (idx + 1)))
    #
    #     plot_fd_img(img=np.abs(img), loc_err=True, cs=cs,
    #                 tum_x=t_coords[idx, 0] / 100,
    #                 tum_y=t_coords[idx, 1] / 100,
    #                 tum_x_uncor=xs_max[idx] / 100,
    #                 tum_y_uncor=ys_max[idx] / 100,
    #                 tum_x_cor=c_xs_max[idx] / 100,
    #                 tum_y_cor=c_ys_max[idx] / 100, tum_rad=tum_rad,
    #                 img_rad=roi_rad, adi_rad=adi_rad, ox=x_cm, oy=y_cm,
    #                 ant_rad=ant_rad, roi_rad=roi_rad, save_fig=True,
    #                 save_close=True,
    #                 save_str=os.path.join(out_dir, 'rt_das/'
    #                                                'img%d_max_4.png'
    #                                       % idx))
    #
    #     plot_fd_img(img=np.abs(img), loc_err=True, cs=cs,
    #                 tum_x=t_coords[idx, 0] / 100,
    #                 tum_y=t_coords[idx, 1] / 100,
    #                 tum_x_uncor=xs_CoM[idx] / 100,
    #                 tum_y_uncor=ys_CoM[idx] / 100,
    #                 tum_x_cor=c_xs_CoM[idx] / 100,
    #                 tum_y_cor=c_ys_CoM[idx] / 100, tum_rad=tum_rad,
    #                 adi_rad=adi_rad, ox=x_cm, oy=y_cm, ant_rad=ant_rad,
    #                 roi_rad=roi_rad, save_fig=True, img_rad=roi_rad,
    #                 save_str=os.path.join(out_dir, 'rt_das/'
    #                                                'img%d_CoM_4.png'
    #                                       % idx))

    ############################################################
