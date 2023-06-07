"""
Tyson Reimer
University of Manitoba
December 15th, 2022
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scipy.stats import spearmanr

matplotlib.use('Qt5Agg')

from umbms import get_proj_path, verify_path

from umbms.beamform.extras import (get_xy_arrs)
from umbms.analysis.acc_size import get_img_max_xy, get_img_CoM

###############################################################################

__D_DIR = os.path.join(get_proj_path(), 'data/tyson//')

__OUT_DIR = os.path.join(get_proj_path(), 'output/iqms-dec/sym/')
verify_path(__OUT_DIR)

# Scan VNA parameters
# __INI_F = 0.7e9 #700e6
# __FIN_F = 8.0e9
# __N_FS = 1001
# __SCAN_FS = np.linspace(__INI_F, __FIN_F, __N_FS)
#
# __N_ANT_POS = 24
# __INI_ANT_ANG = 7.5  # Polar angle of position of 0th antenna
# __ANT_SEP = 15  # Angular separation of adjacent antennas, in [deg]
#
# # Reconstruction parameters
# __N_CORES = 10
__ROI_RAD = 0.08


# __M_SIZE = 150
# __ANT_RAD = 0.379 + 0.0826  # Measured radius + 1-way time delay in [m]
# __SPEED = 2.997e8

###############################################################################


def init_plt():
    """Init plot window
    """

    plt.figure(figsize=(12, 8))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=16)


def measure_symmetry(img, img_rad, use_CoM=True, max_rho=2):
    """Calculate the symmetry of the target response

    Parameters
    ----------
    img : array_like
        2D image
    img_rad : float
        Radius of image, in units of [m]
    use_CoM : bool
        If True, shifts the coordinate axes to be aligned on the CoM
        of the response. If False, uses the maximum pixel response
        location instead
    max_rho : float
        Maximum rho for integration, in [cm]; default 2 cm

    Returns
    -------

    """

    img_to_use = np.abs(img) ** 2
    img_to_use /= np.max(img_to_use)

    # Get xs, ys of pixels in units of [m]
    xs, ys = get_xy_arrs(m_size=np.size(img_to_use, axis=0), ant_rad=img_rad)

    xs *= 100  # Convert to cm
    ys *= 100  # Convert to cm

    if use_CoM:
        center_x, center_y = get_img_CoM(img=img_to_use, img_rad=img_rad)
    else:
        center_x, center_y = get_img_max_xy(img=img_to_use, img_rad=img_rad)

    # Shift coord axes
    xs -= center_x
    ys -= center_y

    ring_width = 0.1  # Width of each ring in [cm]
    rings = np.arange(start=0, stop=max_rho + ring_width, step=ring_width)

    integral_val = 0
    for ii in range(1, len(rings)):  # For each ring

        ring_rho = rings[ii]  # This ring
        ring_rho_min = rings[ii] - ring_width / 2  # Ring min rho
        ring_rho_max = rings[ii] + ring_width / 2  # Ring max rho

        # Find pixels within the ring
        ring_roi = np.logical_and(np.sqrt(xs ** 2 + ys ** 2) > ring_rho_min,
                                  np.sqrt(xs ** 2 + ys ** 2) <= ring_rho_max)

        dA_ring = (np.pi * (ring_rho_max ** 2 - ring_rho_min ** 2)
                   / np.sum(ring_roi))

        val_here = (dA_ring
                    * np.sum(np.abs(img_to_use[ring_roi]
                                    - np.mean(img_to_use[ring_roi])))
                    / (2 * np.pi * ring_rho))

        integral_val += val_here

    integral_val /= max_rho

    sym_metric = integral_val

    return sym_metric


def do_symmetry_analysis(img_list, tar_coords, recon_algos, system_str, logger,
                         use_CoM=True):
    """

    Parameters
    ----------
    img_list
    recon_algos
    system_str
    use_CoM

    """

    logger.info('\t%s system data...' % system_str)

    sym_list = []  # Init list for return

    for ii in range(len(img_list)):  # For each *type* of reconstruction

        imgs = img_list[ii]  # Get the array of images here

        sym_vals = np.zeros([np.size(imgs, axis=0), ])  # Init arr

        for jj in range(np.size(imgs, axis=0)):  # For each image

            # Get the value of the symmetry metric
            sym_vals[jj] = measure_symmetry(img=imgs[jj],
                                            img_rad=__ROI_RAD,
                                            use_CoM=use_CoM,
                                            max_rho=2)

        sym_list.append(sym_vals)

    # Get target polar rho position
    tar_rhos = np.sqrt(tar_coords[:, 0]**2 + tar_coords[:, 1]**2)
    colors = [
        [0, 0, 0],
        [100 / 255, 100 / 255, 100 / 255],
        [205 / 255, 205 / 255, 205 / 255],
    ]

    mean_rhos = []
    unique_rhos = np.unique(tar_rhos)
    for ii in range(len(img_list)):
        mean_rhos.append(np.zeros([np.size(unique_rhos), 2]))
        for jj in range(len(unique_rhos)):
            syms_here = sym_list[ii][tar_rhos == unique_rhos[jj]]
            mean_rhos[ii][jj, :] = np.mean(syms_here), np.std(syms_here)

    # Do Spearman correlation to determine if the symmery score
    # depends on the rho
    for ii in range(len(img_list)):
        corr, pval = spearmanr(a=tar_rhos, b=sym_list[ii],
                               alternative='greater')

        logger.info('\t\t%s: Spearman Correlation = %.3f'
                    % (recon_algos[ii], corr))
        logger.info('\t\t%s: p-value: %.3e' % (recon_algos[ii], pval))

    # Make a plot...
    init_plt()
    for ii in range(len(img_list)):
        # plt.scatter(tar_rhos, sym_list[ii], label='%s' % recon_algos[ii],
        #             color=colors[ii], marker='o', )
        plt.errorbar(unique_rhos, mean_rhos[ii][:, 0],
                     yerr=mean_rhos[ii][:, 1],
                     color=colors[ii], marker='o', capsize=5,
                     label='%s' % recon_algos[ii],
                     linestyle='')
    plt.xlabel(r'Target $\mathdefault{\rho}}$ Position (cm)', fontsize=22)
    plt.ylabel(r'Symmetry Measure, M$_{\mathregular{sym}}$',
               fontsize=22)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(__OUT_DIR, '%s_symmetry.png' % system_str),
                dpi=300, transparent=True)
    plt.savefig(os.path.join(__OUT_DIR, 'NT_%s_symmetry.png' % system_str),
                dpi=300, transparent=False)


###############################################################################