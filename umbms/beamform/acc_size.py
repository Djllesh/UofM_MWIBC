"""
Tyson Reimer
University of Manitoba
December 15th, 2022
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('Qt5Agg')

from scipy.ndimage import rotate

from umbms.beamform.extras import (get_fd_phase_factor, get_xy_arrs)

###############################################################################

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


def get_img_max_xy(img, img_rad):
    """Get the x/y positions of the maximum response in the image in cm

    Parameters
    ----------
    img : array_like
        The reconstructed image
    img_rad : float
        The radius of the image, in units of [cm]

    Returns
    -------

    """

    img_for_this = np.abs(img)  # Take abs

    pix_to_dist = 2 * img_rad / np.size(img, axis=0)  # Ratio for pos

    max_loc = np.argmax(img_for_this)  # Max x/y locs
    max_y_idx, max_x_idx = np.unravel_index(max_loc, np.shape(img))

    # Convert to positions
    max_x_pos = (max_x_idx - np.size(img, 0) // 2) * pix_to_dist
    max_y_pos = (np.size(img, 0) // 2 - max_y_idx) * pix_to_dist

    max_x_pos *= 100
    max_y_pos *= 100

    return max_x_pos, max_y_pos


def get_img_CoM(img, img_rad):
    # img_for_this = np.fliplr(np.abs(img))  # Rotate, take abs
    img_for_this = np.abs(img) ** 2
    # img_for_this = np.fliplr(img_for_this)

    xs, ys = get_xy_arrs(m_size=np.size(img, axis=0), roi_rad=img_rad)

    max_x, max_y = get_img_max_xy(img=img, img_rad=img_rad)
    max_x /= 100
    max_y /= 100

    roi = np.sqrt((max_x - xs)**2 + (max_y - ys)**2) < 0.012

    img_for_this[~roi] = 0

    tot_intensity = np.sum(img_for_this)

    x_cent_mass = np.sum(xs * img_for_this) / tot_intensity
    y_cent_mass = np.sum(ys * img_for_this) / tot_intensity

    x_cent_mass *= 100
    y_cent_mass *= 100

    return x_cent_mass, y_cent_mass


def do_size_analysis(img_here, dx, roi_rad, make_plts=False,
                     save_dir='', save_str='', rotate_img=False):

    # Get center of mass position
    com_x, com_y = get_img_CoM(img=img_here,
                               img_rad=roi_rad)

    com_ang = np.arctan2(com_y, com_x)
    if com_ang > 0:
        angle = 0.5 * np.pi - com_ang
    else:
        angle = np.abs(com_ang) + 0.5 * np.pi

    # Rotate the image so rho-hat --> x-hat and
    # phi-hat --> y-hat
    rot_img = rotate(img_here, angle=np.rad2deg(angle), reshape=False)

    # Get pixel positions for rotated image
    xs_rot = np.flip((np.size(rot_img, axis=0) / 2
              - np.arange(np.size(rot_img, axis=0))) * dx)
    xs_rot *= 100  # Convert from [m] to [cm]

    # Get the CoM position in the rotated image
    rot_com_x, rot_com_y = \
        get_img_CoM(img=rot_img,
                    img_rad=np.max(np.abs(xs_rot)) / 100)

    # Get the coordinates of the center of mass pixels
    rot_com_x_coord = np.argmin(np.abs(xs_rot - rot_com_x))
    rot_com_y_coord = np.argmin(np.abs(np.flip(xs_rot) - rot_com_y))

    rot_img /= np.max(rot_img)

    # rot_img /= rot_img[rot_com_y_coord, rot_com_x_coord]


    # Get xs/ys for plotting, centered on CoM pixel
    plt_xs = xs_rot - rot_com_x
    plt_ys = xs_rot - rot_com_y


    rho_slice = rot_img[:, rot_com_x_coord]
    phi_slice = rot_img[rot_com_y_coord, :]

    if make_plts:
        init_plt()
        plt.plot(plt_xs,
                 rho_slice, 'k-',
                 label=r'$\mathdefault{\hat{\rho}}$')
        plt.plot(plt_ys,
                 phi_slice, 'b--',
                 label=r'$\mathdefault{\hat{\phi}}$')
        plt.plot(np.linspace(-2.5, 2.5, 1000),
                 0.05 * np.ones([1000, ]),
                 'g--',
                 label='5%')
        plt.plot(np.linspace(-2.5, 2.5, 1000),
                 0.1 * np.ones([1000, ]),
                 'r--',
                 label='10%')
        plt.xlabel('Position (cm)', fontsize=22)
        plt.ylabel("Image Intensity", fontsize=22)
        plt.xlim([-2.5, 2.5])
        plt.legend(fontsize=22)
        plt.show()
        plt.savefig(os.path.join(save_dir, save_str), transparent=True, dpi=150)
        plt.savefig(os.path.join(save_dir, 'NT_%s' % save_str),
                    transparent=False, dpi=150)
        plt.close()

    rho_roi = rho_slice >= 0.5
    phi_roi = phi_slice >= 0.5

    while np.all(~phi_roi):
        rot_com_y_coord += 1
        phi_slice = rot_img[rot_com_y_coord, :]
        phi_roi = phi_slice >= 0.5

    rho_slice_width = np.max(plt_xs[rho_roi]) - np.min(plt_xs[rho_roi])
    phi_slice_width = np.max(plt_ys[phi_roi]) - np.min(plt_ys[phi_roi])

    if rotate_img:
        return rho_slice_width, phi_slice_width, rot_img

    return rho_slice_width, phi_slice_width
