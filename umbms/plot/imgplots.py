"""
Tyson Reimer
University of Manitoba
June 18, 2019
"""
import matplotlib
import matplotlib.widgets
import numpy as np
import matplotlib.pyplot as plt

# matplotlib.use('Agg')

import umbms.beamform.breastmodels as breastmodels
from umbms.beamform.utility import get_xy_arrs
from umbms.plot.stlplot import get_phantom_xy_for_z, get_shell_xy_for_z

# ##############################################################################

# Conversion factor from [m] to [cm]
__M_to_CM = 100

# Conversion factor from [GHz] to [Hz]
__GHz_to_Hz = 1e9


###############################################################################

def plot_img(img, tum_x=0.0, tum_y=0.0, tum_rad=0.0, adi_rad=0.0, ant_rad=0.0,
             save_str='', save_fig=False, cmap='inferno', max_val=1.0,
             title='', normalize=True, crop_img=True, cbar_fmt='%.1f',
             norm_cbar=False, cbar_max=1.0, transparent=True, dpi=300,
             save_close=True):
    """Displays a reconstruction, making a publication-ready figure

    Parameters
    ----------
    img : array_like
        The 2D reconstruction that will be displayed
    tum_x : float
        The x-position of the tumor in the scan, in meters
    tum_y : float
        The y-position of the tumor in the scan, in meters
    tum_rad : float
        The radius of the tumor in the scan, in meters
    adi_rad : float*
        The approximate radius of the breast in the scan, in meters
    ant_rad : float
        The corrected antenna radius during the scan (corrected for
        antenna time-delay)
    save_str : str, optional
        The complete path for saving the figure as a .png - only used if
        save_fig is True
    save_fig : bool
        If True, will save the displayed image to the location specified
        in save_str
    cmap : str
        Specifies the colormap that will be used when displaying the
        image
    max_val : float
        The maximum intensity to display in the image
    title : str
        The title for the plot
    normalize : bool
        If True, will normalize the image to have the maximum value
        max_val
    crop_img : bool
        If True, will set the values in the image to NaN outside of the
        inner antenna trajectory
    cbar_fmt : str
        The format for the tick labels on the colorbar
    norm_cbar : bool
        If True, will normalize the colorbar to have min value 0,
        max value of cbar_max
    cbar_max : float
        If norm_cbar, this will be the maximum value of the cbar
    transparent : bool
        If True, will save the image with a transparent background
        (i.e., whitespace will be transparent)
    dpi : int
        The DPI to be used if saving the image
    save_close : bool
        If True, closes the figure after saving
    """

    img_to_plt = img * np.ones_like(img)

    ant_rad *= 100  # Convert from m to cm to facilitate plot

    # If cropping the image at the antenna-trajectory boundary
    if crop_img:
        temp_val = (ant_rad - 14.8) / 0.97 + 10.6

        # Find the region inside the antenna trajectory
        in_ant_trajectory = breastmodels.get_roi(temp_val - 10,
                                                 np.size(img_to_plt, axis=0),
                                                 ant_rad)

        # Set the pixels outside of the antenna trajectory to NaN
        img_to_plt[np.logical_not(in_ant_trajectory)] = np.NaN

    # Define angles for plot the tissue geometry
    draw_angs = np.linspace(0, 2 * np.pi, 1000)

    if normalize:  # If normalizing the image

        # If the max value isn't set to unity, then scale the image so
        # that intensity is between 0 and max_val
        if max_val != 1.0:
            img_to_plt = \
                (img_to_plt
                 - np.min(img_to_plt[np.logical_not(np.isnan(img_to_plt))]))
            img_to_plt = img_to_plt / max_val

        # If the max value is 1.0, scale so that the intensity is
        # between 0 and 1.0
        else:
            img_to_plt = \
                (img_to_plt
                 - np.min(img_to_plt[np.logical_not(np.isnan(img_to_plt))]))
            img_to_plt = \
                (img_to_plt
                 / np.max(img_to_plt[np.logical_not(np.isnan(img_to_plt))]))

    # Rotate and flip img to match proper x/y axes labels
    img_to_plot = (img_to_plt ** 2).T

    temp_val = (ant_rad - 14.8) / 0.97 + 10.6
    ant_xs, ant_ys = ((temp_val - 10) * np.cos(draw_angs),
                      (temp_val - 10) * np.sin(draw_angs))

    # Define the x/y coordinates of the approximate breast outline
    breast_xs, breast_ys = (adi_rad * 100 * np.cos(draw_angs),
                            adi_rad * 100 * np.sin(draw_angs))

    # Define the x/y coordinates of the approximate tumor outline
    tum_xs, tum_ys = (tum_rad * 100 * np.cos(draw_angs) + tum_x * 100,
                      tum_rad * 100 * np.sin(draw_angs) + tum_y * 100)

    tick_bounds = [-ant_rad, ant_rad, -ant_rad, ant_rad]

    # Set the font to times new roman
    plt.rc('font', family='Times New Roman')
    plt.figure()  # Make the figure window

    # If normalizing the image, ensure the cbar axis extends from 0 to 1
    if normalize:
        plt.imshow(img_to_plot, cmap=cmap, extent=tick_bounds,
                   aspect='equal', vmin=0, vmax=1.0)
    elif norm_cbar:
        plt.imshow(img_to_plot, cmap=cmap, extent=tick_bounds,
                   aspect='equal', vmin=0, vmax=cbar_max)
    else:  # If nor normalizing the image
        plt.imshow(img_to_plot, cmap=cmap, extent=tick_bounds, aspect='equal')

    # Set the size of the axis tick labels
    plt.tick_params(labelsize=14)

    # Set the x/y-ticks at multiples of 5 cm
    plt.gca().set_xticks([-10, -5, 0, 5, 10])
    plt.gca().set_yticks([-10, -5, 0, 5, 10])

    # Specify the colorbar tick format and size
    plt.colorbar(format=cbar_fmt).ax.tick_params(labelsize=14)

    # Set the x/y axes limits
    plt.xlim([-temp_val + 10, temp_val - 10])
    plt.ylim([-temp_val + 10, temp_val - 10])

    plt.plot(ant_xs, ant_ys, 'k--', linewidth=2.5)
    plt.plot(breast_xs, breast_ys, 'w--', linewidth=2)

    # Plot the approximate tumor boundary
    plt.plot(tum_xs, tum_ys, 'g', linewidth=1.5)

    plt.title(title, fontsize=20)  # Make the plot title
    plt.xlabel('x-axis (cm)', fontsize=16)  # Make the x-axis label
    plt.ylabel('y-axis (cm)', fontsize=16)  # Make the y-axis label
    plt.tight_layout()  # Remove excess whitespace in the figure

    # If saving the image, save it to the save_str path and close it
    if save_fig:
        plt.savefig(save_str, transparent=transparent, dpi=dpi,
                    bbox_inches='tight')

        if save_close:  # If wanting to close the fig after saving
            plt.close()


###############################################################################


def plot_fd_img(img, *, bound_x=None, bound_y=None, cs=None, mask=None,
                tum_x=0.0, tum_y=0.0, loc_err=False,
                tum_x_uncor=0.0, tum_y_uncor=0.0,
                tum_x_cor=0.0, tum_y_cor=0.0, tum_rad=0.0,
                adi_rad=0.0, ox=0.0, oy=0.0, mid_breast_max=0.0,
                mid_breast_min=0.0, ant_rad=0.0, roi_rad=0.0, img_rad=0.0,
                save_str='', save_fig=False, tar2_x=0.0, tar2_y=0.0,
                tar2_rad=0.0, cmap='inferno', title='', crop_img=True,
                cbar_fmt='%.1f', phantom_id='', plot_stl=False, stl_z=0.0,
                transparent=False, dpi=300, save_close=True,
                partial_ant_idx=None):
    """Displays a reconstruction, making a publication-ready figure

    Parameters
    ----------
    img : array_like
        The 2D reconstruction that will be displayed
    tum_x : float
        The x-position of the tumor in the scan, in meters
    tum_y : float
        The y-position of the tumor in the scan, in meters
    tum_rad : float
        The radius of the tumor in the scan, in meters
    mid_breast_max : float
        major semi-axis of a phantom (b)
    mid_breast_min : float
        minor semi-axis of a phantom (a)
    ant_rad : float
        The corrected antenna radius during the scan (corrected for
        antenna time-delay)
    img_rad : float
        The radius of the image-space
    save_str : str, optional
        The complete path for saving the figure as a .png - only used if
        save_fig is True
    save_fig : bool
        If True, will save the displayed image to the location specified
        in save_str
    cmap : str
        Specifies the colormap that will be used when displaying the
        image
    title : str
        The title for the plot
    crop_img : bool
        If True, will set the values in the image to NaN outside of the
        inner antenna trajectory
    cbar_fmt : str
        The format for the tick labels on the colorbar
    norm_cbar : bool
        If True, will normalize the colorbar to have min value 0,
        max value of cbar_max
    cbar_max : float
        If norm_cbar, this will be the maximum value of the cbar
    transparent : bool
        If True, will save the image with a transparent background
        (i.e., whitespace will be transparent)
    dpi : int
        The DPI to be used if saving the image
    save_close : bool
        If True, closes the figure after saving
    """
    pix_xs, pix_ys = get_xy_arrs(np.size(img, axis=0), roi_rad)
    pix_xs *= 100
    pix_ys *= 100
    img_to_plt = np.abs(img)
    img_to_plt = img * np.ones_like(img)
    img_to_rot = img_to_plt
    # ant_rad *= 100  # Convert from m to cm to facilitate plot
    img_rad *= 100
    # adi_rad *= 100
    roi_rad *= 100

    # If cropping the image at the antenna-trajectory boundary
    if crop_img:
        # Find the region inside the antenna trajectory
        roi = breastmodels.get_roi(roi_rad, np.size(img_to_plt, axis=0),
                                   img_rad)

        # Set the pixels outside of the antenna trajectory to NaN
        img_to_plt[np.logical_not(roi)] = np.NaN

    # Define angles for plot the tissue geometry
    draw_angs = np.linspace(0, 2 * np.pi, 1000)

    # Rotate and flip img to match proper x/y axes labels
    # img_to_plt = (img_to_plt).T

    # new_ant_rad = (ant_rad - 14.8) / 0.97 + 10.6
    # ant_xs, ant_ys = ((new_ant_rad - 10) * np.cos(draw_angs),
    #                   (new_ant_rad - 10) * np.sin(draw_angs))

    # Define the x/y coordinates of the approximate breast outline
    if adi_rad == 0:
        breast_xs, breast_ys = (mid_breast_min * 100 * np.cos(draw_angs),
                                mid_breast_max * 100 * np.sin(draw_angs))
    else:
        breast_xs, breast_ys = (adi_rad * 100 * np.cos(draw_angs) + ox * 100,
                                adi_rad * 100 * np.sin(draw_angs) + oy * 100)

    # Define the x/y coordinates of the approximate tumor outline
    tum_xs, tum_ys = (tum_rad * 100 * np.cos(draw_angs) + tum_x * 100,
                      tum_rad * 100 * np.sin(draw_angs) + tum_y * 100)

    if loc_err:
        tum_xs_unc, tum_ys_unc = \
            (tum_rad * 100 * np.cos(draw_angs) + tum_x_uncor * 100,
             tum_rad * 100 * np.sin(draw_angs) + tum_y_uncor * 100)

        tum_xs_cor, tum_ys_cor = \
            (tum_rad * 100 * np.cos(draw_angs) + tum_x_cor * 100,
             tum_rad * 100 * np.sin(draw_angs) + tum_y_cor * 100)

    img_extent = [-img_rad, img_rad, -img_rad, img_rad]

    # Set the font to times new roman
    plt.rc('font', family='Times New Roman')
    plt.figure()  # Make the figure window

    plt.imshow(img_to_plt, cmap=cmap, extent=img_extent, aspect='equal')

    # Set the size of the axis tick labels
    plt.tick_params(labelsize=14)

    # Set the x/y-ticks at multiples of 5 cm
    plt.gca().set_xticks([-6, -4, -2, 0, 2, 4, 6])
    plt.gca().set_yticks([-6, -4, -2, 0, 2, 4, 6])

    # Specify the colorbar tick format and size
    plt.colorbar(format=cbar_fmt).ax.tick_params(labelsize=14)

    # Set the x/y axes limits
    plt.xlim([-roi_rad, roi_rad])
    plt.ylim([-roi_rad, roi_rad])

    # plt.plot(ant_xs, ant_ys, 'k--', linewidth=2.5)
    plt.plot(breast_xs, breast_ys, 'w--', linewidth=2,
             label='Approximate outline')

    if partial_ant_idx is not None:

        # Start from the initial angle of the scan
        ant_draw_angs = np.linspace(0, np.deg2rad(355), 72) + np.deg2rad(
            -136.0)
        # Make the marks inside the ROI
        new_ant_rad = roi_rad - 0.5
        ant_xs, ant_ys = (new_ant_rad * np.cos(ant_draw_angs),
                          new_ant_rad * np.sin(ant_draw_angs))
        # Assume counter-clockwise rotation
        ant_xs = np.flip(ant_xs)
        ant_ys = np.flip(ant_ys)

        # Plot: if the antenna is on - green, off - red
        for (x, y, i) in zip(ant_xs, ant_ys, range(72)):
            if partial_ant_idx[i]:
                plt.text(x, y, str(i), color='g', fontsize=7,
                         fontweight='bold', family='monospace')
            else:
                plt.text(x, y, str(i), color='r', fontsize=7,
                         fontweight='bold', family='monospace')

    if plot_stl:
        # if len(phantom_id) >= 4:
        #
        #     adi_id = phantom_id.split('F')[0].swapcase()
        #     fib_id = 'f' + phantom_id.split('F')[1]
        #
        #     phantom_xs, phantom_ys = \
        #         get_phantom_xy_for_z(adi_id, fib_id,
        #                              stl_z * 10, slice_thickness=1.0)
        # else:
        #     phantom_xs, phantom_ys = \
        #         get_shell_xy_for_z(phantom_id, stl_z * 10,
        #                            slice_thickness=1.0)

        phantom_xs, phantom_ys = \
            get_shell_xy_for_z(phantom_id.split('F')[0].swapcase(),
                               stl_z * 10, slice_thickness=1.0)
        phantom_xs /= 10
        phantom_ys /= 10

        plt.scatter(phantom_xs + ox * 100, phantom_ys + oy * 100, c='y',
                    s=0.05, label='Ground truth')

    # Plot the approximate tumor boundary
    plt.plot(tum_xs, tum_ys, 'g', label='Observed position', linewidth=1.5)

    if loc_err:
        plt.plot(tum_xs_unc, tum_ys_unc, 'r--',
                 label='Uncorrected reconstructed position', linewidth=1.3)
        plt.plot(tum_xs_cor, tum_ys_cor, 'b--',
                 label='Corrected reconstructed position', linewidth=1.3)

    tar2_xs, tar2_ys = (tar2_rad * 100 * np.cos(draw_angs) + tar2_x * 100,
                        tar2_rad * 100 * np.sin(draw_angs) + tar2_y * 100)

    # Plot secondary target
    plt.plot(tar2_xs, tar2_ys, 'g', linewidth=1.5)

    # plt.title(title, fontsize=20)  # Make the plot title
    plt.xlabel('x-axis (cm)', fontsize=16)  # Make the x-axis label
    plt.ylabel('y-axis (cm)', fontsize=16)  # Make the y-axis label
    plt.tight_layout()  # Remove excess whitespace in the figure

    # If saving the image, save it to the save_str path and close it
    if save_fig:
        if bound_x is not None:
            plt.plot(bound_x, bound_y, 'bx', label='Slices')
        if cs is not None:
            phi = np.deg2rad(np.arange(0, 360, 0.1))
            rho = cs(phi)
            xs = rho * np.cos(phi) * 100
            ys = rho * np.sin(phi) * 100
            plt.plot(xs, ys, 'r-', label='Cubic spline')
            # plt.legend(loc='upper left')

        if mask is not None:
            plt.plot(pix_xs[mask], pix_ys[mask], 'r.',
                     label=r'$\rho_i$ $\leqslant$ $\rho_f(\phi_i)$')
            # plt.plot(pix_xs[~mask], pix_ys[~mask], 'y.',
            #          label=r'$\rho_i > \rho_f(\phi_i)$')
            # plt.legend(loc='upper left')

        # plt.legend(loc='upper left')
        plt.savefig(save_str, transparent=transparent, dpi=dpi,
                    bbox_inches='tight')

        if save_close:  # If wanting to close the fig after saving
            plt.close()


def plot_fd_img_with_intersections(img, cs, ant_pos_x, ant_pos_y, pix_xs,
                                   pix_ys, int_f_xs, int_f_ys, int_b_xs,
                                   int_b_ys, tum_x=0.0, tum_y=0.0, tum_rad=0.0,
                                   mid_breast_max=0.0, mid_breast_min=0.0,
                                   ant_rad=0.0, adi_rad=0.0, ox=0.0, oy=0.0,
                                   roi_rad=0.0, img_rad=0.0, tar2_x=0.0,
                                   tar2_y=0.0, tar2_rad=0.0, cmap='inferno',
                                   title='', crop_img=True, cbar_fmt='%.1f',
                                   transparent=False, dpi=300):
    """Displays a reconstruction, making a publication-ready figure

    Parameters
    ----------
    img : array_like
        The 2D reconstruction that will be displayed
    tum_x : float
        The x-position of the tumor in the scan, in meters
    tum_y : float
        The y-position of the tumor in the scan, in meters
    tum_rad : float
        The radius of the tumor in the scan, in meters
    adi_rad : float
        The approximate radius of the breast in the scan, in meters
    ant_rad : float
        The corrected antenna radius during the scan (corrected for
        antenna time-delay)
    img_rad : float
        The radius of the image-space
    cmap : str
        Specifies the colormap that will be used when displaying the
        image
    title : str
        The title for the plot
    crop_img : bool
        If True, will set the values in the image to NaN outside of the
        inner antenna trajectory
    cbar_fmt : str
        The format for the tick labels on the colorbar
    norm_cbar : bool
        If True, will normalize the colorbar to have min value 0,
        max value of cbar_max
    cbar_max : float
        If norm_cbar, this will be the maximum value of the cbar
    transparent : bool
        If True, will save the image with a transparent background
        (i.e., whitespace will be transparent)
    dpi : int
        The DPI to be used if saving the image
    """

    # img_to_plt = np.abs(img)
    img_to_plt = img * np.ones_like(img)

    ant_rad *= 100  # Convert from m to cm to facilitate plot
    img_rad *= 100
    # adi_rad *= 100
    roi_rad *= 100
    ant_pos_x *= 100
    ant_pos_y *= 100
    int_f_xs *= 100
    # int_f_xs = int_f_xs.T
    int_f_ys *= 100
    # int_f_ys = int_f_ys.T
    int_b_xs *= 100
    # int_b_xs = int_b_xs.T
    int_b_ys *= 100
    # int_b_ys = int_b_ys.T
    pix_xs *= 100
    pix_ys *= 100

    # If cropping the image at the antenna-trajectory boundary
    if crop_img:
        # Find the region inside the antenna trajectory
        roi = breastmodels.get_roi(roi_rad, np.size(img_to_plt, axis=0),
                                   img_rad)

        # Set the pixels outside of the antenna trajectory to NaN
        img_to_plt[np.logical_not(roi)] = np.NaN

    # Define angles for plot the tissue geometry
    draw_angs = np.linspace(0, 2 * np.pi, 1000)

    # Rotate and flip img to match proper x/y axes labels
    # img_to_plot = (img_to_plt).T

    temp_val = (ant_rad - 14.8) / 0.97 + 10.6
    ant_xs, ant_ys = ((temp_val - 10) * np.cos(draw_angs),
                      (temp_val - 10) * np.sin(draw_angs))

    # Define the x/y coordinates of the approximate breast outline
    if adi_rad == 0:
        breast_xs, breast_ys = (mid_breast_min * 100 * np.cos(draw_angs),
                                mid_breast_max * 100 * np.sin(draw_angs))
    else:
        breast_xs, breast_ys = (adi_rad * 100 * np.cos(draw_angs) + ox * 100,
                                adi_rad * 100 * np.sin(draw_angs) + oy * 100)

    # Define the x/y coordinates of the approximate tumor outline
    tum_xs, tum_ys = (tum_rad * 100 * np.cos(draw_angs) + tum_x * 100,
                      tum_rad * 100 * np.sin(draw_angs) + tum_y * 100)

    img_extent = [-img_rad, img_rad, -img_rad, img_rad]

    # Set the font to times new roman
    plt.rc('font', family='Times New Roman')
    plt.figure()  # Make the figure window

    plt.imshow(img_to_plt, cmap=cmap, extent=img_extent, aspect='equal')

    # Set the size of the axis tick labels
    plt.tick_params(labelsize=14)

    # Set the x/y-ticks at multiples of 5 cm
    plt.gca().set_xticks([-6, -4, -2, 0, 2, 4, 6])
    plt.gca().set_yticks([-6, -4, -2, 0, 2, 4, 6])

    # Specify the colorbar tick format and size
    plt.colorbar(format=cbar_fmt).ax.tick_params(labelsize=14)

    # Set the x/y axes limits
    plt.xlim([-roi_rad, roi_rad])
    plt.ylim([-roi_rad, roi_rad])

    plt.plot(ant_xs, ant_ys, 'k--', linewidth=2.5)
    # plt.plot(breast_xs, breast_ys, 'w--', linewidth=2)
    phi = np.deg2rad(np.arange(0, 360, 0.1))
    rho = cs(phi)
    xs = rho * np.cos(phi) * 100
    ys = rho * np.sin(phi) * 100
    plt.plot(xs, ys, 'w-')

    # Plot the approximate tumor boundary
    plt.plot(tum_xs, tum_ys, 'g', linewidth=1.5)

    tar2_xs, tar2_ys = (tar2_rad * 100 * np.cos(draw_angs) + tar2_x * 100,
                        tar2_rad * 100 * np.sin(draw_angs) + tar2_y * 100)

    # Plot secondary target
    plt.plot(tar2_xs, tar2_ys, 'g', linewidth=1.5)

    plt.title(title, fontsize=20)  # Make the plot title
    plt.xlabel('x-axis (cm)', fontsize=16)  # Make the x-axis label
    plt.ylabel('y-axis (cm)', fontsize=16)  # Make the y-axis label
    plt.tight_layout()  # Remove excess whitespace in the figure

    line, = plt.plot((ant_pos_x, pix_xs[0]), (ant_pos_y, pix_ys[0]), 'g-')
    pix_dot, = plt.plot(pix_xs[0], pix_ys[0], 'r.')
    plt.plot(ant_pos_x, ant_pos_y, 'bo')
    int_f_pos, = plt.plot(int_f_xs[30, 0, 0], int_f_ys[30, 0, 0], 'rX')
    int_b_pos, = plt.plot(int_b_xs[30, 0, 0], int_b_ys[30, 0, 0], 'gX')

    class Index:
        px_x = 0
        px_y = 0

        def next_x(self, event):
            if self.px_x == 0 or self.px_x % 149 != 0:
                self.px_x += 1
            elif self.px_y < 149:
                self.px_x = 0
                self.px_y += 1
            line.set_data((ant_pos_x, pix_xs[self.px_x]),
                          (ant_pos_y, pix_ys[self.px_y]))
            pix_dot.set_data(pix_xs[self.px_x], pix_ys[self.px_y])
            int_f_pos.set_data(int_f_xs[0, self.px_y, self.px_x],
                               int_f_ys[0, self.px_y, self.px_x])
            int_b_pos.set_data(int_b_xs[0, self.px_y, self.px_x],
                               int_b_ys[0, self.px_y, self.px_x])
            plt.draw()

        def next_y(self, event):
            if self.px_y == 0 or self.px_y % 149 != 0:
                self.px_y += 1
            elif self.px_x < 149:
                self.px_y = 0
                self.px_x += 1
            line.set_data((ant_pos_x, pix_xs[self.px_x]),
                          (ant_pos_y, pix_ys[self.px_y]))
            pix_dot.set_data(pix_xs[self.px_x], pix_ys[self.px_y])
            int_f_pos.set_data(int_f_xs[0, self.px_y, self.px_x],
                               int_f_ys[0, self.px_y, self.px_x])
            int_b_pos.set_data(int_b_xs[0, self.px_y, self.px_x],
                               int_b_ys[0, self.px_y, self.px_x])
            plt.draw()

        def prev_x(self, event):
            if self.px_x != 0:
                self.px_x -= 1
            elif self.px_y > 0:
                self.px_x = 149
                self.px_y -= 1
            line.set_data((ant_pos_x, pix_xs[self.px_x]),
                          (ant_pos_y, pix_ys[self.px_y]))
            pix_dot.set_data(pix_xs[self.px_x], pix_ys[self.px_y])
            int_f_pos.set_data(int_f_xs[0, self.px_y, self.px_x],
                               int_f_ys[0, self.px_y, self.px_x])
            int_b_pos.set_data(int_b_xs[0, self.px_y, self.px_x],
                               int_b_ys[0, self.px_y, self.px_x])
            plt.draw()

        def prev_y(self, event):
            if self.px_y != 0:
                self.px_y -= 1
            elif self.px_x > 0:
                self.px_y = 149
                self.px_x -= 1
            line.set_data((ant_pos_x, pix_xs[self.px_x]),
                          (ant_pos_y, pix_ys[self.px_y]))
            pix_dot.set_data(pix_xs[self.px_x], pix_ys[self.px_y])
            int_f_pos.set_data(int_f_xs[0, self.px_y, self.px_x],
                               int_f_ys[0, self.px_y, self.px_x])
            int_b_pos.set_data(int_b_xs[0, self.px_y, self.px_x],
                               int_b_ys[0, self.px_y, self.px_x])
            plt.draw()

    callback = Index()
    axprev_x = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext_x = plt.axes([0.81, 0.05, 0.1, 0.075])
    axnext_y = plt.axes([0.59, 0.05, 0.1, 0.075])
    axprev_y = plt.axes([0.48, 0.05, 0.1, 0.075])

    bnext_x = matplotlib.widgets.Button(axnext_x, 'Next X')
    bnext_x.on_clicked(callback.next_x)
    bprev_x = matplotlib.widgets.Button(axprev_x, 'Previous X')
    bprev_x.on_clicked(callback.prev_x)

    bnext_y = matplotlib.widgets.Button(axnext_y, 'Next Y')
    bnext_y.on_clicked(callback.next_y)
    bprev_y = matplotlib.widgets.Button(axprev_y, 'Previous Y')
    bprev_y.on_clicked(callback.prev_y)

    plt.show()


def plot_fd_img_differential(img, *, cs_left=None, cs_right=None,
                             tum_x=0.0, tum_y=0.0, tum_rad=0.0,
                             adi_rad=0.0, x_shift=0.0, y_shift=0.0,
                             roi_rad=0.0, img_rad=0.0, save_str='',
                             save_fig=False, cmap='inferno', title='',
                             crop_img=True, cbar_fmt='%.1f', phantom_id='',
                             transparent=False, dpi=300, save_close=True):
    """Displays a reconstruction, making a publication-ready figure

    Parameters
    ----------
    img : array_like
        The 2D reconstruction that will be displayed
    tum_x : float
        The x-position of the tumor in the scan, in meters
    tum_y : float
        The y-position of the tumor in the scan, in meters
    tum_rad : float
        The radius of the tumor in the scan, in meters
    img_rad : float
        The radius of the image-space
    save_str : str, optional
        The complete path for saving the figure as a .png - only used if
        save_fig is True
    save_fig : bool
        If True, will save the displayed image to the location specified
        in save_str
    cmap : str
        Specifies the colormap that will be used when displaying the
        image
    title : str
        The title for the plot
    crop_img : bool
        If True, will set the values in the image to NaN outside of the
        inner antenna trajectory
    cmap : str
        Color map
    transparent : bool
        If True, will save the image with a transparent background
        (i.e., whitespace will be transparent)
    dpi : int
        The DPI to be used if saving the image
    save_close : bool
        If True, closes the figure after saving
    """
    pix_xs, pix_ys = get_xy_arrs(np.size(img, axis=0), roi_rad)
    pix_xs *= 100
    pix_ys *= 100
    img_to_plt = np.abs(img)
    img_to_plt = img * np.ones_like(img)
    img_rad *= 100
    roi_rad *= 100

    # If cropping the image at the antenna-trajectory boundary
    if crop_img:
        # Find the region inside the antenna trajectory
        roi = breastmodels.get_roi(roi_rad, np.size(img_to_plt, axis=0),
                                   img_rad)

        # Set the pixels outside of the antenna trajectory to NaN
        img_to_plt[np.logical_not(roi)] = np.NaN

    # Define angles for plot the tissue geometry
    draw_angs = np.linspace(0, 2 * np.pi, 1000)

    breast_xs, breast_ys = (adi_rad * 100 * np.cos(draw_angs),
                            adi_rad * 100 * np.sin(draw_angs))

    # Define the x/y coordinates of the approximate tumor outline
    tum_xs, tum_ys = (tum_rad * 100 * np.cos(draw_angs) + tum_x * 100,
                      tum_rad * 100 * np.sin(draw_angs) + tum_y * 100)

    img_extent = [-img_rad, img_rad, -img_rad, img_rad]

    # Set the font to times new roman
    plt.rc('font', family='Times New Roman')
    plt.figure()  # Make the figure window

    plt.imshow(img_to_plt, cmap=cmap, extent=img_extent, aspect='equal')

    # Set the size of the axis tick labels
    plt.tick_params(labelsize=14)

    # Set the x/y-ticks at multiples of 5 cm
    plt.gca().set_xticks([-6, -4, -2, 0, 2, 4, 6])
    plt.gca().set_yticks([-6, -4, -2, 0, 2, 4, 6])

    # Specify the colorbar tick format and size
    plt.colorbar(format=cbar_fmt).ax.tick_params(labelsize=14)

    # Set the x/y axes limits
    plt.xlim([-roi_rad, roi_rad])
    plt.ylim([-roi_rad, roi_rad])

    # plt.plot(ant_xs, ant_ys, 'k--', linewidth=2.5)
    plt.plot(breast_xs, breast_ys, 'w--', linewidth=2,
             label='Approximate outline')

    # Plot the approximate tumor boundary
    plt.plot(tum_xs, tum_ys, 'g', label='Observed position', linewidth=1.5)

    # plt.title(title, fontsize=20)  # Make the plot title
    plt.xlabel('x-axis (cm)', fontsize=16)  # Make the x-axis label
    plt.ylabel('y-axis (cm)', fontsize=16)  # Make the y-axis label
    plt.tight_layout()  # Remove excess whitespace in the figure

    # If saving the image, save it to the save_str path and close it
    if save_fig:

        if cs_left is not None:
            phi = np.deg2rad(np.arange(0, 360, 0.1))
            rho = cs_left(phi)
            xs = rho * np.cos(phi) * 100
            ys = rho * np.sin(phi) * 100
            plt.plot(xs, ys, 'r-', label='Left breast boundary')

        if cs_right is not None:
            phi = np.deg2rad(np.arange(0, 360, 0.1))
            rho = cs_right(phi)
            xs = rho * np.cos(phi) * 100 + x_shift * 100
            ys = rho * np.sin(phi) * 100 + y_shift * 100
            plt.plot(xs, ys, 'b-', label='Right breast boundary ')

        # plt.legend(loc='upper left')
        plt.savefig(save_str, transparent=transparent, dpi=dpi,
                    bbox_inches='tight')

        if save_close:  # If wanting to close the fig after saving
            plt.close()


def antennas_to_shifted_boundary(cs, delta_x, delta_y, ant_rad,
                                 n_ant_pos=72,
                                 ini_ant_ang=-136.0,
                                 fin_ant_ang=355.):
    """Creates a plot of a shifted and unshifted boundaries with
    corresponding antenna positions

    """

    plot_angs = np.deg2rad(np.arange(0, 360, 0.1))
    plot_rhos = cs(plot_angs)

    plot_xs = plot_rhos * np.cos(plot_angs)
    plot_ys = plot_rhos * np.sin(plot_angs)

    plot_xs_shifted = plot_xs + delta_x
    plot_ys_shifted = plot_ys + delta_y
    plt.plot(plot_xs, plot_ys, 'k-', linewidth=1)
    plt.plot(plot_xs_shifted, plot_ys_shifted, 'b--', linewidth=1)
    ant_angs = np.linspace(0, np.deg2rad(fin_ant_ang),
                           n_ant_pos) + np.deg2rad(ini_ant_ang)

    xs = cs(ant_angs) * np.cos(ant_angs)
    ys = cs(ant_angs) * np.sin(ant_angs)

    ant_angs = np.flip(ant_angs)

    ant_xs = np.cos(ant_angs) * ant_rad
    ant_ys = np.sin(ant_angs) * ant_rad
    plt.scatter(ant_xs, ant_ys, marker='x')
    plt.plot(ant_xs[4], ant_ys[4], 'ro')
    plt.plot(ant_xs[52], ant_ys[52], 'ro')
    # plt.plot(xs, ys, 'b.-')

    xs_shifted = xs + delta_x
    ys_shifted = ys + delta_y

    for ant_pos in [4, 52]:

        plt.plot(ant_xs[ant_pos], ant_ys[ant_pos], 'ro')

        unshifted_idx = np.argmin(np.sqrt((ant_xs[ant_pos] - plot_xs) ** 2 +
                                          (ant_ys[ant_pos] - plot_ys) ** 2))

        unshifted_x = plot_xs[unshifted_idx]
        unshifted_y = plot_ys[unshifted_idx]

        closest_unshifted_idx = np.argmin(np.sqrt((unshifted_x - xs) ** 2 +
                                                  (unshifted_y - ys) ** 2))

        closest_unshifted_idxs = [(closest_unshifted_idx - 1) % n_ant_pos,
                                  closest_unshifted_idx,
                                  (closest_unshifted_idx + 1) % n_ant_pos]

        plt.plot(xs[closest_unshifted_idxs], ys[closest_unshifted_idxs], 'k.')

        shifted_idx = np.argmin(np.sqrt((ant_xs[ant_pos] -
                                         plot_xs_shifted) ** 2 +
                                        (ant_ys[ant_pos] -
                                         plot_ys_shifted) ** 2))

        shifted_x = plot_xs_shifted[shifted_idx]
        shifted_y = plot_ys_shifted[shifted_idx]

        closest_shifted_idx = np.argmin(np.sqrt((shifted_x - xs_shifted) ** 2 +
                                                (shifted_y - ys_shifted) ** 2))

        closest_shifted_idxs = [(closest_shifted_idx - 1) % n_ant_pos,
                                closest_shifted_idx,
                                (closest_shifted_idx + 1) % n_ant_pos]

        plt.plot(xs_shifted[closest_shifted_idxs],
                 ys_shifted[closest_shifted_idxs], 'r.')

        plt.plot((ant_xs[ant_pos], unshifted_x),
                 (ant_ys[ant_pos], unshifted_y),
                 'b-', linewidth=0.5)
        plt.plot(unshifted_x, unshifted_y, 'b.')

        plt.plot((ant_xs[ant_pos], plot_xs_shifted[shifted_idx]),
                 (ant_ys[ant_pos], plot_ys_shifted[shifted_idx]),
                 'r-', linewidth=0.5)
        plt.plot(plot_xs_shifted[shifted_idx], plot_ys_shifted[shifted_idx],
                 'r.')

    plt.axis('square')
    plt.show()
