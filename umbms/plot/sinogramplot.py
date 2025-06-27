"""
Tyson Reimer
University of Manitoba
June 27th, 2019
"""

import os
from umbms.plot import init_plt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from umbms import verify_path
from umbms.beamform.iczt import iczt

# matplotlib.use('Agg')
import numpy as np

###############################################################################

# Conversion factor from [GHz] to [Hz]
__GHz_to_Hz = 1e9


###############################################################################


def plot_sino(
    td_data,
    ini_t,
    fin_t,
    title="",
    save_fig=False,
    save_str="",
    normalize=False,
    normalize_values=(0, 1),
    cbar_fmt="%.4f",
    cmap="inferno",
    transparent=True,
    dpi=300,
    close_save=True,
):
    """Plots a time-domain sinogram

    Displays a sinogram in the time domain.

    Parameters
    ----------
    td_data_to_plt : array_like
        Raw data in the time-domain.
    ini_t : float
        The initial time-point in the time-domain, in seconds
    fin_t : float
        The final time-point in the time-domain, in seconds
    title : str
        The title used for displaying the raw data
    save_fig : bool
        If true, will save the figure to a .png file - default is False
    save_str : str
        The title of the .png file to be saved if save_image is true -
        default is empty str, triggering the save_string to be set to
        the title
    normalize : bool
        If set to True, will normalize the colorbar intensity to the
        specified value (default is to the maximum value in the image)
    normalize_values : tuple
        Default set to (0, 1) will cause the colorbar to be normalized
        to the maximum value in the image, that is, the image values
        will be scaled to have maximum 1. If set to any other value,
        the image values will not change, but the colorbar will be
        scaled to have (min, max) values of the two values in the tuple
    cbar_fmt : str
        The format specifier to be used for the colorbar, default
        is '%.3f'
    cmap : str
        The colormap that will be used to display the sinogram
    transparent : bool
        If True, will save the figure when True, else will save
        with default white background
    dpi : int
        The DPI to use if saving the figure
    close_save : bool
        If True, will close the figure after saving it.
    """

    td_data_to_plt = np.abs(td_data)

    n_time_pts = np.size(td_data_to_plt, axis=0)

    # Find the vector of temporal points in the time domain used in the
    # scan
    scan_times = np.linspace(ini_t, fin_t, n_time_pts)

    # Declare the extent for the plot, along x-axis from antenna
    # angle from 0 deg to 355 deg, along y-axis from user specified
    # points in time-domain
    plot_extent = [0, 355, scan_times[-1] * 1e9, scan_times[0] * 1e9]

    # Determine the aspect ratio for the plot to make it have equal axes
    plot_aspect_ratio = 355 / (scan_times[-1] * 1e9)

    # Make the figure for displaying the reconstruction
    plt.figure()

    # Declare the default font for our figure to be Times New Roman
    plt.rc("font", family="Times New Roman")

    # If the user wanted to normalize the data, make our imshow() and
    # set the colorbar() bounds using vmin, vmax
    if normalize:
        # Assert that the normalize values are of the form
        # (cbar_min, cbar_max)
        assert len(normalize_values) == 2, (
            "Normalize values must be 2-element tuple of form "
            "(cbar_min, cbar_max)"
        )
        assert normalize_values[1] > normalize_values[0], (
            "Normalize values must be 2-element tuple of the form "
            "(cbar_min, cbar_max)"
        )

        # If the normalize value is default, indicating to normalize to
        # the maximum value in the reconstruction
        if normalize_values == (0, 1):
            # Map values in reconstructed image to be over range (0, 1)
            td_data_to_plt -= np.min(td_data_to_plt)
            td_data_to_plt /= np.max(td_data_to_plt)

        # Plot the data, using the using the normalize_values as
        # the colorbar bounds
        plt.imshow(
            td_data_to_plt,
            cmap=cmap,
            extent=plot_extent,
            vmin=normalize_values[0],
            vmax=normalize_values[1],
            aspect=plot_aspect_ratio,
        )

    # If user did not want to normalize the data, do not set
    # the colorbar() bounds
    else:
        plt.imshow(
            td_data_to_plt,
            cmap=cmap,
            extent=plot_extent,
            aspect=plot_aspect_ratio,
        )

    # Set the size for the x,y ticks and set which ticks to display
    plt.tick_params(labelsize=14)
    scan_times *= 1e9
    plt.gca().set_yticks(
        [round(ii, 2) for ii in scan_times[:: n_time_pts // 8]]
    )
    plt.gca().set_xticks([round(ii) for ii in np.linspace(0, 355, 355)[::75]])

    # Create the colorbar and set the colorbar tick size, also set
    # the format specifier to be as entered by user - default is '%.3f'
    cbar = plt.colorbar(format=cbar_fmt)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(
        r"|$\mathdefault{S_{11}}$|", rotation=270, fontsize=16, labelpad=20
    )

    # Label the plot axes and assign the plot a title
    plt.title(title, fontsize=20)
    plt.xlabel(
        "Polar Angle of Antenna Position (" + r"$^\circ$" + ")", fontsize=16
    )
    plt.ylabel("Time of Response (ns)", fontsize=16)
    plt.tight_layout()

    # If the user set save_data to True, and therefore wanted to save
    # the figure as a png file
    if save_fig:
        # If the user did not specify a save string, then set the
        # save string to be the title string, without spaces
        if not save_str:
            # Define a string for saving the figure, replace any spaces
            # in the title with underscores
            save_str = title.replace(" ", "_") + ".png"

        # If the user did specify a save string, then add '.png' file
        # extension to it for saving purposes
        else:
            save_str += ".png"

        # Save the figure to a png file
        plt.savefig(save_str, transparent=transparent, dpi=dpi)
        if close_save:
            plt.close()

    else:  # If not saving the figure, then show the figure
        plt.show()


def plt_sino(
    fd,
    title,
    save_str,
    out_dir,
    cbar_fmt="%.2e",
    transparent=True,
    close=True,
    slices=False,
):
    """Plots a time-domain sinogram

    Parameters:
    ----------------
    fd : array_like
       Frequency-domain data
    title : sting
        Plot title
    save_str : string
        Name of the figure
    out_dir : string
        Directory for saving
    cbar_fmt : string
        Colorbar format
    transparent : bool
        Transparent background
    close : bool
        Close figure flag
    slices : bool
        Slices flag
    """

    # Find the minimum retained frequency
    scan_fs = np.linspace(1e9, 9e9, 1001)  # Frequencies used in scan
    min_f = 2e9  # Min frequency to retain
    tar_fs = scan_fs >= min_f  # Target frequencies to retain
    min_retain_f = np.min(scan_fs[tar_fs])  # Min freq actually retained

    # Create variables for plotting
    ts = np.linspace(0.5, 5.5, 700)
    plt_extent = [0, 355, ts[-1], ts[0]]
    plt_aspect_ratio = 355 / ts[-1]

    # Conert to the time-domain
    td = iczt(
        fd,
        ini_t=0.5e-9,
        fin_t=5.5e-9,
        n_time_pts=700,
        ini_f=min_retain_f,
        fin_f=9e9,
    )

    # If the user wants to plot separate slices of the sinogram
    # (individual antenna positions)
    if slices:
        # Iterating over antenna positions
        for ant_idx in range(np.size(td, axis=1)):
            # Create a separate folder for the slices
            # take away the .png at the end
            slice_dir = save_str[:-4] + "_slices/"
            to_save_dir = os.path.join(out_dir, slice_dir)
            verify_path(to_save_dir)

            plt.figure(figsize=(1000 / 120, 800 / 120), dpi=120)

            plt.rc("font", family="Libertinus Serif")
            # Plotting
            plt.plot(ts, np.abs(td[:, ant_idx]), "k-", linewidth=1.6)
            # plt.title("Antenna index #%d" % ant_idx, fontsize=15)
            plt.grid("-")
            plt.xlabel("Time of response (ns)", fontsize=16)
            plt.ylabel("Intensity (a.u.)", fontsize=16)

            # Saving
            plt.tight_layout()
            plt.savefig(
                os.path.join(to_save_dir, "slice_%d.png" % ant_idx), dpi=120
            )
            plt.close()

    show_sinogram(
        data=td,
        aspect_ratio=plt_aspect_ratio,
        extent=plt_extent,
        title=title,
        out_dir=out_dir,
        save_str=save_str,
        ts=ts,
        cbar_fmt=cbar_fmt,
        transparent=transparent,
        close=close,
    )


def plt_fd_sino(
    fd, title, save_str, out_dir, cbar_fmt="%.2e", transparent=True, close=True
):
    # Find the minimum retained frequency
    scan_fs = np.linspace(2e9, 9e9, 1001)  # Frequencies used in scan

    # Create variables for plotting
    fs = scan_fs
    plt_extent = [0, 355, fs[-1] / 1e9, fs[0] / 1e9]
    plt_aspect_ratio = 355 / (fs[-1] / 1e9)

    # Plot primary scatter forward projection only
    plt.figure()
    # init_plt()
    plt.rc("font", family="Libertinus Serif")
    plt.imshow(
        np.abs(fd), aspect=plt_aspect_ratio, cmap="inferno", extent=plt_extent
    )
    plt.colorbar(format=cbar_fmt).ax.tick_params(labelsize=16)
    plt.tick_params(labelsize=14)
    plt.gca().set_yticks([2, 3, 4, 5, 6, 7, 8])
    plt.gca().set_xticks([round(ii) for ii in np.linspace(0, 355, 355)[::75]])
    plt.title("%s" % title, fontsize=20)
    plt.xlabel(
        "Polar Angle of Antenna Position (" + r"$^\circ$" + ")", fontsize=16
    )
    plt.ylabel("Frequency (GHz)", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "%s" % save_str), dpi=300, transparent=transparent
    )
    if close:
        plt.close()


def show_sinogram(
    data,
    aspect_ratio,
    extent,
    title,
    out_dir,
    save_str,
    ts=None,
    bound_angles=None,
    bound_times=None,
    bound_color="r",
    cbar_fmt="%.2e",
    transparent=True,
    close=True,
):
    """Calls imshow function on provided data, formats the plot,
    and saves it

    Parameters
    ----------
    data : array_like
        Frequency- or time-domain data to plot
    aspect_ratio : float
        The aspect ratio for the plot
    extent : array_like
        x- and y-extent array for the plot
    title : string
        Title string that will appear on top of the sinogram
    out_dir : string
        Directory to save in
    save_str : string
        Filename of the sinogram
    ts : array_like
        Array of time points for TD plotting
    bound_angles : array_like
        Angle data for plotting the boundary
    bound_times : array_like
        Time of response data for plotting the boundary
    bound_color : string
        Color of the boundary on top of the sinogram
    cbar_fmt : string
        Numerical format
    transparent : bool
        Transparency flag
    close : bool
        Flag to close the plot after exiting the function
    """

    # Plot primary scatter forward projection only
    plt.figure()
    plt.rc("font", family="Libertinus Serif")
    plt.imshow(np.abs(data), aspect=aspect_ratio, cmap="inferno", extent=extent)
    plt.colorbar(format=cbar_fmt).ax.tick_params(labelsize=16)

    if ts is not None:
        plt.gca().set_yticks([round(ii, 2) for ii in ts[:: np.size(ts) // 8]])
        plt.ylabel("Time of Response (ns)", fontsize=16)
    else:
        # TODO: pass the fd values instead of hardcoding
        plt.gca().set_yticks([2, 3, 4, 5, 6, 7, 8])
        plt.ylabel("Frequency (GHz)", fontsize=16)

    plt.gca().set_xticks([round(ii) for ii in np.linspace(0, 355, 355)[::75]])
    plt.title("%s" % title, fontsize=20)
    plt.xlabel(
        "Polar Angle of Antenna Position (" + r"$^\circ$" + ")", fontsize=16
    )

    if bound_angles is not None and bound_times is not None:
        plt.plot(
            bound_angles,
            bound_times,
            "%s-" % bound_color,
            linewidth=1,
            label="Boundary",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "%s" % save_str), dpi=300, transparent=transparent
    )

    if close:
        plt.close()
