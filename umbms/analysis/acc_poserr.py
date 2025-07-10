"""
Tyson Reimer
University of Manitoba
December 15th, 2022
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# matplotlib.use('Qt5Agg')
from matplotlib.cm import get_cmap

from scipy.stats import spearmanr

from scipy.optimize import minimize

import seaborn as sns

from umbms import get_proj_path, verify_path

from umbms.loadsave import save_pickle
from umbms.analysis.acc_size import get_img_max_xy, get_img_CoM


###############################################################################

__OUT_DIR = os.path.join(get_proj_path(), "output/iqms-dec/pos-err/")
verify_path(__OUT_DIR)
verify_path(os.path.join(__OUT_DIR, "regular_das/"))
verify_path(os.path.join(__OUT_DIR, "binary_das/"))
verify_path(os.path.join(__OUT_DIR, "freq_dep_non_cond_das/"))
verify_path(os.path.join(__OUT_DIR, "freq_dep_das/"))
verify_path(os.path.join(__OUT_DIR, "rt_das_das/"))

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
    """Init plot window"""

    fig = plt.figure(figsize=(12, 8))
    plt.rc("font", family="Libertinus Serif")
    plt.tick_params(labelsize=16)


def plot_pos_errs(
    imgs, xs, ys, compare_xs, compare_ys, roi_rad, save_str, out_dir
):
    """Produce plots of observed-vs-reconstructed target positions & err

    Parameters
    ----------
    imgs : array_like
        DAS reconstructions
    xs : array_like
        x positions of target in each scan, in [cm]
    ys : array_like
        y positions of target in each scan, in [cm]
    roi_rad : float
        The ROI radius, in units of [cm]
    save_str : str
        Str for saving figs

    Returns
    -------
    max_xs : array_like
        x positions of target in each reconstruction, in [cm]
    max_ys : array_like
        y positions of target in each reconstruction, in [cm]
    """

    # Init arrays for storing img xs/ys (max) and errors
    x_errs = np.zeros_like(xs)
    y_errs = np.zeros_like(xs)
    phi_errs = np.zeros_like(xs)

    for ii in range(np.size(imgs, axis=0)):  # For each img
        # Calculate the x/y/phi errors
        x_errs[ii] = compare_xs[ii] - xs[ii]
        y_errs[ii] = compare_ys[ii] - ys[ii]
        phi_errs[ii] = np.rad2deg(
            np.arctan2(ys[ii], xs[ii])
            - np.arctan2(compare_ys[ii], compare_xs[ii])
        )

    init_plt()  # Plot observed vs recon target positions
    plt.scatter(
        compare_xs,
        compare_ys,
        color="k",
        marker="o",
        label="Reconstructed Positions",
        s=15,
    )
    plt.scatter(xs, ys, color="b", marker="x", label="Observed Positions")
    for ii in range(len(compare_xs)):
        plt.plot([compare_xs[ii], xs[ii]], [compare_ys[ii], ys[ii]], "k-")
    plt.xlabel("x-position (cm)", fontsize=20)
    plt.ylabel("y-position (cm)", fontsize=20)
    plt.legend(fontsize="18")
    plt.xlim([-roi_rad, roi_rad])
    plt.ylim([-roi_rad, roi_rad])
    plt.axis("square")
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig(
        os.path.join(out_dir, "%s_recon_vs_obs_pos.png" % save_str),
        dpi=300,
        transparent=True,
    )
    plt.savefig(
        os.path.join(out_dir, "NT_%s_recon_vs_obs_pos.png" % save_str),
        dpi=300,
        transparent=False,
    )

    init_plt()  # Plot x errors
    plt.scatter(
        x=np.arange(len(x_errs)),
        y=x_errs,
        color="k",
        marker="o",
        label="x-differences",
    )
    plt.xlabel("Experiment Number", fontsize=20)
    plt.ylabel("x-Position Difference (mm)", fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig(
        os.path.join(out_dir, "%s_x_diffs.png" % save_str),
        dpi=300,
        transparent=True,
    )
    plt.savefig(
        os.path.join(out_dir, "NT_%s_x_diffs.png" % save_str),
        dpi=300,
        transparent=False,
    )

    init_plt()  # Plot y errors
    plt.scatter(
        x=np.arange(len(y_errs)),
        y=y_errs,
        color="b",
        marker="x",
        label="y-differences",
    )
    plt.xlabel("Experiment Number", fontsize=20)
    plt.ylabel("y-Position Difference (mm)", fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig(
        os.path.join(out_dir, "%s_y_diffs.png" % save_str),
        dpi=300,
        transparent=True,
    )
    plt.savefig(
        os.path.join(out_dir, "NT_%s_y_diffs.png" % save_str),
        dpi=300,
        transparent=False,
    )

    init_plt()  # Plot phi errors
    plt.scatter(
        x=np.arange(len(y_errs)),
        y=phi_errs,
        color="b",
        marker="x",
        label="y-differences",
    )
    plt.xlabel("Experiment Number", fontsize=20)
    plt.ylabel(r"Phi Difference ($^{\circ}$)", fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig(
        os.path.join(out_dir, "%s_phi_diffs.png" % save_str),
        dpi=300,
        transparent=True,
    )
    plt.savefig(
        os.path.join(out_dir, "NT_%s_phi_diffs.png" % save_str),
        dpi=300,
        transparent=False,
    )


def plot_obs_vs_recon(
    xs,
    ys,
    compare_xs,
    compare_ys,
    save_str,
    o_dir_str,
    x_lim_lhs,
    x_lim_rhs,
    y_lim_bot,
    y_lim_top,
):
    out_dir = os.path.join(__OUT_DIR, "%s" % o_dir_str)
    verify_path(out_dir)

    init_plt()  # Plot observed vs recon target positions
    plt.scatter(
        compare_xs,
        compare_ys,
        color="k",
        marker="o",
        label="Reconstructed Positions",
        s=15,
    )
    plt.scatter(xs, ys, color="b", marker="x", label="Observed Positions")
    for ii in range(len(compare_xs)):
        plt.plot([compare_xs[ii], xs[ii]], [compare_ys[ii], ys[ii]], "k-")
    plt.xlabel("x-position (cm)", fontsize=20)
    plt.ylabel("y-position (cm)", fontsize=20)
    plt.legend(fontsize="18")
    plt.xlim([x_lim_lhs, x_lim_rhs])
    plt.ylim([y_lim_bot, y_lim_top])
    # plt.axis('square')
    plt.grid()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "NT_%s_recon_vs_obs_pos.png" % save_str),
        dpi=300,
        transparent=False,
    )


def apply_syst_cor(xs, ys, x_err, y_err, phi_err):
    """Apply systematic error correction
    Parameters
    ----------
    xs : array_like
        Observed x positions of target, in [cm]
    ys : array_like
        Observed y positions of target, in [cm]
    x_err : float
        Systematic x error in observed target position, in [cm]
    y_err : float
        Systematic y error in observed target position, in [cm]
    phi_err
        Systematic phi error in observed target position, in [deg]
    Returns
    -------
    cor_xs2 : array_like
        Corrected observed x positions of target, after applying
        correction for systematic errors, in [cm]
    cor_ys2 : array_like
        Corrected observed x positions of target, after applying
        correction for systematic errors, in [cm]
    """
    cor_xs = xs - x_err
    cor_ys = ys - y_err
    cor_xs2 = cor_xs * np.cos(np.deg2rad(phi_err)) - cor_ys * np.sin(
        np.deg2rad(phi_err)
    )
    cor_ys2 = cor_xs * np.sin(np.deg2rad(phi_err)) + cor_ys * np.cos(
        np.deg2rad(phi_err)
    )

    return cor_xs2, cor_ys2


def position_compare(syst_errs, xs, ys, max_xs, max_ys):
    """Cost function - total error in observed vs calc target positions

    Parameters
    ----------
    x_err, y_err : float
        The systematic error in the x/y directions, in [cm]
    phi_err : float
        The systematic rotation error in phi direction, in [deg]
    xs : array_like
        The observed target x positions, in [cm]
    ys : array_like
        The observed target y positions, in [cm]
    max_xs : array_like
        The image target x positions, in [cm]
    max_ys : array_like
        The image target y positions, in [cm]

    Returns
    -------
    pos_diff : float
        The sum of the square differences between the observed
        and reconstructed target positions
    """

    x_err, y_err, phi_err = syst_errs[0], syst_errs[1], syst_errs[2]

    # Apply systematic corrections to 'measured' x,y positions
    cor_xs, cor_ys = apply_syst_cor(xs, ys, x_err, y_err, phi_err)

    # Calculate the total squared difference in position between
    # the observed and reconstructed target positions
    pos_diff = np.sum(np.sqrt((cor_xs - max_xs) ** 2 + (cor_ys - max_ys) ** 2))

    return pos_diff


def find_systematic_errs(max_xs, max_ys, xs, ys):
    """Finds the systematic error in the x/y/phi directions

    Parameters
    ----------
    max_xs : array_like
        The reconstructed target x positions, in [cm]
    max_ys : array_like
        The reconstructed target y positions, in [cm]
    xs : array_like
        The observed target x positions, in [cm]
    ys : array_like
        The observed target y positions, in [cm]

    Returns
    -------
    x_err : float
        The systematic error in the x directions, in [cm]
    y_err : float
        The systematic error in the y directions, in [cm]
    phi_err : float
        The systematic error in the x,y directions, in [deg]
    """

    # Find the x_err, y_err, phi_err that minimizes the sum-of-square
    # differences between the observed and reconstructed target
    # positions
    opt_res = minimize(
        position_compare,
        x0=np.array([0, 0, 0]),
        args=(xs, ys, max_xs, max_ys),
        bounds=[(-5, 5), (-5, 5), (-20, 20)],
    )
    x_err = opt_res["x"][0]
    y_err = opt_res["x"][1]
    phi_err = opt_res["x"][2]

    return x_err, y_err, phi_err


def do_pos_err_analysis(
    imgs,
    tar_xs,
    tar_ys,
    logger,
    roi_rad,
    o_dir_str="",
    save_str="",
    use_img_maxes=True,
    make_plts=True,
):
    """Do position error analysis

    Parameters
    ----------
    imgs : array_like
        Array of reconstructed images
    tar_xs : array_like
        Target x positions, in units of [m]
    tar_ys : array_like
        Target y positions, in units of [m]
    o_dir_str : str
        The output directory for this run
    save_str : str
        Specific save-string to be used for saving figs
    use_img_maxes : bool
        If True, uses the max intensity pixel as the 'target response
        location' in the image. If False, uses the center of mass
        as the response location in the image.

    Returns
    -------
    img_c_xs : array_like
        Corrected target x-positions
    img_c_ys : arrray_like
        Corrrected target y-positions
    img_xs : array_like

    img_max_ys
    """
    if make_plts:
        # Make the output directory for this run
        out_dir = os.path.join(__OUT_DIR, "%s" % o_dir_str)
        verify_path(out_dir)

    # Init arrays for storing the target positions *in the image space*
    img_xs = np.zeros(
        [
            np.size(imgs, axis=0),
        ]
    )
    img_ys = np.zeros_like(img_xs)

    # For each image, find the target x/y position
    for ii in range(np.size(imgs, axis=0)):
        if use_img_maxes:  # If using image maxes...
            img_xs[ii], img_ys[ii] = get_img_max_xy(
                img=imgs[ii], img_rad=roi_rad
            )
        else:  # If using image center of masses...
            img_xs[ii], img_ys[ii] = get_img_CoM(img=imgs[ii], img_rad=roi_rad)

    # TEMPORARY BELOW --------------------------------------------------
    if "DAS" in save_str:
        print(save_str)
        if use_img_maxes:
            save_pickle(
                (img_xs, img_ys),
                os.path.join(__OUT_DIR, "img_coords_MAX.pickle"),
            )
        else:
            save_pickle(
                (img_xs, img_ys),
                os.path.join(__OUT_DIR, "img_coords_CoM.pickle"),
            )
        save_pickle(
            (tar_xs, tar_ys), os.path.join(__OUT_DIR, "meas_coords.pickle")
        )
    # TEMPORARY ABOVE --------------------------------------------------

    if make_plts:
        # Do analysis
        plot_pos_errs(
            imgs=imgs,
            xs=tar_xs,
            ys=tar_ys,
            compare_xs=img_xs,
            compare_ys=img_ys,
            roi_rad=roi_rad,
            save_str="%s_UNCORRECTED" % save_str,
            out_dir=out_dir,
        )

    # img_x_err, img_y_err, img_phi_err = find_systematic_errs(max_xs=img_xs,
    #                                                          max_ys=img_ys,
    #                                                          xs=tar_xs,
    #                                                          ys=tar_ys)
    #
    # logger.info('\t%s...' % save_str.upper())
    # logger.info('\t\tx-err:\t\t%.3f mm' % (10 * img_x_err))
    # logger.info('\t\ty-err:\t\t%.3f mm' % (10 * img_y_err))
    # logger.info('\t\tphi-err:\t\t%.3f deg' % (img_phi_err))

    img_c_xs, img_c_ys = img_xs, img_ys

    if make_plts:
        plot_pos_errs(
            imgs=imgs,
            xs=tar_xs,
            ys=tar_ys,
            compare_xs=img_c_xs,
            compare_ys=img_c_ys,
            roi_rad=roi_rad,
            save_str="%s_CORRECTED" % save_str,
            out_dir=out_dir,
        )

    plt.close("all")

    return img_c_xs, img_c_ys, img_xs, img_ys


def plt_rand_pos_errs(
    c_xs, c_ys, xs, ys, o_dir_str, logger, save_str, make_plts=True
):
    # Make the output directory for this run
    out_dir = os.path.join(__OUT_DIR, "%s" % o_dir_str)
    verify_path(out_dir)

    pos_errs = np.sqrt((c_xs - xs) ** 2 + (c_ys - ys) ** 2)

    plt.close("all")
    viridis = get_cmap("viridis")
    spearman_val, pval = spearmanr(a=np.sqrt(c_xs**2 + c_ys**2), b=pos_errs)
    logger.info("%s" % save_str)
    logger.info("\tSpearman coefficient: %.4f" % spearman_val)
    logger.info("\tp-value: %.3e" % pval)

    if make_plts:
        init_plt()
        plt.scatter(
            10 * np.sqrt(c_xs**2 + c_ys**2),
            10 * pos_errs,
            marker="o",
            color=viridis(0),
        )
        plt.xlabel(
            r"Polar Radius of Target $\mathdefault{\rho}$ (mm)", fontsize=22
        )
        plt.ylabel("Target Positioning Error (mm)", fontsize=22)
        plt.tight_layout()
        # plt.show()
        plt.savefig(
            os.path.join(out_dir, "%s_posErr_vs_PolarRho.png" % save_str),
            dpi=300,
            transparent=True,
        )
        plt.savefig(
            os.path.join(out_dir, "NT_%s_posErr_vs_PolarRho.png" % save_str),
            dpi=300,
            transparent=False,
        )

        # Plot x-error distribution...
        init_plt()
        sns.histplot(
            10 * (xs - c_xs),
            label="DAS",
            color=viridis(0.0),
            alpha=0.3,
            kde=True,
        )
        plt.axvline(x=10 * np.mean(xs - c_xs), color=viridis(0), linestyle="--")
        plt.xlabel("x-Error (mm)", fontsize=22)
        plt.ylabel("Count / KDE", fontsize=22)
        plt.tight_layout()
        # plt.show()
        plt.savefig(
            os.path.join(out_dir, "%s_xError_Distros.png" % save_str),
            dpi=300,
            transparent=True,
        )
        plt.savefig(
            os.path.join(out_dir, "NT_%s_xError_Distros.png" % save_str),
            dpi=300,
            transparent=False,
        )

        # Plot y-error distribution below...
        init_plt()
        sns.histplot(
            10 * (ys - c_ys),
            label="DAS",
            color=viridis(0.0),
            alpha=0.3,
            kde=True,
        )
        plt.axvline(x=10 * np.mean(ys - c_ys), color=viridis(0), linestyle="--")
        plt.xlabel("y-Error (mm)", fontsize=22)
        plt.ylabel("Count / KDE", fontsize=22)
        plt.tight_layout()
        # plt.show()
        plt.savefig(
            os.path.join(out_dir, "%s_yError_Distros.png" % save_str),
            dpi=300,
            transparent=True,
        )
        plt.savefig(
            os.path.join(out_dir, "NT_%s_yError_Distros.png" % save_str),
            dpi=300,
            transparent=False,
        )

        init_plt()
        sns.histplot(
            10 * pos_errs, label="DAS", color=viridis(0.0), alpha=0.3, kde=True
        )
        plt.axvline(x=10 * np.mean(pos_errs), color=viridis(0), linestyle="-")
        plt.xlabel("Localization Error (mm)", fontsize=22)
        plt.ylabel("Count", fontsize=22)
        plt.tight_layout()
        # plt.show()
        plt.savefig(
            os.path.join(out_dir, "%s_posError_Distros.png" % save_str),
            dpi=300,
            transparent=True,
        )
        plt.savefig(
            os.path.join(out_dir, "NT_%s_posError_Distros.png" % save_str),
            dpi=300,
            transparent=False,
        )

    mean_pos_err = np.mean(pos_errs) * 10  # Mean pos err in [mm]
    pos_err_95 = np.percentile(pos_errs, 95) * 10  # 95th perc pos err

    logger.info("\t\tMean pos error: %.3f mm" % mean_pos_err)
    logger.info("\t\t95%% pos error: %.3f mm" % pos_err_95)


###############################################################################


def get_loc_err_max_old(img, ant_rad, tum_x, tum_y):
    """Return the localization error of the tumor response in the image

    Compute the localization error for the reconstructed image in meters

    Parameters
    ----------
    img : array_like
        The reconstructed image
    ant_rad : float
        The radius of the antenna trajectory during the scan, in meters
    tum_x : float
        The x-position of the tumor during the scan, in meters
    tum_y : float
        The y-position of the tumor during the scan, in meters

    Returns
    -------
    loc_err : float
        The localization error in meters
    """

    # Rotate image to properly compute the distances
    img_for_iqm = np.fliplr(img)

    # Find the conversion factor to convert pixel index to distance
    pix_to_dist = 2 * ant_rad / np.size(img, 0)

    # Set any nan values to zero
    img_for_iqm[np.isnan(img_for_iqm)] = 0

    # Find the index of the maximum response in the reconstruction
    max_loc = np.argmax(img_for_iqm)

    # Find the x/y-indices of the max response in the reconstruction
    max_x_pix, max_y_pix = np.unravel_index(max_loc, np.shape(img))

    # Convert this to the x/y-positions
    max_x_pos = (max_x_pix - np.size(img, 0) // 2) * pix_to_dist
    max_y_pos = (max_y_pix - np.size(img, 0) // 2) * pix_to_dist

    # Compute the localization error
    loc_err = np.sqrt((max_x_pos - tum_x) ** 2 + (max_y_pos - tum_y) ** 2)

    return loc_err
