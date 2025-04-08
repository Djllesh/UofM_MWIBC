"""
Illia Prykhodko
University of Manitoba
September 13th, 2022
"""

import os
import numpy as np
import pandas
import multiprocessing as mp
import scipy.constants

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle

from umbms.plot.imgplots import plot_fd_img

from umbms.beamform.das import fd_das
from umbms.beamform.time_delay import get_pix_ts_old
from umbms.beamform.utility import apply_ant_t_delay, get_fd_phase_factor

from umbms.boundary.boundary_detection import (
    find_boundary,
    polar_fit_cs,
    cart_to_polar,
)

from umbms.beamform.propspeed import estimate_speed, get_breast_speed_freq

###############################################################################

__CPU_COUNT = mp.cpu_count()

__DATA_DIR = os.path.join(get_proj_path(), "data/umbmid/cyl_phantom/")
__OUT_DIR = os.path.join(get_proj_path(), "output/cyl_phantom/")
verify_path(__OUT_DIR)
__FITTED_DATA_DIR = os.path.join(get_proj_path(), "data/freq_data/")

__FD_NAME = "cyl_phantom_diag_s11.pickle"
__MD_NAME = "metadata_cyl_phantom_diag.pickle"
__FITTED_NAME = "Fitted Dielectric Measurements Glycerin.csv"

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

# The approximate radius of each adipose phantom in our array
__ADI_RADS = {
    "A1": 0.05,
    "A2": 0.06,
    "A3": 0.07,
    "A11": 0.06,
    "A12": 0.05,
    "A13": 0.065,
    "A14": 0.06,
    "A15": 0.055,
    "A16": 0.07,
}

__GLASS_CYLINDER_RAD = 0.06
__PHANTOM_RAD = 0.0555
__GLASS_THIKNESS = 0.003
__SPHERE_RAD = 0.0075
__ROD_RAD = 0.002
__ANT_RAD = 0.21

__SPHERE_POS = [
    (np.nan, np.nan),
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
    (np.nan, np.nan),
]

__MID_BREAST_RADS = {
    "A1": (0.053, 0.034),
    "A2": (0.055, 0.051),
    "A3": (0.07, 0.049),
    "A11": (0.062, 0.038),
    "A12": (0.051, 0.049),
    "A13": (0.065, 0.042),
    "A14": (0.061, 0.051),
    "A15": (0.06, 0.058),
    "A16": (0.073, 0.05),
}

# Define propagation speed in vacuum
__VAC_SPEED = scipy.constants.speed_of_light


def load_data():
    """Loads both fd_data and metadata

    Returns:
    --------
    tuple of two loaded variables
    """
    return load_pickle(os.path.join(__DATA_DIR, __FD_NAME)), load_pickle(
        os.path.join(__DATA_DIR, __MD_NAME)
    )


def get_middle_td(pix_ts):
    size = np.size(pix_ts, axis=0)
    idx = np.size(pix_ts, axis=1) // 2 - 1
    output = np.zeros(size)
    for i in range(size):
        output[i] = pix_ts[i, idx, idx]
    return output


###############################################################################


if __name__ == "__main__":
    logger = get_script_logger(__file__)

    # Load the frequency domain data and metadata

    fd_data, metadata = load_data()

    n_expts = np.size(fd_data, axis=0)  # The number of individual scans

    # Get the unique ID of each experiment / scan
    expt_ids = [md["id"] for md in metadata]

    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

    # Only retain frequencies above 2 GHz due to antenna VSWR
    tar_fs = scan_fs >= 2e9

    # Read .csv file of fitted values of permittivity and conductivity
    df = pandas.read_csv(os.path.join(__FITTED_DATA_DIR, __FITTED_NAME))

    # Calculate velocity array
    freqs = np.array(df["Freqs"].values, dtype=float)
    permittivities = np.array(df["Permittivity"].values)
    conductivities = np.array(df["Conductivity"].values)
    # conductivities = np.zeros_like(conductivities)
    velocities = get_breast_speed_freq(freqs, permittivities, conductivities)

    # Determine fibroglandular percentage
    fibr_perc = 0

    # initialize shared worker pool for raytrace/analytical shape
    # time-delay calculations and for reconstruction
    worker_pool = mp.Pool(__CPU_COUNT - 1)

    for ii in range(n_expts - 1):
        # The output dir, where the reconstructions will be stored
        out_dir = os.path.join(__OUT_DIR, "recons/Gen2/Diagonal")
        verify_path(out_dir)

        logger.info("Scan [%3d / %3d]..." % (ii + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = fd_data[ii, :, :]
        tar_md = metadata[ii]

        if ~np.isnan(tar_md["emp_ref_id"]):
            expt_adi_out_dir = os.path.join(
                out_dir, "id-%d-adi-%.1f-fibr-perc/" % (ii, fibr_perc)
            )
            verify_path(expt_adi_out_dir)

            # Get metadata for plotting
            scan_rad = tar_md["ant_rad"] / 100
            tum_x = tar_md["tum_x"] / 100
            tum_y = tar_md["tum_y"] / 100
            tum_rad = 0.5 * (tar_md["tum_diam"] / 100)
            adi_rad = __PHANTOM_RAD

            # Correct for how the scan radius is measured (from a
            # point on the antenna stand, not at the SMA connection
            # point)
            scan_rad += 0.03618

            # Define the radius of the region of interest
            roi_rad = adi_rad + 0.01

            # Get the area of each pixel in the image domain
            dv = ((2 * roi_rad) ** 2) / (__M_SIZE**2)

            # Correct for the antenna time delay
            # NOTE: Only the new antenna was used in UM-BMID Gen-3
            ant_rad = apply_ant_t_delay(scan_rad=scan_rad, new_ant=True)

            # breast_speed = get_breast_speed(fibr_perc)
            # # # breast_speed = get_speed_from_perm(perm)
            # breast_speed = np.average(velocities)
            # #
            # # # Get the one-way propagation times for each pixel,
            # # # for each antenna position and intersection points
            # pix_ts, int_f_xs, int_f_ys, int_b_xs, int_b_ys = \
            #     get_pix_ts(ant_rad=ant_rad, m_size=__M_SIZE,
            #                roi_rad=roi_rad, air_speed=__VAC_SPEED,
            #                breast_speed=breast_speed, adi_rad=adi_rad)

            speed = estimate_speed(
                adi_rad=adi_rad, ant_rad=scan_rad, new_ant=True
            )

            pix_ts = get_pix_ts_old(
                ant_rad=ant_rad, m_size=__M_SIZE, roi_rad=roi_rad, speed=speed
            )

            # Get the phase factor for efficient computation
            phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

            # Get the adipose-only reference data for this scan
            adi_fd = fd_data[expt_ids.index(tar_md["emp_ref_id"]), :, :]

            adi_cal_cropped = (tar_fd - adi_fd)[tar_fs, :]

            # If the scan does include a tumour
            if ~np.isnan(tar_md["tum_diam"]):
                # Set a str for plotting
                plt_str = "%.1f cm rod in\nID: %d" % (tar_md["tum_diam"], ii)
            else:
                plt_str = "Empty cylinder\nID: %d" % ii

            das_adi_recon = fd_das(
                fd_data=adi_cal_cropped,
                phase_fac=phase_fac,
                freqs=scan_fs[tar_fs],
                worker_pool=worker_pool,
            )

            # Reconstruct a DAS image
            # das_adi_recon =\
            #     fd_das_vel_freq(fd_data=adi_cal_cropped,
            #                     int_f_xs=int_f_xs, int_f_ys=int_f_ys,
            #                     int_b_xs=int_b_xs, int_b_ys=int_b_ys,
            #                     velocities=velocities, ant_rad=ant_rad,
            #                     freqs=scan_fs[tar_fs], adi_rad=adi_rad,
            #                     m_size=__M_SIZE, roi_rad=roi_rad,
            #                     air_speed=__VAC_SPEED, worker_pool=worker_pool)

            # Save that DAS reconstruction to a .pickle file
            save_pickle(
                das_adi_recon, os.path.join(expt_adi_out_dir, "das_adi.pickle")
            )

            bound_x, bound_y = find_boundary(
                np.abs(das_adi_recon), roi_rad, n_slices=120
            )
            rho, phi = cart_to_polar(bound_x, bound_y)

            cs = polar_fit_cs(rho, phi)
            # mask = get_binary_mask(cs, m_size=__M_SIZE, roi_rad=roi_rad)

            # Plot the DAS reconstruction
            plot_fd_img(
                img=np.abs(das_adi_recon),
                bound_x=bound_x * 100,
                bound_y=bound_y * 100,
                cs=cs,
                tum_x=tum_x,
                tum_y=tum_y,
                tum_rad=tum_rad,
                adi_rad=adi_rad,
                ant_rad=ant_rad,
                roi_rad=roi_rad,
                img_rad=roi_rad,
                title=plt_str,
                save_fig=True,
                save_str=os.path.join(
                    out_dir,
                    "id_%d_adi_cal_das_%.1f_fibr_perc.png" % (ii, fibr_perc),
                ),
                save_close=True,
            )

            #
            # # Reconstruct an ORR image
            # adi_orr = orr_recon(np.real(das_adi_recon),
            #                     freqs=scan_fs[tar_fs],
            #                     m_size=__M_SIZE,
            #                     fd=adi_cal_cropped,
            #                     pos=__SPHERE_POS[ii],
            #                     tum_rad=__SPHERE_RAD,
            #                     velocities=velocities,
            #                     ant_rad=ant_rad,
            #                     adi_rad=adi_rad,
            #                     phase_fac=phase_fac,
            #                     int_f_xs=int_f_xs, int_f_ys=int_f_ys,
            #                     int_b_xs=int_b_xs, int_b_ys=int_b_ys,
            #                     out_dir=expt_adi_out_dir,
            #                     air_speed=__VAC_SPEED,
            #                     worker_pool=worker_pool, logger=logger)
            # # Plot the ORR image
            # plot_fd_img(img=np.abs(adi_orr), tum_x=tum_x, tum_y=tum_y,
            #             tum_rad=tum_rad,
            #             ant_rad=ant_rad, roi_rad=roi_rad,
            #             img_rad=roi_rad, adi_rad=adi_rad,
            #             title='%s\nAdi Cal' % plt_str,
            #             save_fig=True,
            #             save_str=os.path.join(out_dir,
            #                         'id_%d_adi_cal_orr_%.1f_fibr_perc.png'
            #                         % (ii, fibr_perc)), save_close=True)
