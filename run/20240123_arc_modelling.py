"""Illia Prykhodko

University of Manitoba,
January 23rd, 2024
"""

import multiprocessing as mp
import os

import numpy as np
import pandas
import scipy.constants

from umbms import get_proj_path, get_script_logger, verify_path
from umbms.beamform.propspeed import estimate_speed, get_breast_speed_freq
from umbms.beamform.time_delay import (
    get_pix_ts,
    get_pix_ts_old,
    time_signal_per_antenna_modelled,
)
from umbms.beamform.utility import apply_ant_t_delay
from umbms.hardware.antenna import apply_ant_pix_delay
from umbms.loadsave import load_pickle
from umbms.plot.imgplots import (
    calculate_arc_map_known_time,
    plot_known_arc_map,
)

__CPU_COUNT = mp.cpu_count()

# SPECIFY CORRECT DATA AND OUTPUT PATHS
########################################################################

__DATA_DIR = os.path.join(get_proj_path(), "data/umbmid/cyl_phantom/")
__OUT_DIR = os.path.join(get_proj_path(), "output/cyl_phantom/")
verify_path(__OUT_DIR)
__DIEL_DATA_DIR = os.path.join(get_proj_path(), "data/freq_data/")

__FD_NAME = "20240109_s11_data.pickle"
__MD_NAME = "20240109_metadata.pickle"
__DIEL_NAME = "20240109_DGBE90.csv"

########################################################################

# SPECIFY CORRECT SCAN PARAMETERS
########################################################################

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

########################################################################

# Define propagation speed in vacuum
__VAC_SPEED = scipy.constants.speed_of_light
__PHANTOM_RAD = 0.0555


def load_data():
    """Loads both fd_data and metadata

    Returns:
    --------
    tuple of two loaded variables
    """
    return load_pickle(os.path.join(__DATA_DIR, __FD_NAME)), load_pickle(
        os.path.join(__DATA_DIR, __MD_NAME)
    )


if __name__ == "__main__":
    # initialize shared worker pool for raytrace/analytical shape
    # time-delay calculations and for reconstruction
    worker_pool = mp.Pool(__CPU_COUNT - 1)

    logger = get_script_logger(__file__)

    # Load the frequency domain data and metadata
    fd_data, metadata = load_data()

    n_expts = np.size(fd_data, axis=0)  # The number of individual scans

    # Get the unique ID of each experiment / scan
    expt_ids = [md["id"] for md in metadata]

    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

    # ICZT times
    iczt_time = np.linspace(__INI_T, __FIN_T, __N_TS)

    # Read .csv file of permittivity and conductivity values
    df = pandas.read_csv(os.path.join(__DIEL_DATA_DIR, __DIEL_NAME))

    # Calculate velocity array
    freqs = np.array(df["Freqs"].values, dtype=float) * 1e6
    permittivities = np.array(df["Permittivity"].values)
    conductivities = np.array(df["Conductivity"].values)
    zero_conductivities = np.zeros_like(conductivities)
    velocities_zero_cond = get_breast_speed_freq(
        freqs, permittivities, zero_conductivities
    )
    velocities = get_breast_speed_freq(freqs, permittivities, conductivities)

    # Calculate the time delay for a target according to different enhs.
    # Assume signal attenuates with 1/r^2
    # Plot
    out_dir = os.path.join(
        __OUT_DIR,
        "recons/Immediate reference/20240109_glass_rod/arc_investigation/",
    )
    verify_path(out_dir)

    for expt in range(n_expts):  # for all scans
        # for expt in [4]:
        logger.info("Scan [%3d / %3d]..." % (expt + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = fd_data[expt, :, :]
        tar_md = metadata[expt]

        # if the scan has both empty and adipose references and is not
        # a rod reference
        if (
            ~np.isnan(tar_md["emp_ref_id"])
            and ~np.isnan(tar_md["adi_ref_id2"])
            and tar_md["type"] != "rod reference"
        ):
            # If the scan does include a tumour
            if ~np.isnan(tar_md["tum_diam"]):
                # Set a str for plotting
                plt_str = "%.1f cm rod in\nID: %d" % (tar_md["tum_diam"], expt)
            else:
                plt_str = "Empty phantom\nID: %d" % expt
            # TEMPORARY
            plt_str = ""

            # Get metadata for plotting
            scan_rad = tar_md["ant_rad"] / 100
            tum_x = tar_md["tum_x"] / 100
            tum_y = tar_md["tum_y"] / 100
            tum_rad = 0.5 * (tar_md["tum_diam"] / 100)

            # Cylindrical phantom metadata doesn't have such a field,
            # its radius is hard-coded in the scan parameters section
            # adi_rad = tar_md['adi_rad']
            adi_rad = __PHANTOM_RAD

            # Obtain the true rho of the phase center of the antenna
            # ant_rad = to_phase_center(meas_rho=scan_rad)

            ant_rad = apply_ant_t_delay(scan_rad=scan_rad, new_ant=True)

            # Define the radius of the region of interest
            roi_rad = adi_rad + 0.01

            # Get the area of each pixel in the image domain
            dv = ((2 * roi_rad) ** 2) / (__M_SIZE**2)

            # Get the adipose-only and empty reference data
            # for this scan
            adi_fd_emp = fd_data[expt_ids.index(tar_md["emp_ref_id"]), :, :]
            adi_fd = fd_data[expt_ids.index(tar_md["rod_ref_id"]), :, :]
            adi_cal_cropped_emp = tar_fd - adi_fd_emp
            adi_cal_cropped = tar_fd - adi_fd

            # HOMOGENEOUS

            plt_str_regular_das = "Homogeneous DAS\n%s" % plt_str

            logger.info("\tHomogeneous DAS...")

            # Estimate the average speed for the whole imaging domain
            # Assume homogeneous media and straight line propagation
            speed = estimate_speed(
                adi_rad=adi_rad, ant_rad=scan_rad, new_ant=True
            )

            logger.info("\tTime-delay calculation...")

            pix_ts = get_pix_ts_old(
                ant_rad=ant_rad, m_size=__M_SIZE, roi_rad=roi_rad, speed=speed
            )

            # Account for antenna time delay
            pix_ts = apply_ant_pix_delay(pix_ts=pix_ts)

            times_signals = time_signal_per_antenna_modelled(
                tar_x=tum_x,
                tar_y=tum_y,
                tar_rad=tum_rad,
                ant_rad=ant_rad,
                speed=speed,
            )

            arc_map = calculate_arc_map_known_time(
                pix_ts, times_signals=times_signals
            )

            plot_known_arc_map(
                img_roi=roi_rad * 100,
                save_str=os.path.join(
                    out_dir, f"theoretical_arc_homog_{expt}.png"
                ),
                arc_map=arc_map,
            )

            # find the index for the tumor
            x_dists = np.linspace(-roi_rad, roi_rad, __M_SIZE)
            x_idx = np.argmax(
                np.isclose(x_dists, [tum_x for _ in range(__M_SIZE)], atol=5e-4)
            )
            y_dists = np.linspace(-roi_rad, roi_rad, __M_SIZE)
            y_idx = np.argmax(
                np.isclose(y_dists, [tum_y for _ in range(__M_SIZE)], atol=5e-4)
            )

            # BINARY

            logger.info("\tBinary DAS...")

            breast_speed = np.average(velocities)

            pix_ts, int_f_xs, int_f_ys, int_b_xs, int_b_ys = get_pix_ts(
                ant_rad=ant_rad,
                m_size=__M_SIZE,
                roi_rad=roi_rad,
                air_speed=__VAC_SPEED,
                breast_speed=breast_speed,
                adi_rad=adi_rad,
                worker_pool=worker_pool,
            )

            pix_ts = apply_ant_pix_delay(pix_ts=pix_ts)

            times_signals = time_signal_per_antenna_modelled(
                tar_x=tum_x,
                tar_y=tum_y,
                tar_rad=tum_rad,
                ant_rad=ant_rad,
                speed=speed,
                int_f_xs=int_f_xs,
                int_f_ys=int_f_ys,
                x_idx=x_idx,
                y_idx=y_idx,
                breast_speed=breast_speed,
                air_speed=__VAC_SPEED,
            )

            arc_map = calculate_arc_map_known_time(
                pix_ts, times_signals=times_signals
            )

            plot_known_arc_map(
                img_roi=roi_rad * 100,
                save_str=os.path.join(
                    out_dir, f"theoretical_arc_bin_{expt}.png"
                ),
                arc_map=arc_map,
            )

            # # FREQUENCY-DEPENDENT
            logger.info("\tFrequency DAS...")

            arc_map = np.zeros(shape=(150, 150))

            breast_speed = np.average(velocities)

            for ff in range(scan_fs.size):
                pix_ts, _, _, _, _ = get_pix_ts(
                    ant_rad=ant_rad,
                    m_size=__M_SIZE,
                    roi_rad=roi_rad,
                    air_speed=__VAC_SPEED,
                    breast_speed=velocities[ff],
                    adi_rad=adi_rad,
                    int_f_xs=int_f_xs,
                    int_f_ys=int_f_ys,
                    int_b_xs=int_b_xs,
                    int_b_ys=int_b_ys,
                )

                pix_ts = apply_ant_pix_delay(pix_ts=pix_ts)

                times_signals = time_signal_per_antenna_modelled(
                    tar_x=tum_x,
                    tar_y=tum_y,
                    tar_rad=tum_rad,
                    ant_rad=ant_rad,
                    speed=speed,
                    int_f_xs=int_f_xs,
                    int_f_ys=int_f_ys,
                    x_idx=x_idx,
                    y_idx=y_idx,
                    breast_speed=velocities[ff],
                    air_speed=__VAC_SPEED,
                )

                arc_map += calculate_arc_map_known_time(
                    pix_ts, times_signals=times_signals
                )

            plot_known_arc_map(
                img_roi=roi_rad * 100,
                save_str=os.path.join(
                    out_dir, f"theoretical_arc_freq_dep_{expt}.png"
                ),
                arc_map=arc_map,
            )
