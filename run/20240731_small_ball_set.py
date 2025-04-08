"""
Illia Prykhodko

Univerity of Manitoba,
May 31st, 2023
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas
from time import perf_counter
import multiprocessing as mp
import scipy.constants

from umbms import get_proj_path, verify_path, get_script_logger
from umbms.loadsave import load_pickle, save_pickle
from umbms.hardware.antenna import apply_ant_pix_delay, to_phase_center
from umbms.beamform.das import fd_das, fd_das_freq_dep
from umbms.beamform.dmas import fd_dmas
from umbms.beamform.time_delay import get_pix_ts, get_pix_ts_old
from umbms.beamform.utility import apply_ant_t_delay, get_fd_phase_factor
from umbms.boundary.boundary_detection import get_boundary_iczt, get_binary_mask
from umbms.beamform.propspeed import estimate_speed, get_breast_speed_freq
from umbms.boundary.raytrace import find_boundary_rt
from umbms.plot.imgplots import plot_fd_img
from umbms.plot.sinogramplot import plt_fd_sino

__CPU_COUNT = mp.cpu_count()

# SPECIFY CORRECT DATA AND OUTPUT PATHS
########################################################################

__DATA_DIR = os.path.join(
    get_proj_path(), "data/umbmid/cyl_phantom/speed_paper/"
)
__OUT_DIR = os.path.join(get_proj_path(), "output/cyl_phantom/")
verify_path(__OUT_DIR)
__DIEL_DATA_DIR = os.path.join(get_proj_path(), "data/freq_data/")

__FD_NAME = "s11_small_data.pickle"
__MD_NAME = "20240731_metadata.pickle"
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
__SAMPLING = 12

########################################################################

# Define propagation speed in vacuum
__VAC_SPEED = scipy.constants.speed_of_light
__PHANTOM_RAD = 0.0555


def interp_perms(data_fs, perms, conds, tar_fs):
    """Interpolate measured permittivity to a set of target freqs
    Parameters
    ----------
    data_fs : array_like
        Frequencies at which measured perms/conds are defined
    perms : array_like
        Measured real part of the relative permittivity
    conds : array_like
        Measured conductivity, [S/m]
    tar_fs : array_like
        Target frequencies
    Returns
    -------
    interp_perms : array_like
        Interpolated permittivities
    interp_conds : array_like
        Interpolated conductivities, in [S/m]
    """

    interp_perms = np.interp(x=tar_fs, xp=data_fs, fp=perms)
    interp_conds = np.interp(x=tar_fs, xp=data_fs, fp=conds)

    return interp_perms, interp_conds


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
    # Downsample for computational benefit
    fd_data = fd_data[:, ::__SAMPLING, :]

    n_expts = np.size(fd_data, axis=0)  # The number of individual scans

    # Get the unique ID of each experiment / scan
    expt_ids = [md["id"] for md in metadata]

    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

    # DOWNsample the frequencies
    recon_fs = scan_fs[::__SAMPLING]

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
    permittivities, conductivities = interp_perms(
        freqs, permittivities, conductivities, recon_fs
    )
    velocities = get_breast_speed_freq(recon_fs, permittivities, conductivities)

    # Calculate the time delay for a target according to different enhs.
    # Assume signal attenuates with 1/r^2
    # Plot
    out_dir = os.path.join(
        __OUT_DIR, "recons/Immediate reference/Speed paper/small_target/"
    )
    verify_path(out_dir)

    # Get metadata for plotting
    scan_rad = 0.21

    # Obtain the true rho of the phase center of the antenna
    ant_rad = to_phase_center(meas_rho=scan_rad)

    adi_rad = __PHANTOM_RAD

    # Define the radius of the region of interest
    roi_rad = adi_rad + 0.01

    # Get the area of each pixel in the image domain
    dv = ((2 * roi_rad) ** 2) / (__M_SIZE**2)

    # Estimate the average speed for the whole imaging domain
    # Assume homogeneous media and straight line propagation
    speed = estimate_speed(adi_rad=adi_rad, ant_rad=scan_rad, new_ant=True)
    # Homogeneous
    logger.info("\tTime-delay calculation...")
    time_delay_tp_start = perf_counter()
    pix_ts = get_pix_ts_old(
        ant_rad=ant_rad, m_size=__M_SIZE, roi_rad=roi_rad, speed=speed
    )
    time_delay_tp_end = perf_counter()
    logger.info("\t\tTime: %.3f s" % (time_delay_tp_end - time_delay_tp_start))
    # Account for antenna time delay
    pix_ts = apply_ant_pix_delay(pix_ts=pix_ts)

    phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

    # Binary
    breast_speed = np.average(velocities)
    logger.info("\tTime-delay calculation (binary)...")
    time_delay_tp_start = perf_counter()
    pix_ts_bin, int_f_xs, int_f_ys, int_b_xs, int_b_ys = get_pix_ts(
        ant_rad=ant_rad,
        m_size=__M_SIZE,
        roi_rad=roi_rad,
        air_speed=__VAC_SPEED,
        breast_speed=breast_speed,
        adi_rad=adi_rad,
        worker_pool=worker_pool,
    )

    # Account for antenna time delay
    pix_ts_bin = apply_ant_pix_delay(pix_ts=pix_ts_bin)
    time_delay_tp_end = perf_counter()
    logger.info("\t\tTime: %.3f s" % (time_delay_tp_end - time_delay_tp_start))
    phase_fac_bin = get_fd_phase_factor(pix_ts=pix_ts_bin)

    for expt in range(n_expts):
        # for expt in [2, 4]:

        logger.info("Scan [%3d / %3d]..." % (expt + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = fd_data[expt, :, :]
        tar_md = metadata[expt]

        # if the scan has both empty and adipose references
        if ~np.isnan(tar_md["emp_ref_id"]) and ~np.isnan(tar_md["adi_ref_id2"]):
            # If the scan does include a tumour
            if ~np.isnan(tar_md["tum_diam"]):
                # Set a str for plotting
                plt_str = "%.1f cm rod in\nID: %d" % (tar_md["tum_diam"], expt)
            else:
                plt_str = "Empty phantom\nID: %d" % expt

            # Create a directory for storing .pickle files
            pickle_dir = os.path.join(out_dir, "pickles/")
            verify_path(pickle_dir)

            # Get metadata for plotting
            tum_x = tar_md["tum_x"] / 100
            tum_y = tar_md["tum_y"] / 100
            tum_rad = 0.5 * (tar_md["tum_diam"] / 100)

            # Get the adipose-only and empty reference data
            # for this scan
            adi_fd_emp = fd_data[expt_ids.index(tar_md["emp_ref_id"]), :, :]
            adi_fd = fd_data[expt_ids.index(tar_md["emp_ref_id"]) - 1, :, :]
            adi_cal_cropped_emp = tar_fd - adi_fd_emp
            adi_cal_cropped = tar_fd - adi_fd

            # 5 DIFFERENT RECONSTRUCTIONS
            ############################################################

            # 1. Homogeneous DAS (regular)

            plt_str_regular_das = "Homogeneous DAS\n%s" % plt_str

            logger.info("\tHomogeneous Reconstruction...")

            phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

            # Time the reconstruction
            recon_start = perf_counter()
            das_regular_recon = fd_das(
                fd_data=adi_cal_cropped,
                phase_fac=phase_fac,
                freqs=recon_fs,
                worker_pool=worker_pool,
            )
            recon_end = perf_counter()
            logger.info("\t\tTime: %.3f s" % (recon_end - recon_start))

            save_pickle(
                das_regular_recon,
                path=os.path.join(pickle_dir, "id%d_hom.pickle" % expt),
            )

            plot_fd_img(
                img=np.abs(das_regular_recon),
                tum_x=tum_x,
                tum_y=tum_y,
                tum_rad=tum_rad,
                adi_rad=adi_rad,
                ant_rad=ant_rad,
                roi_rad=roi_rad,
                img_rad=roi_rad,
                title="",
                save_fig=True,
                save_str=os.path.join(out_dir, "id_%d_das_hom.png" % expt),
                save_close=True,
            )

            ############################################################
            # 2. Binary DAS (domain partitioning)

            logger.info("\tBinary Reconstruction...")
            recon_start = perf_counter()
            das_binary_recon = fd_das(
                fd_data=adi_cal_cropped,
                phase_fac=phase_fac_bin,
                freqs=recon_fs,
                worker_pool=worker_pool,
            )
            recon_end = perf_counter()
            logger.info("\t\tTime: %.3f" % (recon_end - recon_start))

            save_pickle(
                das_binary_recon,
                path=os.path.join(pickle_dir, "id%d_binary.pickle" % expt),
            )

            plot_fd_img(
                img=np.abs(das_binary_recon),
                tum_x=tum_x,
                tum_y=tum_y,
                tum_rad=tum_rad,
                adi_rad=adi_rad,
                ant_rad=ant_rad,
                roi_rad=roi_rad,
                img_rad=roi_rad,
                title="",
                save_fig=True,
                save_str=os.path.join(out_dir, "id_%d_das_binary.png" % expt),
                save_close=True,
            )

            ############################################################
            # 3. Frequency-dependent DAS (zero cond, short - FDNC)

            plt_str_fdnc_das = (
                "Frequency-dependent DAS (zero conductivity)\n%s" % plt_str
            )

            logger.info("\tFrequency-dependent DAS (zero conductivity)...")

            fdnc_das_start = perf_counter()

            recon_start = perf_counter()
            logger.info("\tReconstruction...")
            das_freq_dep_zero_cond_recon = fd_das_freq_dep(
                fd_data=adi_cal_cropped,
                int_f_xs=int_f_xs,
                int_f_ys=int_f_ys,
                int_b_xs=int_b_xs,
                int_b_ys=int_b_ys,
                velocities=velocities_zero_cond,
                ant_rad=ant_rad,
                freqs=recon_fs,
                adi_rad=adi_rad,
                m_size=__M_SIZE,
                roi_rad=roi_rad,
                air_speed=__VAC_SPEED,
                worker_pool=worker_pool,
            )
            recon_end = perf_counter()
            logger.info("\t\tTime: %.3f s" % (recon_end - recon_start))

            save_pickle(
                das_freq_dep_zero_cond_recon,
                path=os.path.join(pickle_dir, "id%d_fdnc.pickle" % expt),
            )

            plot_fd_img(
                img=np.abs(das_freq_dep_zero_cond_recon),
                tum_x=tum_x,
                tum_y=tum_y,
                tum_rad=tum_rad,
                adi_rad=adi_rad,
                ant_rad=ant_rad,
                roi_rad=roi_rad,
                img_rad=roi_rad,
                title="",
                save_fig=True,
                save_str=os.path.join(
                    out_dir, "id_%d_das_freq_dep_zero_cond.png" % expt
                ),
                save_close=True,
            )

            fdnc_das_end = perf_counter()
            logger.info(
                "\tThe whole reconstruction time: %.3f s"
                % (fdnc_das_end - fdnc_das_start)
            )

            ############################################################
            # 4. Frequency-dependent DAS (short - FD)

            plt_str_fd_das = "Frequency-dependent DAS\n%s" % plt_str

            logger.info("\tFrequency-dependent DAS...")

            fd_das_start = perf_counter()

            recon_start = perf_counter()
            logger.info("\tReconstruction...")
            das_freq_dep_zero_cond_recon = fd_das_freq_dep(
                fd_data=adi_cal_cropped,
                int_f_xs=int_f_xs,
                int_f_ys=int_f_ys,
                int_b_xs=int_b_xs,
                int_b_ys=int_b_ys,
                velocities=velocities,
                ant_rad=ant_rad,
                freqs=recon_fs,
                adi_rad=adi_rad,
                m_size=__M_SIZE,
                roi_rad=roi_rad,
                air_speed=__VAC_SPEED,
                worker_pool=worker_pool,
            )
            recon_end = perf_counter()
            logger.info("\t\tTime: %.3f s" % (recon_end - recon_start))

            save_pickle(
                das_freq_dep_zero_cond_recon,
                path=os.path.join(pickle_dir, "id%d_fd.pickle" % expt),
            )

            plot_fd_img(
                img=np.abs(das_freq_dep_zero_cond_recon),
                tum_x=tum_x,
                tum_y=tum_y,
                tum_rad=tum_rad,
                adi_rad=adi_rad,
                ant_rad=ant_rad,
                roi_rad=roi_rad,
                img_rad=roi_rad,
                title="",
                save_fig=True,
                save_str=os.path.join(out_dir, "id_%d_das_freq_dep.png" % expt),
                save_close=True,
            )

            fd_das_end = perf_counter()
            logger.info(
                "\tThe whole reconstruction time: %.3f s"
                % (fd_das_end - fd_das_start)
            )

            ############################################################
            # 5. Ray-tracing

            plt_str_rt_das = (
                "DAS with raytracing, frequency-dependent\n%s" % plt_str
            )

            logger.info("\tDAS with raytracing...")

            rt_das_start = perf_counter()

            # TEMPORARY:
            # In order to account for straight-line antenna time-delay
            # apply the old correction

            scan_rad_for_rt = scan_rad + 0.03618
            ant_rad_bound = apply_ant_t_delay(
                scan_rad=scan_rad_for_rt, new_ant=True
            )

            # Routine start: extract the boundary from the sinogram
            cs, x_cm, y_cm = get_boundary_iczt(
                adi_cal_cropped_emp, ant_rad_bound
            )

            # Apply the cubic spline onto the grid
            mask = get_binary_mask(cs, m_size=__M_SIZE, roi_rad=roi_rad)

            logger.info("\tTime-delay calculation...")
            time_delay_tp_start = perf_counter()
            # Recalculate intersection points according to Siddon's algorithm
            int_f_xs, int_f_ys, int_b_xs, int_b_ys = find_boundary_rt(
                mask, ant_rad, roi_rad, worker_pool=worker_pool
            )
            time_delay_tp_end = perf_counter()
            logger.info(
                "\t\tTime: %.3f s" % (time_delay_tp_end - time_delay_tp_start)
            )

            logger.info("\tReconstruction...")
            recon_start = perf_counter()
            das_rt_recon = fd_das_freq_dep(
                fd_data=adi_cal_cropped,
                int_f_xs=int_f_xs,
                int_f_ys=int_f_ys,
                int_b_xs=int_b_xs,
                int_b_ys=int_b_ys,
                velocities=velocities,
                ant_rad=ant_rad,
                freqs=recon_fs,
                adi_rad=adi_rad,
                m_size=__M_SIZE,
                roi_rad=roi_rad,
                air_speed=__VAC_SPEED,
                worker_pool=worker_pool,
            )
            recon_end = perf_counter()
            logger.info("\t\tTime: %.3f s" % (recon_end - recon_start))

            save_pickle(
                das_rt_recon,
                path=os.path.join(pickle_dir, "id%d_rt.pickle" % expt),
            )

            plot_fd_img(
                img=np.abs(das_rt_recon),
                cs=cs,
                tum_x=tum_x,
                tum_y=tum_y,
                tum_rad=tum_rad,
                adi_rad=adi_rad,
                ox=x_cm,
                oy=y_cm,
                ant_rad=ant_rad,
                roi_rad=roi_rad,
                img_rad=roi_rad,
                title="",
                save_fig=True,
                save_str=os.path.join(out_dir, "id_%d_das_rt.png" % expt),
                save_close=True,
            )

            rt_das_end = perf_counter()
            logger.info(
                "\tThe whole reconstruction time: %.3f s"
                % (rt_das_end - rt_das_start)
            )

    worker_pool.close()
