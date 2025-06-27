"""
Illia Prykhodko
University of Manitoba
June 25th, 2025
"""

import multiprocessing as mp
import os

import numpy as np
import scipy.constants

from umbms import get_proj_path, get_script_logger, verify_path
from umbms.beamform.das import fd_das
from umbms.beamform.propspeed import (
    estimate_speed,
)
from umbms.beamform.time_delay import (
    get_pix_ts_old,
)
from umbms.beamform.utility import (
    apply_ant_t_delay,
    get_fd_phase_factor,
)

from umbms.boundary.boundary_detection import get_boundary_iczt

from umbms.loadsave import load_pickle, save_pickle
from umbms.plot.imgplots import plot_fd_img

###############################################################################

__CPU_COUNT = mp.cpu_count()

__DATA_DIR = os.path.join(get_proj_path(), "data/umbmid/g3/")

__OUT_DIR = os.path.join(get_proj_path(), "output/g3/msc_thesis/")
verify_path(__OUT_DIR)

__FD_NAME = "fd_data_gen_three_s11.pickle"
__MD_NAME = "metadata_gen_three.pickle"

# The frequency parameters from the scan
__INI_F = 1e9
__FIN_F = 9e9
__N_FS = 1001

# The time-domain conversion parameters
__INI_T = 0.5e-9
__FIN_T = 5.5e-9
__N_TS = 700

# The size of the reconstructed image along one dimension
__M_SIZE = 150

__SAMPLING = 12

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
__SPHERE_RAD = 0.0075

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


###############################################################################


if __name__ == "__main__":
    logger = get_script_logger(__file__)

    # Load the frequency domain data and metadata
    fd_data, metadata = load_data()

    n_expts = len(metadata)  # The number of individual scans

    # Get the unique ID of each experiment / scan
    expt_ids = [md["id"] for md in metadata]

    # Get the unique IDs of the adipose-only and adipose-fibroglandular
    # (healthy) reference scans for each experiment/scan
    adi_ref_ids = [md["adi_ref_id"] for md in metadata]
    fib_ref_ids = [md["fib_ref_id"] for md in metadata]

    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

    # Only retain frequencies above 2 GHz due to antenna VSWR
    tar_fs = scan_fs >= 2e9

    scan_fs = scan_fs[tar_fs]

    recon_fs = scan_fs[::__SAMPLING]

    out_dir = os.path.join(__OUT_DIR, "recons/")
    verify_path(out_dir)

    for ii in range(170, n_expts):  # For each scan / experiment
        # for ii in range(5):  # For each scan / experiment

        # initialize shared worker pool for raytrace/analytical shape
        # time-delay calculations and for reconstruction
        worker_pool = mp.Pool(__CPU_COUNT - 1)

        logger.info("Scan [%3d / %3d]..." % (ii + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = fd_data[ii, :, :]
        tar_md = metadata[ii]

        # If the scan had a fibroglandular shell (indicating it was of
        # a complete tumour-containing or healthy phantom)
        if "F" in tar_md["phant_id"]:
            # Create the output directory for the adipose-only
            # reference reconstructions
            expt_adi_out_dir = os.path.join(
                out_dir,
                "id-%d/" % (tar_md["id"]),
            )
            verify_path(expt_adi_out_dir)

            # Create the output directory for the adipose-fibroglandular
            # reference reconstructions
            expt_fib_out_dir = os.path.join(
                out_dir,
                "id-%d/" % (tar_md["id"]),
            )
            verify_path(expt_fib_out_dir)

            # Get metadata for plotting
            scan_rad = tar_md["ant_rad"] / 100
            tum_x = tar_md["tum_x"] / 100
            tum_y = tar_md["tum_y"] / 100
            tum_rad = 0.5 * (tar_md["tum_diam"] / 100)
            adi_rad = __ADI_RADS[tar_md["phant_id"].split("F")[0]]
            mid_breast_max, mid_breast_min = (
                __MID_BREAST_RADS[tar_md["phant_id"].split("F")[0]][0],
                __MID_BREAST_RADS[tar_md["phant_id"].split("F")[0]][1],
            )

            # correction for a phantom
            # mid_breast_max -= 0.00832
            # mid_breast_min -= 0.0071231

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

            # Estimate the propagation speed in the imaging domain
            speed = estimate_speed(
                adi_rad=adi_rad, ant_rad=scan_rad, new_ant=True
            )
            pix_ts = get_pix_ts_old(
                ant_rad=ant_rad, m_size=__M_SIZE, roi_rad=roi_rad, speed=speed
            )

            # Get the phase factor for efficient computation
            phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

            # Get the adipose-only reference data for this scan
            adi_fd = fd_data[expt_ids.index(tar_md["adi_ref_id"]), :, :]

            fd_emp = fd_data[expt_ids.index(tar_md["emp_ref_id"]), :, :]

            # Subtract reference and retain the frequencies above 2 GHz
            adi_cal_cropped = (tar_fd - adi_fd)[tar_fs, :]

            # Subtract the empty reference for the boundary detection
            emp_cal_cropped = (tar_fd - fd_emp)[tar_fs, :]

            # Reconstruct a DAS image
            das_adi_recon = fd_das(
                fd_data=adi_cal_cropped[::__SAMPLING, :],
                phase_fac=phase_fac,
                freqs=recon_fs,
                worker_pool=worker_pool,
            )

            cs, _, _ = get_boundary_iczt(
                adi_emp_cropped=emp_cal_cropped,
                ant_rad=ant_rad,
                plot_points=True,
                out_dir=out_dir,
                save_str="id_%d_" % ii,
            )

            # Save that DAS reconstruction to a .pickle file
            save_pickle(
                das_adi_recon,
                os.path.join(expt_adi_out_dir, "das_adi.pickle"),
            )

            # Plot the DAS reconstruction
            plot_fd_img(
                img=np.abs(das_adi_recon),
                tum_x=tum_x,
                tum_y=tum_y,
                tum_rad=tum_rad,
                ant_rad=ant_rad,
                roi_rad=roi_rad,
                img_rad=roi_rad,
                cs=cs,
                save_fig=True,
                save_str=os.path.join(
                    out_dir,
                    "id_%d_adi_cal_das.png" % (tar_md["id"]),
                ),
                save_close=True,
            )
