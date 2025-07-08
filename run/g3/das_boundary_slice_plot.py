"""
Illia Prykhodko
University of Manitoba
July 6th, 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import multiprocessing as mp

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.beamform.das import fd_das

from umbms.loadsave import load_pickle

from umbms.beamform.utility import (
    apply_ant_t_delay,
    get_fd_phase_factor,
)
from umbms.beamform.propspeed import estimate_speed
from umbms.beamform.time_delay import get_pix_ts_old

###############################################################################

__CPU_COUNT = mp.cpu_count()

__DATA_DIR = os.path.join(get_proj_path(), "data/umbmid/g3/")

__OUT_DIR = os.path.join(get_proj_path(), "output/g3/")
verify_path(__OUT_DIR)

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

###############################################################################


if __name__ == "__main__":
    # initialize shared worker pool for raytrace/analytical shape
    # time-delay calculations and for reconstruction
    worker_pool = mp.Pool(__CPU_COUNT - 1)

    logger = get_script_logger(__file__)

    # Load the frequency domain data and metadata
    fd_data = load_pickle(
        os.path.join(__DATA_DIR, "fd_data_gen_three_s11.pickle")
    )
    metadata = load_pickle(
        os.path.join(__DATA_DIR, "metadata_gen_three.pickle")
    )

    fd_data = fd_data[:, ::__SAMPLING, :]
    n_expts = len(metadata)  # The number of individual scans

    # Get the unique ID of each experiment / scan
    expt_ids = [md["id"] for md in metadata]

    # Get the unique IDs of the adipose-only and adipose-fibroglandular
    # (healthy) reference scans for each experiment/scan
    adi_ref_ids = [md["adi_ref_id"] for md in metadata]
    fib_ref_ids = [md["fib_ref_id"] for md in metadata]

    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

    scan_fs = scan_fs[::__SAMPLING]

    # Only retain frequencies above 2 GHz due to antenna VSWR
    tar_fs = scan_fs >= 2e9

    scan_fs = scan_fs[tar_fs]

    # The output dir, where the reconstructions will be stored
    out_dir = os.path.join(__OUT_DIR, "recons/")
    verify_path(out_dir)

    for ii in range(n_expts):  # For each scan / experiment
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
                out_dir, "id-%d-adi/" % tar_md["id"]
            )
            verify_path(expt_adi_out_dir)

            # Create the output directory for the adipose-fibroglandular
            # reference reconstructions
            expt_fib_out_dir = os.path.join(
                out_dir, "id-%d-fib/" % tar_md["id"]
            )
            verify_path(expt_fib_out_dir)

            # Get metadata for plotting
            scan_rad = tar_md["ant_rad"] / 100
            tum_x = tar_md["tum_x"] / 100
            tum_y = tar_md["tum_y"] / 100
            tum_rad = 0.5 * (tar_md["tum_diam"] / 100)
            adi_rad = __ADI_RADS[tar_md["phant_id"].split("F")[0]]

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

            # Get the one-way propagation times for each pixel,
            # for each antenna position
            pix_ts = get_pix_ts_old(
                ant_rad=ant_rad, m_size=__M_SIZE, roi_rad=roi_rad, speed=speed
            )

            # Get the phase factor for efficient computation
            phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

            # Get the adipose-only reference data for this scan
            adi_fd = fd_data[expt_ids.index(tar_md["adi_ref_id"]), :, :]
            emp_fd = fd_data[expt_ids.index(tar_md["emp_ref_id"]), :, :]

            # Subtract reference and retain the frequencies above 2 GHz
            adi_cal_cropped = (tar_fd - adi_fd)[tar_fs, :]
            emp_cal_cropped = (tar_fd - emp_fd)[tar_fs, :]

            # If the scan does include a tumour
            if ~np.isnan(tar_md["tum_diam"]):
                # Set a str for plotting
                plt_str = "%.1f cm tum in Class %d %s, ID: %d" % (
                    tar_md["tum_diam"],
                    tar_md["birads"],
                    tar_md["phant_id"],
                    tar_md["id"],
                )

            else:  # If the scan does NOT include a tumour
                plt_str = "Class %d %s, ID: %d" % (
                    tar_md["birads"],
                    tar_md["phant_id"],
                    tar_md["id"],
                )

            # Empty reference reconstruction below ----------------------------

            # HACK: comment below to get an actual DAS reconstruction
            das_adi_recon = fd_das(
                fd_data=emp_cal_cropped,
                phase_fac=phase_fac,
                freqs=scan_fs,
                worker_pool=worker_pool,
            )

            # Empty reference reconstruction above ----------------------------

            img_rad = roi_rad * 100
            img_extent = [-img_rad, img_rad, -img_rad, img_rad]
            img = np.abs(das_adi_recon)

            alpha = -30

            slope = np.tan(np.deg2rad(180 + alpha))

            # Set the font to times new roman
            plt.rc("font", family="Libertinus Serif")
            plt.figure()  # Make the figure window

            plt.imshow(
                img,
                cmap="inferno",
                extent=img_extent,
                aspect="equal",
            )

            plt.axline((0, 0), slope=slope, linewidth=1.3, color="r")
            plt.axline(
                (0, 0), slope=0, linewidth=0.7, color="black", linestyle="--"
            )
            plt.axline(
                (0, 0), (0, 1), linewidth=0.7, color="black", linestyle="--"
            )

            angles = np.deg2rad(np.linspace(150, 180, 50))

            angle_xs = 2 * np.cos(angles)
            angle_ys = 2 * np.sin(angles)

            annotate_x = 2.5 * np.cos(angles[np.size(angles) // 2])
            annotate_y = 2.5 * np.sin(angles[np.size(angles) // 2])

            plt.plot(angle_xs, angle_ys, "r-", linewidth=1)
            plt.text(
                annotate_x,
                annotate_y,
                s=r"$\alpha$",
                ha="right",
                va="center",
                fontsize=20,
            )

            # Set the size of the axis tick labels
            plt.tick_params(labelsize=14)

            # Set the x/y-ticks at multiples of 5 cm
            plt.gca().set_xticks([-6, -4, -2, 0, 2, 4, 6])
            plt.gca().set_yticks([-6, -4, -2, 0, 2, 4, 6])

            # Set the x/y axes limits
            plt.xlim([-roi_rad * 100, roi_rad * 100])
            plt.ylim([-roi_rad * 100, roi_rad * 100])

            plt.xlabel("x-axis (cm)", fontsize=16)  # Make the x-axis label
            plt.ylabel("y-axis (cm)", fontsize=16)  # Make the y-axis label
            plt.tight_layout()  # Remove excess whitespace in the figure

            plt.savefig(
                os.path.join(out_dir, "das_boundary_with_line.png"),
                transparent=True,
                dpi=300,
            )
            plt.close()

            # Set the font to times new roman
            plt.rc("font", family="Libertinus Serif")
            plt.figure()  # Make the figure window

            img_to_slice = ndimage.rotate(
                img, alpha, reshape=False, prefilter=False
            )

            # take a slice right in the middle that corresponds to y = 0
            slice = img_to_slice[np.size(img, axis=0) // 2, :]
            slice_extent = np.linspace(
                -roi_rad * 100, roi_rad * 100, np.size(slice)
            )

            plt.plot(slice_extent, slice, "r-", linewidth=1.3)

            # Set the size of the axis tick labels
            plt.tick_params(labelsize=14)
            plt.xlabel("Image extent (cm)", fontsize=16)
            plt.ylabel("Intensity (a.u.)", fontsize=16)
            plt.grid("-", linewidth=0.6)
            plt.tight_layout()
            plt.savefig(
                os.path.join(out_dir, "das_slice_at_alpha.png"),
                transparent=True,
                dpi=300,
            )
            plt.close()
            break
