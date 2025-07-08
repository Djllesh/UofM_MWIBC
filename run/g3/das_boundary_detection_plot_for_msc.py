import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np

from umbms import get_proj_path, get_script_logger, verify_path
from umbms.beamform.das import fd_das
from umbms.beamform.propspeed import estimate_speed
from umbms.beamform.time_delay import get_pix_ts_old
from umbms.beamform.utility import (
    apply_ant_t_delay,
    get_fd_phase_factor,
)
from umbms.loadsave import load_pickle, save_pickle
from umbms.plot.imgplots import plot_fd_img
from umbms.boundary.boundary_detection import (
    find_boundary,
    cart_to_polar,
    polar_fit_cs,
)

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

    for ii in [119]:  # For each scan / experiment
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
            roi_rad = adi_rad + 0.02

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

            # Save that DAS reconstruction to a .pickle file
            save_pickle(
                das_adi_recon, os.path.join(expt_adi_out_dir, "das_adi.pickle")
            )

            bound_x, bound_y = find_boundary(
                np.abs(das_adi_recon), roi_rad, n_slices=110
            )
            rho, phi = cart_to_polar(bound_x, bound_y)
            # cs = polar_fit_cs(rho, phi)
            #
            # For mathtext to match Libertinus Serif:
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            plt.rc("font", family="Libertinus Serif")
            plt.figure()  # Make the figure window

            plt.tick_params(labelsize=14)
            plt.plot(phi, rho, "bx", label="Slice data")
            # plt.plot(phi, cs(phi), "r-", label="Cubic spline")
            plt.xlabel(r"$\varphi$ (rad)", fontsize=16)
            plt.ylabel(r"$\rho$ (m)", fontsize=16)
            plt.grid("-", linewidth=0.7)
            plt.legend(fontsize=15)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    out_dir, "das_extracted_boundary_spline_id_%d.png" % ii
                ),
                dpi=300,
                transparent=True,
            )

            # Plot the DAS reconstruction
            plot_fd_img(
                img=np.abs(das_adi_recon),
                # cs=cs,
                bound_x=bound_x * 100,
                bound_y=bound_y * 100,
                tum_x=tum_x,
                tum_y=tum_y,
                tum_rad=0,
                adi_rad=0,
                ant_rad=ant_rad,
                roi_rad=roi_rad,
                img_rad=roi_rad,
                title="",
                save_fig=True,
                save_str=os.path.join(
                    out_dir,
                    "das_recon_with_das_extracted_boundary_id%d.png" % ii,
                ),
                save_close=True,
                transparent=True,
            )

            break
