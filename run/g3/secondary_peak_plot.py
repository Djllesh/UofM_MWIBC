"""
Illia Prykhodko
University of Manitoba
July 7th, 2025
"""

import multiprocessing as mp
import matplotlib.pyplot as plt
import os

import numpy as np
import scipy.constants

from scipy.signal import find_peaks, correlate, correlation_lags
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

from umbms.boundary.boundary_detection import (
    get_boundary_iczt,
    prepare_fd_data,
    polar_fit_cs,
)

from umbms.loadsave import load_pickle, save_pickle
from umbms.plot.imgplots import plot_fd_img
from umbms.plot.sinogramplot import plt_sino, show_sinogram
from umbms.beamform.iczt import iczt

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


def plot_sino_dent(
    fd,
    bound_times,
    title,
    save_str,
    out_dir,
    cbar_fmt="%.2e",
    transparent=True,
    close=True,
):
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

    bound_angles = np.linspace(0, 355, np.size(td, axis=1))

    show_sinogram(
        data=td,
        bound_times=bound_times,
        bound_angles=bound_angles,
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


def find_bound_times_with_dent(
    adi_emp_cropped,
    ant_rad,
    *,
    adi_rad=0,
    n_ant_pos=72,
    ini_ant_ang=-136.0,
    ini_t=0.5e-9,
    fin_t=5.5e-9,
    n_time_pts=700,
    ini_f=2e9,
    fin_f=9e9,
):
    td, ts, kernel = prepare_fd_data(
        adi_emp_cropped=emp_cal_cropped,
        ini_t=__INI_T,
        fin_t=__FIN_T,
        n_time_pts=__N_TS,
        ini_f=ini_f,
        fin_f=fin_f,
        ant_rad=ant_rad,
    )

    # creating an array of polar distances for storing
    rho = np.array([])
    # initializing an array of time-responces
    ToR = np.array([])

    for ant_pos in range(np.size(td, axis=1)):
        # corresponding intensities
        position = np.abs(td[:, ant_pos])

        # ------PEAK SELECTION------- #

        peak = np.argmax(position)

        # store the time response obtained from CMPPS method
        ToR = np.append(ToR, ts[peak])
        # polar radius of a corresponding highest intensity response
        # (corrected radius - radius of a time response)
        rad = ant_rad - ts[peak] * __VAC_SPEED / 2
        # TODO: account for new antenna time delay
        # appending polar radius to rho array
        rho = np.append(rho, rad)

    rho = np.flip(rho)
    # polar angle data
    angles = np.linspace(0, np.deg2rad(355), n_ant_pos) + np.deg2rad(
        ini_ant_ang
    )
    # CubicSpline interpolation on a given set of data
    cs = polar_fit_cs(rho, angles)

    return ToR, cs


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

    for ii in range(205, n_expts):  # For each scan / experiment
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
            pix_ts = get_pix_ts_old(
                ant_rad=ant_rad, m_size=__M_SIZE, roi_rad=roi_rad, speed=speed
            )

            ant_pos = 59

            # Get the phase factor for efficient computation
            phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

            # Get the adipose-only reference data for this scan
            adi_fd = fd_data[expt_ids.index(tar_md["adi_ref_id"]), :, :]

            fd_emp = fd_data[expt_ids.index(tar_md["emp_ref_id"]), :, :]

            # Subtract reference and retain the frequencies above 2 GHz
            adi_cal_cropped = (tar_fd - adi_fd)[tar_fs, :]

            # Subtract the empty reference for the boundary detection
            emp_cal_cropped = (tar_fd - fd_emp)[tar_fs, :]

            # ---------------------Dent reconstruction here--------------------

            # ToR, cs = find_bound_times_with_dent(
            #     adi_emp_cropped=emp_cal_cropped, ant_rad=ant_rad
            # )
            #
            # plot_sino_dent(
            #     fd=emp_cal_cropped,
            #     bound_times=ToR * 1e9,
            #     title="",
            #     save_str="sinogram_with_dent.png",
            #     out_dir=out_dir,
            # )
            #
            # das_adi_recon = fd_das(
            #     fd_data=emp_cal_cropped,
            #     phase_fac=phase_fac,
            #     freqs=scan_fs,
            #     worker_pool=worker_pool,
            # )
            #
            # plot_fd_img(
            #     img=np.abs(das_adi_recon),
            #     cs=cs,
            #     roi_rad=roi_rad,
            #     img_rad=roi_rad,
            #     title="",
            #     save_fig=True,
            #     save_str=os.path.join(out_dir, "das_recon_with_dent.png"),
            #     save_close=True,
            # )

            # ---------------------Dent reconstruction here--------------------

            td, ts, kernel = prepare_fd_data(
                adi_emp_cropped=emp_cal_cropped,
                ini_t=__INI_T,
                fin_t=__FIN_T,
                n_time_pts=__N_TS,
                ini_f=__INI_F,
                fin_f=__FIN_F,
                ant_rad=ant_rad,
            )

            max_avg = np.max(kernel)
            avg_peak, _ = find_peaks(kernel, height=max_avg - 1e-9)

            # corresponding intensities
            position = np.abs(td[:, ant_pos])
            # correlation
            corr = correlate(position, kernel, "same")
            # resizing correlation back corresponding to the
            # antenna position array length
            lags = correlation_lags(len(position), len(kernel), "same")
            max_corr = np.max(corr)
            corr_peaks, _ = find_peaks(corr, height=max_corr - 1e-9)

            # ------PEAK SELECTION------- #

            # positive index - average signal is "shifted" to the right wrt
            # the actual signal
            # negative - to the left
            indx_lag = lags[corr_peaks]
            # approximate anticipated peak index
            approx_peak_idx = avg_peak + indx_lag
            # find the closest actual peak index to approximate
            peaks, _ = find_peaks(position)
            peak = peaks[np.argmin(np.abs(peaks - approx_peak_idx))]
            secondary_peak = peaks[
                np.argmin(np.abs(peaks - approx_peak_idx)) + 1
            ]

            print(f"Skin peak = {ts[np.argmax(td[:, ant_pos])]} ns")
            print(f"Kernel peak = {ts[np.argmax(kernel)]} ns")
            print(
                f"Difference = {ts[np.argmax(td[:, ant_pos])] - ts[np.argmax(kernel)]} ns"
            )
            print(f"Correlation peak = {(ts - np.max(ts) / 2)[corr_peaks]} ns")
            print(
                f"""Distance = {
                    np.abs(
                        (ts - np.max(ts) / 2)[corr_peaks]
                        - ts[np.argmax(td[:, ant_pos])]
                        + ts[np.argmax(kernel)]
                    )
                    * 3e8
                } m"""
            )

            plt.figure()
            plt.rc("font", family="Libertinus Serif")

            plt.tick_params(labelsize=14)

            # plt.plot(ts * 1e9, np.abs(td[:, ant_pos]), "k-", linewidth=1.3)

            # plt.plot(ts * 1e9, np.abs(kernel), "r-", linewidth=1.3)

            plot_time = (ts - np.max(ts) / 2) * 1e9

            plt.plot(plot_time, corr, "g-", linewidth=1.3)

            plt.plot(
                plot_time[corr_peaks],
                corr[corr_peaks],
                "bX",
                label="Correlation maximum",
            )

            # plt.plot(
            #     ts[peak] * 1e9,
            #     np.abs(td[peak, ant_pos]),
            #     "rX",
            #     label="Skin peak",
            # )

            # plt.plot(
            #     ts[secondary_peak] * 1e9,
            #     np.abs(td[secondary_peak, ant_pos]),
            #     "bX",
            #     label="Secondary peak",
            # )

            plt.legend(fontsize=15)
            plt.xlabel("Lag (ns)", fontsize=18)
            plt.ylabel("Intensity (a.u.)", fontsize=18)
            plt.grid(linewidth=0.7)
            plt.tight_layout()
            # plt.show()
            plt.savefig(
                os.path.join(out_dir, "correlation.png"),
                dpi=300,
            )
            break
