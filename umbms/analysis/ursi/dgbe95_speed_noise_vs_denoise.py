"""
Illia Prykhodko
University of Manitoba
July 11th, 2025
"""

import csv
import pandas
from scipy.optimize import minimize
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import os
from umbms import get_proj_path
from umbms.loadsave import load_pickle
from umbms.beamform.propspeed import (
    cole_cole,
    phase_shape,
    phase_diff_MSE,
    phase_shape_wrapped,
    get_breast_speed_freq,
)
from umbms.analysis.stats import ccc

import matplotlib.ticker as ticker  # <-- ADDED

__DATA_DIR = os.path.join(
    "C:/Users/prikh/Desktop/Master's thesis/ursi/data/", "DGBE95.csv"
)

__DIEL_DIR = os.path.join(get_proj_path(), "data/freq_data/")

__DIEL_NAME = "20241219_DGBE95.csv"
# The phase
phase = np.deg2rad(
    np.genfromtxt(__DATA_DIR, skip_header=3, delimiter=",")[:, 1]
)
phase_unwrapped = np.unwrap(phase)
__INI_F = 2e9
__FIN_F = 9e9
__N_FS = 1001
__MY_DPI = 120
freqs = np.linspace(__INI_F, __FIN_F, __N_FS)
length = 0.42  # + (0.0449 + 0.03618) * 2
phantom_width = 0.11
if __name__ == "__main__":
    # Read .csv file of permittivity and conductivity values
    df = pandas.read_csv(
        os.path.join(__DIEL_DIR, __DIEL_NAME),
        delimiter=";",
        decimal=",",
        skiprows=9,
    )
    diel_data_arr = df.values

    perms = np.array(diel_data_arr[:, 1])
    conds = np.array(diel_data_arr[:, 3])

    results = minimize(
        fun=phase_diff_MSE,
        x0=np.array([3.40, 17.93, 101.75, 0.18, -phase[0]]),
        bounds=((1, 7), (8, 80), (7, 103), (0.0, 0.25), (None, None)),
        args=(phase_unwrapped, freqs, length, False),
        method="trust-constr",
        options={"maxiter": 2000},
    )

    results_unwrapped = minimize(
        fun=phase_diff_MSE,
        x0=np.array(results.x),
        # bounds=((1, 7), (8, 80), (7, 103),
        #         (0.0, 0.25), (None, None)),
        args=(phase_unwrapped, freqs, length, False),
        method="trust-constr",
        tol=1e-15,
    )

    e_h, e_s, tau, alpha, shift = results_unwrapped.x
    epsilon = cole_cole(freqs, e_h, e_s, tau, alpha)
    shape_unwrapped = phase_shape(freqs, length, epsilon, shift)

    phase_speed_4pi = (
        -2 * np.pi * freqs * length / (shape_unwrapped - 2 * np.pi * 4)
    )
    phase_speed_4pi_in = phantom_width / (
        length / phase_speed_4pi - (length - phantom_width) / 3e8
    )

    phase_speed_exp = -2 * np.pi * freqs * length / (phase_unwrapped)
    phase_speed_exp_in = phantom_width / (
        length / phase_speed_exp - (length - phantom_width) / 3e8
    )

    plt.rcParams["font.family"] = "Libertinus Serif"
    __MY_DPI = 300
    fig, ax = plt.subplots(**dict(figsize=(1800 / __MY_DPI, 1800 / __MY_DPI)))

    plot_freqs = np.linspace(2, 9, 1001)

    ax.plot(
        plot_freqs,
        phase_speed_exp_in,
        "r--",
        label="Speed based on the experimental data",
        linewidth=0.9,
    )
    # ax.plot(
    #     plot_freqs,
    #     phase_speed_4pi_in,
    #     "r-",
    #     label="Speed based on the denoised data",
    #     linewidth=1.3,
    # )

    ax.set_title("DGBE 95%", fontsize=18)
    ax.ticklabel_format(style="plain", axis="y")
    ax.tick_params(labelsize=14)

    # --- ADDED: define a custom formatter, e.g., "2.30 × 10^7"
    def speed_formatter(value, pos):
        # value is the actual numeric data on the axis (like 2.3e7).
        # Convert to #.## × 10^7 style:
        plt_value = value / 1e7
        exp_str = r"$\cdot$ 10$^7$"
        return f"{plt_value:.1f}" + exp_str

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(speed_formatter))

    ax.grid(linewidth=0.5)
    # ax.legend(fontsize=15)
    ax.set_xlabel("Frequency (GHz)", fontsize=16)
    ax.set_ylabel("Propagation speed (m/s)", fontsize=16)
    plt.tight_layout()
    # plt.show()

    plt.savefig(
        "C:/Users/prikh/Desktop/Illia_thesis_main/images/propspeed/dgbe95_speed_no_shift.png",
        dpi=__MY_DPI,
    )
