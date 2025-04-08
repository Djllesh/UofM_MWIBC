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

# The phase
__INI_F = 2e9
__FIN_F = 9e9
__N_FS = 1001
__MY_DPI = 120
length = 0.42
phantom_width = 0.11
freqs = np.linspace(2e9, 9e9, 1001)

glycerin_data = np.genfromtxt(
    "data/glycerin_phase.csv", skip_header=1, delimiter=","
)
glycerin_data_speed = 1e-8 * np.genfromtxt(
    "data/glycerin_full.csv", skip_header=1, delimiter=","
)
glycerin_exp = glycerin_data[:, 1]
glycerin_speed = glycerin_data_speed[:, 1]
glycerin_speed_raw = (
    -2 * np.pi * freqs * length / (glycerin_exp - 2 * np.pi * 4)
)
glycerin_speed_raw = 1e-8 * (
    phantom_width
    / (length / glycerin_speed_raw - (length - phantom_width) / 3e8)
)

dgbe95_data = np.genfromtxt(
    "data/dgbe95_phase.csv", skip_header=1, delimiter=","
)
dgbe95_data_speed = 1e-8 * np.genfromtxt(
    "data/dgbe95_full.csv", skip_header=1, delimiter=","
)
dgbe95_exp = dgbe95_data[:, 1]
dgbe95_speed = dgbe95_data_speed[:, 1]
dgbe95_speed_raw = -2 * np.pi * freqs * length / (dgbe95_exp - 2 * np.pi * 4)
dgbe95_speed_raw = 1e-8 * (
    phantom_width / (length / dgbe95_speed_raw - (length - phantom_width) / 3e8)
)

dgbe90_data = np.genfromtxt(
    "data/dgbe90_phase.csv", skip_header=1, delimiter=","
)
dgbe90_data_speed = 1e-8 * np.genfromtxt(
    "data/dgbe90_full.csv", skip_header=1, delimiter=","
)
dgbe90_exp = dgbe90_data[:, 1]
dgbe90_speed = dgbe90_data_speed[:, 1]
dgbe90_speed_raw = -2 * np.pi * freqs * length / (dgbe90_exp - 2 * np.pi * 4)
dgbe90_speed_raw = 1e-8 * (
    phantom_width / (length / dgbe90_speed_raw - (length - phantom_width) / 3e8)
)

dgbe70_data = np.genfromtxt(
    "data/dgbe70_phase.csv", skip_header=1, delimiter=","
)
dgbe70_data_speed = 1e-8 * np.genfromtxt(
    "data/dgbe70_full.csv", skip_header=1, delimiter=","
)
dgbe70_exp = dgbe70_data[:, 1]
dgbe70_speed = dgbe70_data_speed[:, 1]
dgbe70_speed_raw = -2 * np.pi * freqs * length / (dgbe70_exp - 2 * np.pi * 6)
dgbe70_speed_raw = 1e-8 * (
    phantom_width / (length / dgbe70_speed_raw - (length - phantom_width) / 3e8)
)

if __name__ == "__main__":
    __MY_DPI = 120
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(
        2,
        2,
        **dict(figsize=(900 / __MY_DPI, 900 / __MY_DPI), dpi=__MY_DPI),
        sharex=True,
        sharey=True,
    )
    plot_freqs = np.linspace(2, 9, 1001)
    mask = plot_freqs <= 4

    ax[0, 0].plot(plot_freqs, glycerin_speed, "r--", linewidth=0.9)
    ax[0, 0].plot(
        plot_freqs,
        glycerin_speed_raw,
        "r-",
        label=r"Estimated speed inside, shift = $-2 \cdot 4\pi$",
        linewidth=1.3,
    )
    ax[0, 0].set_title("Glycerin", fontsize=16)

    ax[0, 1].plot(plot_freqs, dgbe95_speed, "r--", linewidth=0.9)
    ax[0, 1].plot(
        plot_freqs,
        dgbe95_speed_raw,
        "r-",
        label=r"Estimated speed inside, shift = $-2 \cdot 4\pi$",
        linewidth=1.3,
    )
    ax[0, 1].set_title("DGBE 95%", fontsize=16)

    ax[1, 0].plot(plot_freqs, dgbe90_speed, "r--", linewidth=0.9)
    ax[1, 0].plot(
        plot_freqs,
        dgbe90_speed_raw,
        "r-",
        label=r"Estimated speed inside, shift = $-2 \cdot 4\pi$",
        linewidth=1.3,
    )
    ax[1, 0].set_title("DGBE 90%", fontsize=16)

    ax[1, 1].plot(
        plot_freqs,
        dgbe70_speed,
        "r--",
        label="Actual propagation speed",
        linewidth=0.9,
    )
    ax[1, 1].plot(
        plot_freqs,
        dgbe70_speed_raw,
        "r-",
        label=r"Experimental propagation speed",
        linewidth=1.3,
    )
    ax[1, 1].set_title("DGBE 70%", fontsize=16)
    ax[1, 1].legend()
    for axis in ax.flatten():
        # axis.ticklabel_format(style='plain', axis='y')
        # axis.yaxis.set_major_formatter(ticker.FuncFormatter(speed_formatter))
        axis.grid(linewidth=0.5)

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    # ax.legend(prop={'size': 8})
    plt.xlabel("Frequency (GHz)", fontsize=16)
    plt.ylabel(r"Propagation speed (m/s)", fontsize=16, labelpad=12)

    plt.tight_layout()
    # plt.show()
    plt.savefig("C:/Users/prikh/Desktop/speed_noise.png", dpi=__MY_DPI)

