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

glycerin_data = 1e-8 * np.genfromtxt(
    "data/glycerin_full.csv", skip_header=1, delimiter=","
)
glycerin_exp = glycerin_data[:, 1]
glycerin_phase = glycerin_data[:, 2]

dgbe95_data = 1e-8 * np.genfromtxt(
    "data/dgbe95_full.csv", skip_header=1, delimiter=","
)
dgbe95_exp = dgbe95_data[:, 1]
dgbe95_phase = dgbe95_data[:, 2]

dgbe90_data = 1e-8 * np.genfromtxt(
    "data/dgbe90_full.csv", skip_header=1, delimiter=","
)
dgbe90_exp = dgbe90_data[:, 1]
dgbe90_phase = dgbe90_data[:, 2]

dgbe70_data = 1e-8 * np.genfromtxt(
    "data/dgbe70_full.csv", skip_header=1, delimiter=","
)
dgbe70_exp = dgbe70_data[:, 1]
dgbe70_phase = dgbe70_data[:, 2]

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

    ax[0, 0].plot(plot_freqs, glycerin_exp, "k-", linewidth=1.3)
    ax[0, 0].plot(
        plot_freqs,
        glycerin_phase,
        "r-",
        label=r"Estimated speed inside, shift = $-2 \cdot 4\pi$",
        linewidth=1.3,
    )
    ax[0, 0].set_title("Glycerin", fontsize=16)

    ax[0, 1].plot(plot_freqs, dgbe95_exp, "k-", linewidth=1.3)
    ax[0, 1].plot(
        plot_freqs,
        dgbe95_phase,
        "r-",
        label=r"Estimated speed inside, shift = $-2 \cdot 4\pi$",
        linewidth=1.3,
    )
    ax[0, 1].set_title("DGBE 95%", fontsize=16)

    ax[1, 0].plot(plot_freqs, dgbe90_exp, "k-", linewidth=1.3)
    ax[1, 0].plot(
        plot_freqs,
        dgbe90_phase,
        "r-",
        label=r"Estimated speed inside, shift = $-2 \cdot 4\pi$",
        linewidth=1.3,
    )
    ax[1, 0].set_title("DGBE 90%", fontsize=16)

    ax[1, 1].plot(plot_freqs, dgbe70_exp, "k-", linewidth=1.3)
    ax[1, 1].plot(
        plot_freqs,
        dgbe70_phase,
        "r-",
        label=r"Estimated speed inside, shift = $-2 \cdot 4\pi$",
        linewidth=1.3,
    )
    ax[1, 1].set_title("DGBE 70%", fontsize=16)

    # --- ADDED: define a custom formatter, e.g., "2.30 × 10^7"
    def speed_formatter(value, pos):
        # value is the actual numeric data on the axis (like 2.3e7).
        # Convert to #.## × 10^7 style:
        plt_value = value / 1e7
        exp_str = r"$\cdot$ 10$^7$"
        return f"{plt_value:.1f}" + exp_str

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
    plt.ylabel(r"Propagation speed (10$^8$m/s)", fontsize=16)
    plt.tight_layout()
    plt.show()
    # plt.savefig('C:/Users/prikh/Desktop/speeds_full.png', dpi=__MY_DPI)
