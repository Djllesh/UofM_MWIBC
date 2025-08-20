"""
Illia Prykhodko
University of Manitoba
July 11th, 2025
"""

import csv
import pandas
from scipy.optimize import minimize
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

__DATA_DIR_DGBE95 = os.path.join(
    "C:/Users/prikh/Desktop/Master's thesis/ursi/data/", "DGBE95.csv"
)

__DATA_DIR_EMPTY = os.path.join(
    "C:/Users/prikh/Desktop/MWIBC/UofM_MWIBC/data/"
    "umbmid/cyl_phantom/speed_paper/second_attempt/",
    "s21_big_dataset_correction.pickle",
)

# The phase
phase = np.deg2rad(
    np.genfromtxt(__DATA_DIR_DGBE95, skip_header=3, delimiter=",")[:, 1]
)
__INI_F = 2e9
__FIN_F = 9e9
__N_FS = 1001
__MY_DPI = 120
freqs = np.linspace(__INI_F, __FIN_F, __N_FS)
length = 0.42  # + (0.0449 + 0.03618) * 2
phantom_width = 0.11

fd_data = load_pickle(__DATA_DIR_EMPTY)
s21_data_empty = fd_data[0, :, 0]
phase_empty = np.angle(s21_data_empty)

if __name__ == "__main__":
    __MY_DPI = 300

    plt.rcParams["font.family"] = "Libertinus Serif"
    plt.figure()
    plot_freqs = np.linspace(2, 9, 1001)
    plt.plot(plot_freqs, phase, "r-", linewidth=0.9)

    plt.title("DGBE 95%", fontsize=18)
    plt.grid(linewidth=0.5)
    # plt.legend(prop={'size': 8})
    plt.xlabel("Frequency (GHz)", fontsize=16)
    plt.ylabel("Phase shift (radians)", fontsize=16)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        "C:/Users/prikh/Desktop/Illia_thesis_main/images/propspeed/dgbe95_phase_wrapped.png",
        dpi=__MY_DPI,
    )
    plt.close()

    plt.rcParams["font.family"] = "Libertinus Serif"
    plt.figure()
    plot_freqs = np.linspace(2, 9, 1001)
    plt.plot(plot_freqs, phase_empty, "r-", linewidth=0.9)

    plt.title("Empty chamber", fontsize=18)
    plt.grid(linewidth=0.5)
    # plt.legend(prop={'size': 8})
    plt.xlabel("Frequency (GHz)", fontsize=16)
    plt.ylabel("Phase shift (radians)", fontsize=16)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        "C:/Users/prikh/Desktop/Illia_thesis_main/images/propspeed/empty_chamber_unwrapped_phase.png",
        dpi=__MY_DPI,
    )
