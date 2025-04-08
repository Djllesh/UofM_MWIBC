import pandas
import csv
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

__DATA_DIR = os.path.join(get_proj_path(), "data/umbmid/cyl_phantom/ursi_data/")
__DIEL_DIR = os.path.join(get_proj_path(), "data/freq_data/")

__FD_NAME = "20250115_s21_data.pickle"
__DIEL_NAME = "20250115_DGBE70.csv"
# The phase
__INI_F = 2e9
__FIN_F = 9e9
__N_FS = 1001
__MY_DPI = 120

data = load_pickle(os.path.join(__DATA_DIR, __FD_NAME))
freqs = np.linspace(__INI_F, __FIN_F, __N_FS)
phase = np.angle(data)
target_phase = phase[1, :, 0]
target_phase_unwrapped = np.unwrap(phase[1, :, 0])
length = 0.42
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
        x0=np.array([3.40, 17.93, 101.75, 0.18, -target_phase[0]]),
        bounds=((1, 7), (8, 80), (7, 103), (0.0, 0.25), (None, None)),
        args=(target_phase_unwrapped, freqs, length, False),
        method="trust-constr",
        options={"maxiter": 2000},
    )

    results_unwrapped = minimize(
        fun=phase_diff_MSE,
        x0=np.array(results.x),
        args=(target_phase_unwrapped, freqs, length, False),
        method="trust-constr",
        tol=1e-15,
    )

    e_h, e_s, tau, alpha, shift = results_unwrapped.x
    epsilon = cole_cole(freqs, e_h, e_s, tau, alpha)
    shape_unwrapped = phase_shape(freqs, length, epsilon, shift)

    # --- Exporting the data
    export_data = np.concatenate(
        (
            freqs[:, None],
            target_phase_unwrapped[:, None],
            shape_unwrapped[:, None],
        ),
        axis=-1,
    )

    fields = ["f", "exp_phase", "fit_phase"]
    filename = "data/dgbe70_phase.csv"
    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(export_data)

    __MY_DPI = 120
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(
        **dict(figsize=(500 / __MY_DPI, 500 / __MY_DPI), dpi=__MY_DPI)
    )

    plot_freqs = np.linspace(2, 9, 1001)

    ax.plot(plot_freqs, target_phase_unwrapped, "r--", linewidth=0.9)
    ax.plot(plot_freqs, shape_unwrapped, "r-", linewidth=1.0)
    ax.set_title("DGBE 70%", fontsize=16)

    ax.grid(linewidth=0.5)
    # ax.legend(prop={'size': 8})
    ax.set_xlabel("Frequency (GHz)", fontsize=14)
    ax.set_ylabel("Phase shift (radians)", fontsize=14)
    plt.tight_layout()
    plt.show()
    # plt.savefig('C:/Users/prikh/Desktop/dgbe70_phase.png', dpi=__MY_DPI)
