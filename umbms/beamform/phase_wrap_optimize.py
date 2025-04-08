"""
Illia Prykhodko
University of Manitoba
November 15th, 2024
"""

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
)

__DATA_DIR = os.path.join(
    get_proj_path(), "data/umbmid/cyl_phantom/speed_paper/"
)
__FIG_DIR = os.path.join(get_proj_path(), "output/cyl_phantom/")
__FD_NAME = "20240819_s21_data.pickle"

# the frequency parameters from the scan
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

if __name__ == "__main__":
    mask = freqs > 3.5e9
    mse = {}
    titles = [
        "Wrapped Phase Initial Fit",
        "Wrapped Phase on Wrapped Fit",
        "Wrapped Phase on Unwrapped Fit",
        "Unwrapped Phase Initial Fit",
        "Unwrapped Phase on Wrapped Fit",
        "Unwrapped Phase on Unwrapped Fit",
    ]

    # create a figure
    fig, ax = plt.subplots(
        2,
        3,
        sharex=True,
        sharey="row",
        **dict(figsize=(1400 / __MY_DPI, 800 / __MY_DPI), dpi=__MY_DPI),
        layout="constrained",
    )

    # create a small figure
    fig_ini, ax_ini = plt.subplots(
        2,
        1,
        sharex=True,
        sharey="row",
        **dict(figsize=(720 / __MY_DPI, 720 / __MY_DPI), dpi=__MY_DPI),
        layout="tight",
    )

    ax_ini[0].set_ylim(-10, 10)
    ax_ini[0].set_ylabel("Phase, (rad)")
    ax_ini[1].set_ylabel("Phase, (rad)")

    ax_ini[0].grid()
    ax_ini[1].grid()

    ax_ini[1].set_xlabel("Frequency, (Hz)")
    results = minimize(
        fun=phase_diff_MSE,
        x0=np.array([3.40, 17.93, 101.75, 0.18, -target_phase[0]]),
        bounds=((1, 7), (8, 80), (7, 103), (0.0, 0.25), (None, None)),
        args=(target_phase_unwrapped, freqs, length, False),
        method="trust-constr",
        options={"maxiter": 2000},
    )

    print(results)

    e_h, e_s, tau, alpha, shift = results.x
    epsilon = cole_cole(freqs, e_h, e_s, tau, alpha)
    shape = phase_shape_wrapped(freqs, length, epsilon, shift)
    shape_unwrapped = phase_shape(freqs, length, epsilon, shift)

    mse["Wrapped Phase Initial Fit"] = (1 / np.size(shape)) * (
        np.sum((target_phase - shape) ** 2)
    )
    mse["Unwrapped Phase Initial Fit"] = (1 / np.size(shape)) * (
        np.sum((target_phase_unwrapped - shape_unwrapped) ** 2)
    )

    # Big plot
    ax[0, 0].plot(
        freqs, target_phase, "k--", label="Experimental phase", linewidth=0.7
    )
    ax[0, 0].plot(freqs, shape, "k-", label="Fit", linewidth=1.5)
    ax[1, 0].plot(
        freqs,
        np.unwrap(target_phase),
        "k--",
        label="Experimental phase",
        linewidth=0.7,
    )
    ax[1, 0].plot(freqs, shape_unwrapped, "k-", label="Fit", linewidth=1.5)

    # Small plot
    ax_ini[0].plot(
        freqs, target_phase, "k--", label="Experimental phase", linewidth=0.7
    )
    ax_ini[0].plot(freqs, shape, "k-", label="Fit", linewidth=1.5)
    ax_ini[0].set_title(
        f"Wrapped Phase Initial Fit. MSE = "
        f"{mse['Wrapped Phase Initial Fit']:.3f}"
    )
    ax_ini[1].plot(
        freqs,
        np.unwrap(target_phase),
        "k--",
        label="Experimental phase",
        linewidth=0.7,
    )
    ax_ini[1].plot(freqs, shape_unwrapped, "k-", label="Fit", linewidth=1.5)
    ax_ini[1].set_title(
        f"Unwrapped Phase Initial Fit. MSE = "
        f"{mse['Unwrapped Phase Initial Fit']:.3f}"
    )

    ax_ini[1].legend()
    fig_ini.savefig(os.path.join(__FIG_DIR, "ini_fit.png"), transparent=True)
    fig_ini.clf()
    # Now do the wrapped optimization, but using the guess from the
    # unwrapped one

    results_wrapped = minimize(
        fun=phase_diff_MSE,
        x0=np.array(results.x),
        # bounds=((1, 7), (8, 80), (7, 103),
        #         (0.0, 0.25), (None, None)),
        args=(target_phase, freqs, length, True),
        method="trust-constr",
    )

    e_h, e_s, tau, alpha, shift = results_wrapped.x
    epsilon = cole_cole(freqs, e_h, e_s, tau, alpha)
    shape = phase_shape_wrapped(freqs, length, epsilon, shift)
    shape_unwrapped = phase_shape(freqs, length, epsilon, shift)

    print(results_wrapped)

    mse["Wrapped Phase on Wrapped Fit"] = (1 / np.size(shape)) * (
        np.sum((target_phase - shape) ** 2)
    )
    mse["Unwrapped Phase on Wrapped Fit"] = (1 / np.size(shape)) * (
        np.sum((target_phase_unwrapped - shape_unwrapped) ** 2)
    )

    # create a small figure
    fig_ini, ax_ini = plt.subplots(
        2,
        1,
        sharex=True,
        sharey="row",
        **dict(figsize=(720 / __MY_DPI, 720 / __MY_DPI), dpi=__MY_DPI),
        layout="tight",
    )

    ax_ini[0].set_ylim(-10, 10)
    ax_ini[0].set_ylabel("Phase, (rad)")
    ax_ini[1].set_ylabel("Phase, (rad)")

    ax_ini[0].grid()
    ax_ini[1].grid()

    ax_ini[1].set_xlabel("Frequency, (Hz)")

    # Big plot
    ax[0, 1].plot(
        freqs, target_phase, "k--", label="Experimental phase", linewidth=0.7
    )
    ax[0, 1].plot(freqs, shape, "k-", label="Fit", linewidth=1.5)
    ax[1, 1].plot(
        freqs,
        np.unwrap(target_phase),
        "k--",
        label="Experimental phase",
        linewidth=0.7,
    )
    ax[1, 1].plot(freqs, shape_unwrapped, "k-", label="Fit", linewidth=1.5)

    # Small plot
    ax_ini[0].plot(
        freqs, target_phase, "k--", label="Experimental phase", linewidth=0.7
    )
    ax_ini[0].plot(freqs, shape, "k-", label="Fit", linewidth=1.5)
    ax_ini[0].set_title(
        f"Wrapped Phase on Wrapped Fit. MSE = "
        f"{mse['Wrapped Phase on Wrapped Fit']:.3f}"
    )
    ax_ini[1].plot(
        freqs,
        np.unwrap(target_phase),
        "k--",
        label="Experimental phase",
        linewidth=0.7,
    )
    ax_ini[1].plot(freqs, shape_unwrapped, "k-", label="Fit", linewidth=1.5)
    ax_ini[1].set_title(
        f"Unwrapped Phase on Wrapped Fit. MSE = "
        f"{mse['Unwrapped Phase on Wrapped Fit']:.3f}"
    )

    ax_ini[1].legend()
    fig_ini.savefig(os.path.join(__FIG_DIR, "wrap_fit.png"), transparent=True)
    fig_ini.clf()

    results_unwrapped = minimize(
        fun=phase_diff_MSE,
        x0=np.array(results.x),
        # bounds=((1, 7), (8, 80), (7, 103),
        #         (0.0, 0.25), (None, None)),
        args=(target_phase_unwrapped, freqs, length, False),
        method="trust-constr",
        tol=1e-15,
    )

    print(results_unwrapped)

    e_h, e_s, tau, alpha, shift = results_unwrapped.x
    epsilon = cole_cole(freqs, e_h, e_s, tau, alpha)
    shape = phase_shape_wrapped(freqs, length, epsilon, shift)
    shape_unwrapped = phase_shape(freqs, length, epsilon, shift)

    fig_ini, ax_ini = plt.subplots(
        2,
        1,
        sharex=True,
        sharey="row",
        **dict(figsize=(720 / __MY_DPI, 720 / __MY_DPI), dpi=__MY_DPI),
        layout="tight",
    )

    ax_ini[0].set_ylim(-10, 10)
    ax_ini[0].set_ylabel("Phase, (rad)")
    ax_ini[1].set_ylabel("Phase, (rad)")

    ax_ini[0].grid()
    ax_ini[1].grid()

    ax_ini[1].set_xlabel("Frequency, (Hz)")
    mse["Wrapped Phase on Unwrapped Fit"] = (1 / np.size(shape)) * (
        np.sum((target_phase - shape) ** 2)
    )
    mse["Unwrapped Phase on Unwrapped Fit"] = (1 / np.size(shape)) * (
        np.sum((target_phase_unwrapped - shape_unwrapped) ** 2)
    )

    # Big plot
    ax[0, 2].plot(
        freqs, target_phase, "k--", label="Experimental phase", linewidth=0.7
    )
    ax[0, 2].plot(freqs, shape, "k-", label="Fit", linewidth=1.5)
    ax[1, 2].plot(
        freqs,
        np.unwrap(target_phase),
        "k--",
        label="Experimental phase",
        linewidth=0.7,
    )
    ax[1, 2].plot(freqs, shape_unwrapped, "k-", label="Fit", linewidth=1.5)

    # Small plot
    ax_ini[0].plot(
        freqs, target_phase, "k--", label="Experimental phase", linewidth=0.7
    )
    ax_ini[0].plot(freqs, shape, "k-", label="Fit", linewidth=1.5)
    ax_ini[0].set_title(
        f"Wrapped Phase on Unwrapped Fit. MSE = "
        f"{mse['Wrapped Phase on Unwrapped Fit']:.3f}"
    )
    ax_ini[1].plot(
        freqs,
        np.unwrap(target_phase),
        "k--",
        label="Experimental phase",
        linewidth=0.7,
    )
    ax_ini[1].plot(freqs, shape_unwrapped, "k-", label="Fit", linewidth=1.5)
    ax_ini[1].set_title(
        f"Unwrapped Phase on Unwrapped Fit. MSE = "
        f"{mse['Unwrapped Phase on Unwrapped Fit']:.3f}"
    )

    ax_ini[1].legend()
    fig_ini.savefig(os.path.join(__FIG_DIR, "unwrap_fit.png"), transparent=True)
    fig_ini.clf()

    # Create the grid and the legend on every subplot
    for idx, axis in enumerate(ax.flat):
        axis.grid()
        axis.set_title(
            f"{titles[idx]} MSE = {mse[titles[idx]]:.3f}", fontsize=10
        )

    for axis in ax[0]:
        axis.set_ylim(-10, 10)

    ax[0, 0].set_ylabel("Phase (rad)")
    ax[1, 0].set_ylabel("Phase (rad)")

    plt.xlabel("Frequency (Hz)")
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax[0, 0]]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="outside lower center", ncols=2)
    plt.tight_layout()
    plt.show()
    # fig.savefig(os.path.join(__FIG_DIR, 'big_fig.png'), transparent=True)
