"""
Illia Prykhodko

University of Manitoba
January 15th, 2025
"""

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


__DATA_DIR = os.path.join(get_proj_path(), "data/umbmid/cyl_phantom/")
__DIEL_DIR = os.path.join(get_proj_path(), "data/freq_data/")
__FIG_DIR = os.path.join(get_proj_path(), "output/cyl_phantom/")
__FD_NAME = "20250115_s21_data.pickle"

# the frequency parameters from the scan
__INI_F = 2e9
__FIN_F = 9e9
__N_FS = 1001
__MY_DPI = 120

data = load_pickle(os.path.join(__DATA_DIR, __FD_NAME))
freqs = np.linspace(__INI_F, __FIN_F, __N_FS)
phase = np.angle(data)
target_phase = phase[0, :, 0]
target_phase_unwrapped = np.unwrap(phase[0, :, 0])
length = 0.42
phantom_width = 0.11

if __name__ == "__main__":
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
        # bounds=((1, 7), (8, 80), (7, 103),
        #         (0.0, 0.25), (None, None)),
        args=(target_phase_unwrapped, freqs, length, False),
        method="trust-constr",
        tol=1e-15,
    )

    e_h, e_s, tau, alpha, shift = results_unwrapped.x
    epsilon = cole_cole(freqs, e_h, e_s, tau, alpha)
    shape_unwrapped = phase_shape(freqs, length, epsilon, shift)

    raw_speed_2pi = (
        -2 * np.pi * freqs * length / (target_phase_unwrapped - 2 * np.pi * 2)
    )

    raw_speed_3pi = (
        -2 * np.pi * freqs * length / (target_phase_unwrapped - 2 * np.pi * 3)
    )

    phase_speed_2pi = (
        -2 * np.pi * freqs * length / (shape_unwrapped - 2 * np.pi * 2)
    )

    phase_speed_3pi = (
        -2 * np.pi * freqs * length / (shape_unwrapped - 2 * np.pi * 3)
    )
    __MY_DPI = 120
    fig, ax = plt.subplots(
        **dict(figsize=(1500 / __MY_DPI, 720 / __MY_DPI), dpi=__MY_DPI)
    )

    ax.plot(
        freqs,
        raw_speed_2pi,
        "r--",
        label=r"Unfitted speed "
        r"$-2 \cdot 2\pi$",
        linewidth=0.7,
    )
    ax.plot(
        freqs,
        raw_speed_3pi,
        "b--",
        label=r"Unfitted speed "
        r"$-2 \cdot 3\pi$",
        linewidth=0.7,
    )

    ax.plot(
        freqs,
        phase_speed_2pi,
        "r--",
        label=r"Extracted speed, shift = "
        r"$-2 \cdot 2\pi$",
        linewidth=1.2,
    )
    ax.plot(
        freqs,
        phase_speed_3pi,
        "b--",
        label=r"Extracted speed, shift = "
        r"$-2 \cdot 3\pi$",
        linewidth=1.2,
    )

    ax.axhline(y=3e8, color="r", linestyle="-", linewidth=1.6)
    ax.grid()
    ax.legend(prop={"size": 8})
    ax.set_xlabel("Frequency, (Hz)")
    ax.set_ylabel("Propagation speed, (m/s)")
    plt.tight_layout()
    plt.show()

