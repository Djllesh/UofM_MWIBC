"""
Illia Prykhodko

Univerity of Manitoba,
April 8th, 2025
"""

from scipy.constants import c
from scipy.optimize import minimize
from umbms import get_proj_path
from umbms.beamform.propspeed import (
    speed_diff_MSE,
    get_speed_from_epsilon,
    cole_cole,
)
import numpy as np
import matplotlib.pyplot as plt
import os

D = 0.42
phantom_width = 0.11
data_path = os.path.join(os.path.expanduser("~"), "Desktop/Exp data/20250407/")
prop_speed_data_path = os.path.join(
    get_proj_path(), "umbms/analysis/ursi/data/"
)
freqs = np.linspace(2e9, 9e9, 1001)


def calculate_propagation_speed(td, thickness, D):
    """Calculates the propagation speed inside of the dielectric
    material based on the time delay and the thickness of the material

    Parameters:
    td : array-like
        Time delay
    thickness : float
        Material thickness in meters
    D : float
        Antenna separation in meters
    """

    return thickness / (td - (D - thickness) / c)


empty_chamber_time_init = np.genfromtxt(
    os.path.join(data_path, "empty_chamber_time.csv"),
    delimiter=",",
    skip_header=3,
)[:, 1]

init_avg_time = np.mean(empty_chamber_time_init)

empty_chamber_time_1hr = np.genfromtxt(
    os.path.join(data_path, "empty_chamber_time_1hr.csv"),
    delimiter=",",
    skip_header=3,
)[:, 1]

one_hr_avg_time = np.mean(empty_chamber_time_1hr)

empty_chamber_time_fix = np.genfromtxt(
    os.path.join(data_path, "empty_chamber_time_fix.csv"),
    delimiter=",",
    skip_header=3,
)[:, 1]

fix_avg_time = np.mean(empty_chamber_time_fix)


dgbe95_time = np.genfromtxt(
    os.path.join(data_path, "dgbe95_time.csv"),
    delimiter=",",
    skip_header=3,
)[:, 1]

dgbe95_speed_true = np.genfromtxt(
    os.path.join(prop_speed_data_path, "dgbe95_full.csv"),
    delimiter=",",
    skip_header=1,
)[:, 1]

dgbe95_avg_time = np.mean(dgbe95_time)

dgbe90_time = np.genfromtxt(
    os.path.join(data_path, "dgbe90_time.csv"),
    delimiter=",",
    skip_header=3,
)[:, 1]

dgbe90_speed_true = np.genfromtxt(
    os.path.join(prop_speed_data_path, "dgbe90_full.csv"),
    delimiter=",",
    skip_header=1,
)[:, 1]

dgbe90_avg_time = np.mean(dgbe90_time)

dgbe70_time = np.genfromtxt(
    os.path.join(data_path, "dgbe70_time.csv"),
    delimiter=",",
    skip_header=3,
)[:, 1]

dgbe70_speed_true = np.genfromtxt(
    os.path.join(prop_speed_data_path, "dgbe70_full.csv"),
    delimiter=",",
    skip_header=1,
)[:, 1]

dgbe70_avg_time = np.mean(dgbe70_time)

# Assume the experimental lenght (speed = c)
exp_length = fix_avg_time * c

empty_chamber_time_1hr_speed = one_hr_avg_time * c / empty_chamber_time_1hr
empty_chamber_time_fix_speed = exp_length / empty_chamber_time_fix

dgbe95_time_speed = calculate_propagation_speed(
    dgbe95_time, thickness=phantom_width, D=exp_length
)
dgbe90_time_speed = calculate_propagation_speed(
    dgbe90_time, thickness=phantom_width, D=exp_length
)
dgbe70_time_speed = calculate_propagation_speed(
    dgbe70_time, thickness=phantom_width, D=exp_length
)

args = [
    (
        empty_chamber_time_1hr_speed,
        freqs,
        exp_length,
    ),
    (
        empty_chamber_time_fix_speed,
        freqs,
        exp_length,
    ),
    (
        dgbe95_time_speed,
        freqs,
        exp_length,
    ),
    (
        dgbe90_time_speed,
        freqs,
        exp_length,
    ),
    (
        dgbe70_time_speed,
        freqs,
        exp_length,
    ),
]


# List of fit results
results = [
    minimize(
        fun=speed_diff_MSE,
        x0=np.array([3.40, 17.93, 101.75, 0.18]),
        bounds=((1, 7), (8, 80), (7, 103), (0.0, 0.25)),
        args=arg,
        method="Nelder-Mead",
        tol=1e-5,
    )
    for arg in args
]

(
    fit_speed_empty,
    fit_speed_fix,
    fit_speed_dgbe95,
    fit_speed_dgbe90,
    fit_speed_dgbe70,
) = (get_speed_from_epsilon(cole_cole(freqs, *result.x)) for result in results)


if __name__ == "__main__":
    # Empty chamber ===================================================
    MY_DPI = 120

    fig, ax = plt.subplots(
        1, 2, figsize=(1000 / MY_DPI, 800 / MY_DPI), sharex=True
    )

    ax[0].plot(
        freqs,
        empty_chamber_time_1hr,
        "b--",
        linewidth=0.5,
        label="Initial",
    )

    # ax[0].plot(freqs, fit_speed_empty, "r-", linewidth=1.3, label="Fit")

    ax[0].plot(
        freqs,
        empty_chamber_time_fix,
        "r--",
        linewidth=0.5,
        label="Adjust",
    )

    # ax[0].plot(freqs, fit_speed_fix, "r-", linewidth=1.3, label="Fit")

    ax[1].hlines(one_hr_avg_time, xmin=2e9, xmax=9e9, color="b")
    ax[1].hlines(fix_avg_time, xmin=2e9, xmax=9e9, color="r")

    print(
        f"Difference in separation after the fix = {
            100 * (fix_avg_time - one_hr_avg_time) * c
        } cm"
    )

    print(f"Separation if c = {100 * fix_avg_time * c} cm")
    print(f"Separation diff = {100 * (fix_avg_time * c - D)} cm")

    ax[0].set_xlabel("Frequency, Hz")
    ax[0].set_ylabel("Time delay, s")

    ax[0].set_title("Experimental data")
    ax[1].set_title("Averages")

    ax[0].legend()
    ax[0].grid(linewidth=0.9)
    ax[1].grid(linewidth=0.9)
    plt.tight_layout()
    # plt.show()

    # plt.savefig(
    #     os.path.join(os.path.expanduser("~"), "Desktop/init_vs_adjust.png"),
    #     dpi=MY_DPI,
    # )

    plt.close()

    # Materials =======================================================
    fig, ax = plt.subplots(
        1, 2, figsize=(1000 / MY_DPI, 800 / MY_DPI), sharex=True
    )

    ax[0].plot(
        freqs,
        dgbe95_time,
        "b--",
        linewidth=0.5,
        label="DGBE 95%",
    )

    ax[0].plot(
        freqs,
        dgbe90_time,
        "k--",
        linewidth=0.5,
        label="DGBE 90%",
    )

    ax[0].plot(
        freqs,
        dgbe70_time,
        "r--",
        linewidth=0.5,
        label="DGBE 70%",
    )

    ax[1].hlines(dgbe95_avg_time, xmin=2e9, xmax=9e9, color="b")
    ax[1].hlines(dgbe90_avg_time, xmin=2e9, xmax=9e9, color="k")
    ax[1].hlines(dgbe70_avg_time, xmin=2e9, xmax=9e9, color="r")

    print(
        f"DGBE 70% = {
            calculate_propagation_speed(
                dgbe70_avg_time, thickness=phantom_width, D=D
            )
        }\nDGBE 90% = {
            calculate_propagation_speed(
                dgbe90_avg_time, thickness=phantom_width, D=D
            )
        }\nDGBE 95% = {
            calculate_propagation_speed(
                dgbe95_avg_time, thickness=phantom_width, D=D
            )
        }"
    )

    ax[0].set_xlabel("Frequency, Hz")
    ax[0].set_ylabel("Time delay, s")

    ax[0].set_title("Experimental data")
    ax[1].set_title("Averages")

    ax[0].legend()
    ax[0].grid(linewidth=0.9)
    ax[1].grid(linewidth=0.9)
    plt.tight_layout()
    # plt.show()

    # plt.savefig(
    #     os.path.join(os.path.expanduser("~"), "Desktop/dielectric_td.png"),
    #     dpi=MY_DPI,
    # )

    plt.close()

    # Propagation speed empty chamber =================================
    fig, ax = plt.subplots(
        1, 1, figsize=(1000 / MY_DPI, 800 / MY_DPI), sharex=True
    )

    ax.plot(freqs, empty_chamber_time_1hr_speed, "b--", linewidth=0.3)
    ax.plot(freqs, fit_speed_empty, "b-", linewidth=1.3)

    ax.plot(freqs, empty_chamber_time_fix_speed, "r--", linewidth=0.3)
    ax.plot(freqs, fit_speed_fix, "r-", linewidth=1.3)

    plt.tight_layout()
    # plt.show()
    plt.close()

    # Propagation speed dielectrics ===================================
    fig, ax = plt.subplots(
        1, 1, figsize=(1000 / MY_DPI, 800 / MY_DPI), sharex=True
    )

    ax.plot(
        freqs, dgbe95_time_speed, "b--", linewidth=0.3, label="DGBE 95% fit"
    )
    ax.plot(
        freqs,
        fit_speed_dgbe95,
        "b-",
        linewidth=1.3,
        label="DGBE 95% experiment",
    )

    ax.plot(
        freqs, dgbe90_time_speed, "k--", linewidth=0.3, label="DGBE 90% fit"
    )
    ax.plot(
        freqs,
        fit_speed_dgbe90,
        "k-",
        linewidth=1.3,
        label="DGBE 90% experiment",
    )

    ax.plot(
        freqs, dgbe70_time_speed, "r--", linewidth=0.3, label="DGBE 70% fit"
    )
    ax.plot(
        freqs,
        fit_speed_dgbe70,
        "r-",
        linewidth=1.3,
        label="DGBE 70% experiment",
    )

    ax.set_xlabel("Frequency, Hz")
    ax.set_ylabel("Propagation speed, m/s")
    ax.legend(loc="lower left")

    ax.grid(linewidth=0.9)

    plt.tight_layout()
    # plt.show()

    # plt.savefig(
    #     os.path.join(
    #         os.path.expanduser("~"), "Desktop/dielectric_speed_vs_fit.png"
    #     ),
    #     dpi=MY_DPI,
    # )

    plt.close()

    # Propagation speed dielectrics ===================================
    fig, ax = plt.subplots(
        1, 1, figsize=(1000 / MY_DPI, 800 / MY_DPI), sharex=True
    )

    ax.plot(freqs, fit_speed_dgbe95, "b-", linewidth=1.0, label="DGBE 95% fit")
    ax.plot(
        freqs, dgbe95_speed_true, "b--", linewidth=1.3, label="DGBE 95% true"
    )

    ax.plot(freqs, fit_speed_dgbe90, "k-", linewidth=1.0, label="DGBE 90% fit")
    ax.plot(
        freqs, dgbe90_speed_true, "k--", linewidth=1.3, label="DGBE 90% true"
    )

    ax.plot(freqs, fit_speed_dgbe70, "r-", linewidth=1.0, label="DGBE 70% fit")
    ax.plot(
        freqs, dgbe70_speed_true, "r--", linewidth=1.3, label="DGBE 70% true"
    )

    # WARN: the averages look good only under the condition that D=D,
    # which is not true due to the phase shift.

    # TODO: find out why the values are so high if we use the
    # "corrected" separation

    # ax.hlines(
    #     calculate_propagation_speed(
    #         dgbe95_avg_time, thickness=phantom_width, D=D
    #     ),
    #     xmin=2e9,
    #     xmax=9e9,
    #     color="b",
    #     label="DGBE 95% average",
    # )
    # ax.hlines(
    #     calculate_propagation_speed(
    #         dgbe90_avg_time, thickness=phantom_width, D=D
    #     ),
    #     xmin=2e9,
    #     xmax=9e9,
    #     color="k",
    #     label="DGBE 90% average",
    # )
    # ax.hlines(
    #     calculate_propagation_speed(
    #         dgbe70_avg_time, thickness=phantom_width, D=D
    #     ),
    #     xmin=2e9,
    #     xmax=9e9,
    #     color="r",
    #     label="DGBE 70% average",
    # )

    ax.set_xlabel("Frequency, Hz")
    ax.set_ylabel("Propagation speed, m/s")

    ax.grid(linewidth=0.9)

    ax.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(os.path.expanduser("~"), "Desktop/true_vs_fit.png"),
        dpi=MY_DPI,
    )

    # plt.show()
    # plt.close()
