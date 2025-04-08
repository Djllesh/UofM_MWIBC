"""
Illia Prykhodko

Univerity of Manitoba,
April 8th, 2025
"""

from scipy.constants import c
import numpy as np
import matplotlib.pyplot as plt
import os

D = 0.42
phantom_width = 0.11
data_path = os.path.join(os.path.expanduser("~"), "Desktop/Exp data/20250407/")
freqs = np.linspace(2e9, 9e9, 1001)

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

dgbe95_avg_time = np.mean(dgbe95_time)

dgbe90_time = np.genfromtxt(
    os.path.join(data_path, "dgbe90_time.csv"),
    delimiter=",",
    skip_header=3,
)[:, 1]

dgbe90_avg_time = np.mean(dgbe90_time)

dgbe70_time = np.genfromtxt(
    os.path.join(data_path, "dgbe70_time.csv"),
    delimiter=",",
    skip_header=3,
)[:, 1]

dgbe70_avg_time = np.mean(dgbe70_time)


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


if __name__ == "__main__":
    # Empty chamber ===================================================
    fig, ax = plt.subplots(1, 2, sharex=True)
    ax[0].plot(
        freqs,
        empty_chamber_time_1hr,
        "b-",
        linewidth=1.2,
        label="Initial",
    )

    ax[0].plot(
        freqs,
        empty_chamber_time_fix,
        "r-",
        linewidth=1.2,
        label="Adjust",
    )

    ax[1].hlines(one_hr_avg_time, xmin=2e9, xmax=9e9, color="b")
    ax[1].hlines(fix_avg_time, xmin=2e9, xmax=9e9, color="r")

    print(
        f"Difference in separation after the fix = {
            100 * (fix_avg_time - one_hr_avg_time) * c
        } cm"
    )

    print(f"Separation if c = {100 * one_hr_avg_time * c} cm")
    print(f"Separation diff = {100 * (one_hr_avg_time * c - D)} cm")

    ax[0].set_xlabel("Frequency, Hz")
    ax[0].set_ylabel("Time delay, s")

    ax[0].legend()
    ax[0].grid(linewidth=0.9)
    ax[1].grid(linewidth=0.9)
    plt.tight_layout()
    plt.show()
    plt.close()

    # Materials =======================================================
    fig, ax = plt.subplots(1, 2)

    ax[0].plot(
        freqs,
        dgbe95_time,
        "b-",
        linewidth=1.2,
        label="DGBE 95%",
    )

    ax[0].plot(
        freqs,
        dgbe90_time,
        "k-",
        linewidth=1.2,
        label="DGBE 90%",
    )

    ax[0].plot(
        freqs,
        dgbe70_time,
        "r-",
        linewidth=1.2,
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

    ax[0].legend()
    ax[0].grid(linewidth=0.9)
    ax[1].grid(linewidth=0.9)
    plt.tight_layout()
    plt.show()
