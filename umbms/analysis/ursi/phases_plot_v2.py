import os

import matplotlib.pyplot as plt
import numpy as np

# The phase
__INI_F = 2e9
__FIN_F = 9e9
__N_FS = 1001
__MY_DPI = 120

glycerin_data = np.genfromtxt(
    "data/glycerin_phase.csv", skip_header=1, delimiter=","
)
glycerin_exp = glycerin_data[:, 1]
glycerin_fit = glycerin_data[:, 2]

dgbe95_data = np.genfromtxt(
    "data/dgbe95_phase.csv", skip_header=1, delimiter=","
)
dgbe95_exp = dgbe95_data[:, 1]
dgbe95_fit = dgbe95_data[:, 2]

dgbe90_data = np.genfromtxt(
    "data/dgbe90_phase.csv", skip_header=1, delimiter=","
)
dgbe90_exp = dgbe90_data[:, 1]
dgbe90_fit = dgbe90_data[:, 2]

dgbe70_data = np.genfromtxt(
    "data/dgbe70_phase.csv", skip_header=1, delimiter=","
)
dgbe70_exp = dgbe70_data[:, 1]
dgbe70_fit = dgbe70_data[:, 2]

if __name__ == "__main__":
    __MY_DPI = 120

    plt.rcParams["font.family"] = "Garamond"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["ytick.color"] = "white"

    fig, ax = plt.subplots(
        2,
        2,
        **dict(figsize=(900 / __MY_DPI, 900 / __MY_DPI), dpi=__MY_DPI),
        sharex=True,
        sharey=True,
        facecolor="#0e2841",
    )

    plot_freqs = np.linspace(2, 9, 1001)
    mask = plot_freqs > 1

    ax[0, 0].plot(plot_freqs[mask], glycerin_exp[mask], "r--", linewidth=0.9)
    ax[0, 0].plot(
        plot_freqs[mask],
        glycerin_fit[mask],
        "r-",
        label=r"Estimated speed inside, shift = $-2 \cdot 4\pi$",
        linewidth=1.3,
    )
    ax[0, 0].set_title("Glycerin", fontsize=16, fontdict={"color": "white"})

    ax[0, 1].plot(plot_freqs[mask], dgbe95_exp[mask], "r--", linewidth=0.9)
    ax[0, 1].plot(
        plot_freqs[mask],
        dgbe95_fit[mask],
        "r-",
        label=r"Estimated speed inside, shift = $-2 \cdot 4\pi$",
        linewidth=1.3,
    )
    ax[0, 1].set_title("DGBE 95%", fontsize=16, fontdict={"color": "white"})

    ax[1, 0].plot(plot_freqs[mask], dgbe90_exp[mask], "r--", linewidth=0.9)
    ax[1, 0].plot(
        plot_freqs[mask],
        dgbe90_fit[mask],
        "r-",
        label=r"Estimated speed inside, shift = $-2 \cdot 4\pi$",
        linewidth=1.3,
    )
    ax[1, 0].set_title("DGBE 90%", fontsize=16, fontdict={"color": "white"})

    ax[1, 1].plot(
        plot_freqs[mask],
        dgbe70_exp[mask],
        "r--",
        label="Experimental phase",
        linewidth=0.9,
    )
    ax[1, 1].plot(
        plot_freqs[mask],
        dgbe70_fit[mask],
        "r-",
        label=r"Fitted phase",
        linewidth=1.3,
    )
    ax[1, 1].set_title("DGBE 70%", fontsize=16, fontdict={"color": "white"})
    ax[1, 1].legend(labelcolor="k")

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
    plt.ylabel(r"Phase shift (radians)", fontsize=16, labelpad=12)

    plt.tight_layout()
    plt.show()
    # plt.savefig(
    #     "C:/Users/prikh/Desktop/phases.png",
    #     dpi=__MY_DPI,
    # )
