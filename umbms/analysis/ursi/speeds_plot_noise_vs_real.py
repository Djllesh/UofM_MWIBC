import numpy as np
import matplotlib.pyplot as plt

# The phase
__INI_F = 2e9
__FIN_F = 9e9
__N_FS = 1001
__MY_DPI = 120

freqs = np.linspace(__INI_F, __FIN_F, __N_FS)

L = 0.42
d = 0.11
d_air = (L - d) / 2

glycerin_data_phase = np.genfromtxt(
    "data/glycerin_phase.csv", skip_header=1, delimiter=","
)
glycerin_noisy_phase = glycerin_data_phase[:, 1] - 2 * 4 * np.pi
glycerin_noisy_avg_speed = -2 * np.pi * freqs * L / glycerin_noisy_phase
glycerin_noisy_speed = (
    1e-8 * d / (L / glycerin_noisy_avg_speed - 2 * d_air / 3e8)
)

glycerin_data = 1e-8 * np.genfromtxt(
    "data/glycerin_full.csv", skip_header=1, delimiter=","
)
glycerin_exp = glycerin_data[:, 1]
glycerin_phase = glycerin_data[:, 2]

dgbe95_data_phase = np.genfromtxt(
    "data/dgbe95_phase.csv", skip_header=1, delimiter=","
)
dgbe95_noisy_phase = dgbe95_data_phase[:, 1] - 2 * 4 * np.pi
dgbe95_noisy_avg_speed = -2 * np.pi * freqs * L / dgbe95_noisy_phase
dgbe95_noisy_speed = 1e-8 * d / (L / dgbe95_noisy_avg_speed - 2 * d_air / 3e8)
dgbe95_data = 1e-8 * np.genfromtxt(
    "data/dgbe95_full.csv", skip_header=1, delimiter=","
)
dgbe95_exp = dgbe95_data[:, 1]
dgbe95_phase = dgbe95_data[:, 2]

dgbe90_data_phase = np.genfromtxt(
    "data/dgbe90_phase.csv", skip_header=1, delimiter=","
)
dgbe90_noisy_phase = dgbe90_data_phase[:, 1] - 2 * 4 * np.pi
dgbe90_noisy_avg_speed = -2 * np.pi * freqs * L / dgbe90_noisy_phase
dgbe90_noisy_speed = 1e-8 * d / (L / dgbe90_noisy_avg_speed - 2 * d_air / 3e8)
dgbe90_data = 1e-8 * np.genfromtxt(
    "data/dgbe90_full.csv", skip_header=1, delimiter=","
)
dgbe90_exp = dgbe90_data[:, 1]
dgbe90_phase = dgbe90_data[:, 2]

dgbe70_data_phase = np.genfromtxt(
    "data/dgbe70_phase.csv", skip_header=1, delimiter=","
)
dgbe70_noisy_phase = dgbe70_data_phase[:, 1] - 2 * 6 * np.pi
dgbe70_noisy_avg_speed = -2 * np.pi * freqs * L / dgbe70_noisy_phase
dgbe70_noisy_speed = 1e-8 * d / (L / dgbe70_noisy_avg_speed - 2 * d_air / 3e8)
dgbe70_data = 1e-8 * np.genfromtxt(
    "data/dgbe70_full.csv", skip_header=1, delimiter=","
)
dgbe70_exp = dgbe70_data[:, 1]
dgbe70_phase = dgbe70_data[:, 2]

if __name__ == "__main__":
    __MY_DPI = 250
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(
        2,
        2,
        figsize=(1800 / __MY_DPI, 1800 / __MY_DPI),
        sharex=True,
        sharey=True,
    )
    plot_freqs = np.linspace(2, 9, 1001)
    mask = plot_freqs < 4

    ax[0, 0].plot(plot_freqs[mask], glycerin_exp[mask], "r--", linewidth=0.9)
    ax[0, 0].plot(
        plot_freqs[mask],
        glycerin_noisy_speed[mask],
        "r-",
        label=r"Estimated speed inside, shift = $-2 \cdot 4\pi$",
        linewidth=1.3,
    )
    ax[0, 0].set_title("Glycerin", fontsize=16)

    ax[0, 1].plot(plot_freqs[mask], dgbe95_exp[mask], "r--", linewidth=0.9)
    ax[0, 1].plot(
        plot_freqs[mask],
        dgbe95_noisy_speed[mask],
        "r-",
        label=r"Estimated speed inside, shift = $-2 \cdot 4\pi$",
        linewidth=1.3,
    )
    ax[0, 1].set_title("DGBE 95%", fontsize=16)

    ax[1, 0].plot(plot_freqs[mask], dgbe90_exp[mask], "r--", linewidth=0.9)
    ax[1, 0].plot(
        plot_freqs[mask],
        dgbe90_noisy_speed[mask],
        "r-",
        label=r"Estimated speed inside, shift = $-2 \cdot 4\pi$",
        linewidth=1.3,
    )
    ax[1, 0].set_title("DGBE 90%", fontsize=16)

    ax[1, 1].plot(
        plot_freqs[mask],
        dgbe70_exp[mask],
        "r--",
        linewidth=0.9,
        label="Actual propagation speed",
    )
    ax[1, 1].plot(
        plot_freqs[mask],
        dgbe70_noisy_speed[mask],
        "r-",
        label="Experimental propagation speed",
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
    ax[1, 1].legend(prop={"size": 8})
    plt.xlabel("Frequency (GHz)", fontsize=16)
    plt.ylabel(r"Propagation speed ($10^8$m/s)", fontsize=16)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        "C:/Users/prikh/Desktop/Master's thesis/ursi/speed_noise_vs_real_cropped.png",
        dpi=__MY_DPI,
    )
