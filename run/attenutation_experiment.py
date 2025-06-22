import numpy as np
import matplotlib.pyplot as plt
import os
from umbms import get_proj_path
from umbms.beamform.iczt import iczt
from collections import defaultdict, OrderedDict


__SAVE_DIR = os.path.expanduser("~")
__DATA_DIR = os.path.join(get_proj_path(), "data/attenuation_experiment/dgbe/")
__MY_DPI = 120


def transform_to_td(data):
    ts_data = iczt(
        data, ini_t=0.5e-9, fin_t=5.5e-9, ini_f=2e9, fin_f=9e9, n_time_pts=1000
    )
    return ts_data


def process_data():
    groups = defaultdict(list)  # {volume: [file, file, file]}
    control = defaultdict(list)  # {volume: [file, file, file]}

    for fname in os.listdir(__DATA_DIR):
        name = fname.split("ml")[0]
        if name.isdigit():
            name = int(name)
            groups[name].append(os.path.join(__DATA_DIR, fname))
        else:
            name = name.split(".")[0][:-1]
            control[name].append(os.path.join(__DATA_DIR, fname))

    averaged_vol = {}  # {volume: np.ndarray}
    no_liquid_data = {}

    stacks_open = [
        np.genfromtxt(f, delimiter=",", skip_header=3)[:, 1:]
        for f in control["open"]
    ]
    open_reference = np.mean(
        tuple(data[:, 0] + 1j * data[:, 1] for data in stacks_open), axis=0
    )

    stacks = [
        np.genfromtxt(f, delimiter=",", skip_header=3)[:, 1:]
        for f in control["no_liquid"]
    ]
    # Process the real and imaginary parts into one number
    stacks_fd = tuple(data[:, 0] + 1j * data[:, 1] for data in stacks)
    no_liquid_data["no_liquid"] = np.abs(
        transform_to_td(np.mean(stacks_fd, axis=0) - open_reference)
    )

    for vol, files in groups.items():
        stacks = [
            np.genfromtxt(f, delimiter=",", skip_header=3)[:, 1:] for f in files
        ]
        # Process the real and imaginary parts into one number
        stacks_fd = tuple(data[:, 0] + 1j * data[:, 1] for data in stacks)
        averaged_vol[vol] = np.abs(
            transform_to_td(np.mean(stacks_fd, axis=0) - open_reference)
        )

    return OrderedDict(sorted(averaged_vol.items())), no_liquid_data


def make_plot():
    averaged_vol, no_liquid_data = process_data()
    time = np.linspace(0.5, 5.5, 1000)

    cmap = plt.get_cmap("plasma")
    colors = [cmap(i) for i in np.linspace(0, 1, len(averaged_vol) + 1)]

    fig, ax = plt.subplots(
        figsize=(1200 / __MY_DPI, 800 / __MY_DPI), dpi=__MY_DPI
    )

    ax.plot(
        time,
        no_liquid_data["no_liquid"],
        label="No liquid",
        color=colors[0],
        linewidth=1,
    )

    col_idx = 1
    for vol, data in averaged_vol.items():
        ax.plot(
            time,
            data,
            label=f"{str(vol)} ml",
            color=colors[col_idx],
            linewidth=0.6,
        )
        col_idx += 1

    ax.set_title("Intensity of the metal disc with increasing volume")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.legend(fontsize=7)
    ax.grid()
    plt.tight_layout()
    fig.savefig(os.path.join(__SAVE_DIR, "Desktop/conductivity_plot.png"))
    # plt.show()


def main():
    make_plot()


if __name__ == "__main__":
    main()
