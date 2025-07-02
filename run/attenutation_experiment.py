import numpy as np
import matplotlib.pyplot as plt
import os
from umbms import get_proj_path
from umbms.beamform.iczt import iczt
from collections import defaultdict, OrderedDict


__SAVE_DIR = os.path.expanduser("~")
__DATA_DIR = os.path.join(
    get_proj_path(), "data/attenuation_experiment/illia_attenuation/"
)
__MY_DPI = 150


def transform_to_td(data):
    ts_data = iczt(
        data, ini_t=0.5e-9, fin_t=5.5e-9, ini_f=2e9, fin_f=9e9, n_time_pts=1000
    )
    return ts_data


def average_groups(groups):
    averaged_vol = {}

    for vol, files in groups.items():
        stacks = [
            np.genfromtxt(f, delimiter=",", skip_header=3)[:, 1:] for f in files
        ]
        # Process the real and imaginary parts into one number
        stacks_fd = tuple(data[:, 0] + 1j * data[:, 1] for data in stacks)
        averaged_vol[vol] = np.abs(transform_to_td(np.mean(stacks_fd, axis=0)))

    return averaged_vol


def process_data():
    groups_target = defaultdict(list)  # {volume: [file, file, file]}
    groups_ref = defaultdict(list)  # {volume: [file, file, file]}
    calibrated_data = {}

    for fname in os.listdir(__DATA_DIR):
        name = fname.split("ml_")[0]
        type = fname.split("ml_")[1]
        name = int(name)

        if "target" in type:
            groups_target[name].append(os.path.join(__DATA_DIR, fname))
        elif "ref" in type:
            groups_ref[name].append(os.path.join(__DATA_DIR, fname))

    averaged_vol_ref = average_groups(groups_ref)
    averaged_vol_target = average_groups(groups_target)

    for vol, data in averaged_vol_ref.items():
        print(f"The volume is {vol}")
        calibrated_data_vol = data - averaged_vol_target[vol]
        calibrated_data[vol] = calibrated_data_vol

    return OrderedDict(sorted(calibrated_data.items()))


def make_plot():
    calibrated_data = process_data()
    time = np.linspace(0.5, 5.5, 1000)

    cmap = plt.get_cmap("plasma")
    colors = [cmap(i) for i in np.linspace(0, 1, len(calibrated_data) + 1)]

    fig, ax = plt.subplots(
        figsize=(1200 / __MY_DPI, 800 / __MY_DPI), dpi=__MY_DPI
    )

    col_idx = 1
    for vol, data in calibrated_data.items():
        if vol == 0:
            ax.plot(
                time,
                data,
                label="No liquid",
                color=colors[0],
                linewidth=1,
            )

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
    # fig.savefig(os.path.join(__SAVE_DIR, "Desktop/conductivity_plot.png"))
    plt.show()


def main():
    make_plot()


if __name__ == "__main__":
    main()
