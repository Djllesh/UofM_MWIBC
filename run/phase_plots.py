"""
Illia Prykhodko
University of Manitoba
October 15, 2024
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import numpy as np
from umbms import get_proj_path, verify_path, get_script_logger
from umbms.loadsave import load_pickle, save_pickle

__DATA_DIR = os.path.join(
    get_proj_path(), "data/umbmid/cyl_phantom/speed_paper/"
)

MY_DPI = 120
plt.rcParams["animation.ffmpeg_path"] = (
    "C:/Program Files ("
    "x86)/ffmpeg-2023-10-26-git-2b300eb533-full_build/"
    "ffmpeg-2023-10-26-git-2b300eb533-full_build/bin/ffmpeg.exe"
)
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
target_phase_unwrapped = np.unwrap(target_phase)

fig, ax1 = plt.subplots(figsize=(1000 / __MY_DPI, 800 / __MY_DPI))
ax1.set_xlabel("Frequency, (Hz)")
ax1.set_ylabel("Phase, rad", color="blue")
ax1.plot(freqs, target_phase, "b--", linewidth=1)
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax1.set_xlim(1.8e9, 9.2e9)
ax2.set_xlim(1.8e9, 9.2e9)
ax2.set_ylim(
    top=target_phase_unwrapped[0] + 3, bottom=target_phase_unwrapped[-1] - 23
)
ax2.set_ylabel("Unwrapped phase, rad", color="black")
unwrapped_phase_plot = ax2.plot(
    freqs, target_phase_unwrapped, "k-", linewidth=1.4
)

initial_phase = ax2.axhline(
    y=target_phase_unwrapped[0],
    xmin=(2 - 1.8) / (9.8 - 1.8),
    color="black",
    linestyle="-",
    linewidth=0.9,
    alpha=0.8,
)
(point,) = ax2.plot(freqs[0], target_phase_unwrapped[0], "kx")

# Create an empty annotation box without an arrow
annotation = ax2.annotate(
    f"Shift: - {0:.2f} rad",
    xy=(9.2e9, target_phase_unwrapped[0]),
    xytext=(8.5e9, target_phase_unwrapped[0] - 1),
    # Adjust the box position to be slightly left
    ha="center",
    va="bottom",
    fontsize=9,
    color="black",
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
)
annotation.set_visible(False)  # Start invisible


def shift_line(shift):
    # Update the plot and the line positions
    unwrapped_phase_plot[0].set_ydata(target_phase_unwrapped - shift)
    initial_phase.set_data(
        [(2 - 1.8) / (9.8 - 1.8), 1],
        [target_phase_unwrapped[0] - shift, target_phase_unwrapped[0] - shift],
    )

    point.set_ydata(target_phase_unwrapped[0] - shift)

    # Update the annotation box position, text, and visibility
    annotation.xy = (
        9.2e9,
        target_phase_unwrapped[0] - shift,
    )  # Point for the box
    annotation.set_position((8.5e9, target_phase_unwrapped[0] - 1 - shift))
    annotation.set_text(f"Shift: - {shift:.2f} rad")  # Update the shift text
    annotation.set_visible(True)  # Make sure the annotation is visible

    return unwrapped_phase_plot[0], initial_phase, point, annotation


ani = animation.FuncAnimation(
    fig=fig,
    func=shift_line,
    frames=np.linspace(0, 8 * np.pi, 800),
    interval=7,
    blit=True,
)

plt.tight_layout()
FFwriter = animation.FFMpegWriter(fps=30, extra_args=["-vcodec", "libx264"])
plt.show()
# ani.save('shift.mp4', writer=FFwriter, dpi=__MY_DPI)

