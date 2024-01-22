"""
Illia Prykhodko

Univerity of Manitoba,
December 1st, 2023
"""


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os

from umbms.beamform.utility import get_xy_arrs, get_ant_scan_xys
from umbms.hardware.antenna import apply_ant_pix_delay, to_phase_center
from umbms.beamform.time_delay import get_pix_ts, get_pix_ts_old
from umbms.beamform.propspeed import estimate_speed, get_breast_speed_freq

M_SIZE = 150

if __name__ == "__main__":
    a = [[0.5, 0.3], [0.5, 0.3]]
    b = [[0.7, 0.1], [0.8, 0.05]]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Initial image plot with 'a'
    img = ax.imshow(a, vmin=0, vmax=1, cmap='viridis', animated=True)


    # Update function for animation
    def update(frame):
        if frame == 0:
            img.set_data(a)
        elif frame == 1:
            img.set_data(b)
        return [img]


    # Create animation
    animation = FuncAnimation(fig, update, frames=2, interval=1000, blit=True)

    # Show the plot
    plt.show()

    # TODO:
    # 1) Calculate two arrays of time-delays
    # 2) For each antenna define the time-delay dependence on distance
    #   2.1) Find the indices of the pixels that correspond to the ray
    #   2.2) Convert each pixel to distance to the antenna
    #   2.3) On the plot: left - dependence, right - schematic of the
    #        ray propagating through the phantom
    # 3) Compare different rays



    pass
