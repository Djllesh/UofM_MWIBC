"""
Illia Prykhodko
University of Manitoba
April 10, 2025
"""

from functools import partial
import matlab.engine
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from backprojection_simulation import (
    calculate_avg_speed,
    create_antenna_array,
    find_xy_ant_bound_circle,
)

MY_DPI = 120
plt.rcParams["animation.ffmpeg_path"] = (
    "C:/Program Files ("
    "x86)/ffmpeg-2023-10-26-git-2b300eb533-full_build/"
    "ffmpeg-2023-10-26-git-2b300eb533-full_build/bin/ffmpeg.exe"
)


def main():
    eng = matlab.engine.start_matlab()

    n_ant_pos = 300
    ant_rad = 0.21
    phantom_rad = 0.055
    roi = phantom_rad + 0.01
    x_ants, y_ants = create_antenna_array(ant_rad=ant_rad, n_ants=n_ant_pos)

    int_f_xs, int_f_ys, int_b_xs, int_b_ys = find_xy_ant_bound_circle(
        ant_xs=x_ants, ant_ys=y_ants, n_ant_pos=n_ant_pos, adi_rad=phantom_rad
    )

    ant_speed = calculate_avg_speed(
        n_ant_pos=n_ant_pos,
        ant_xs=x_ants,
        ant_ys=y_ants,
        int_f_xs=int_f_xs,
        int_f_ys=int_f_ys,
        int_b_xs=int_b_xs,
        int_b_ys=int_b_ys,
        air_speed=3e8,
        breast_speed=1.7e8,
    )

    ant_frame_speed = np.zeros_like(ant_speed)

    pixel_length = roi * 2 / 150
    D = ant_rad / pixel_length
    fan_rotation_increment = 360 / len(x_ants)
    fan_sensor_spacing = fan_rotation_increment / 2

    fig, ax = plt.subplots(1, 1, figsize=(600 / MY_DPI, 700 / MY_DPI))

    init_image = eng.ifanbeam(
        np.zeros_like(ant_speed).T,
        D,
        "FanRotationIncrement",
        fan_rotation_increment,
        "FanSensorSpacing",
        fan_sensor_spacing,
        "OutputSize",
        150.0,
        "Filter",
        "Hamming",
    )

    generated_images = []

    for row in range(0, n_ant_pos):
        # On every frame update the ant_speed array to contain one
        # more column
        ant_frame_speed[: row + 1, :] = ant_speed[: row + 1, :]

        # print(np.count_nonzero(ant_frame_speed))

        _image = eng.ifanbeam(
            ant_frame_speed.T,
            D,
            "FanRotationIncrement",
            fan_rotation_increment,
            "FanSensorSpacing",
            fan_sensor_spacing,
            "OutputSize",
            150.0,
            "Filter",
            "Hamming",
        )

        _image_numpy = np.array(_image._data).reshape((150, 150))

        generated_images.append(_image_numpy)  # Store the NumPy array

    projected_image = ax.imshow(init_image, animated=True)
    # fig.colorbar(projected_image, ax=ax)

    def project(frame):
        projected_image.set_data(generated_images[frame])

        vmin, vmax = (
            np.min(generated_images[frame]),
            np.max(generated_images[frame]),
        )

        projected_image.set_clim(vmin, vmax)

        return (projected_image,)

    ani = animation.FuncAnimation(
        fig=fig,
        func=project,
        frames=n_ant_pos,
        interval=10 * 1000 / n_ant_pos,
        blit=True,
    )

    plt.tight_layout()
    # plt.show()

    FFwriter = animation.FFMpegWriter(
        fps=n_ant_pos / 10, extra_args=["-vcodec", "libx264"]
    )

    ani.save(
        "C:/Users/prikh/Desktop/backprojection_ani.mp4",
        writer=FFwriter,
        dpi=MY_DPI,
    )

    eng.exit()


if __name__ == "__main__":
    main()
