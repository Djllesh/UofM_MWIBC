"""
Illia Prykhodko
University of Manitoba
March 17, 2025
"""

import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import os


def create_domain(roi=0.2, phantom_rad=0.055):
    """Create the domain with a uniform circular phantom in the middle"""
    x = np.linspace(-roi, roi, 300)
    y = np.flip(x)
    xs, ys = np.meshgrid(x, y, indexing="xy")
    phantom_mask = xs**2 + ys**2 <= phantom_rad
    return xs, ys, phantom_mask


def create_antenna_array(ant_rad=0.2, n_ants=24):
    """Create the array of antennas as a set of xy-coordinates
    Indexing: 0 - first next clockwise

    """
    angs = np.flip(np.deg2rad(np.linspace(0, 360, n_ants, endpoint=False)))
    x_ants = ant_rad * np.cos(angs)
    y_ants = ant_rad * np.sin(angs)
    return x_ants, y_ants


def find_xy_ant_bound_circle(
    ant_xs, ant_ys, n_ant_pos, adi_rad, *, ox=0.0, oy=0.0
):
    """Finds breast boundary intersection coordinates
    with propagation trajectory from antenna position
    to corresponding pixel

    Parameters
    ----------
    ant_xs : array_like Mx1
        Antenna x-coordinates
    ant_ys : array_like Mx1
        Antenna y-coordinates
    n_ant_pos : int
        Number of antenna positions
    adi_rad : float
        Approximate radius of a phantom
    ox : float
        x_coord of the centre of the circle
    oy : float
        y_coord of the centre of the circle

    Returns
    ----------
    int_f_xs : array-like NxN
        x-coordinates of each front intersection
    int_f_ys : array-like NxN
        y-coordinates of each front intersection
    int_b_xs : array-like NxN
        x-coordinates of each back intersection
    int_b_ys : array-like NxN
        y-coordinates of each back intersection
    """

    # initializing arrays for storing intersection coordinates
    # front intersection - closer to antenna
    int_f_xs = np.empty([len(ant_xs), len(ant_xs) - 1], dtype=float)
    int_f_ys = np.empty_like(int_f_xs)

    # back intersection - farther from antenna
    int_b_xs = np.empty_like(int_f_xs)
    int_b_ys = np.empty_like(int_f_xs)

    for a_pos in range(n_ant_pos):
        # singular antenna position
        ant_pos_x = ant_xs[a_pos]
        ant_pos_y = ant_ys[a_pos]

        recentered_ant_x = np.roll(ant_xs, -a_pos)[1:]
        recentered_ant_y = np.roll(ant_ys, -a_pos)[1:]

        # for each other antenna
        for px_x, px_y, idx in zip(
            recentered_ant_x, recentered_ant_y, range(len(recentered_ant_x))
        ):
            # calculating the roots
            if np.isclose(px_x, ant_pos_x, atol=1e-8):
                x_roots = [px_x, px_x]
                y_roots = [
                    oy + np.sqrt(adi_rad**2 - (px_x - ox) ** 2),
                    oy - np.sqrt(adi_rad**2 - (px_x - ox) ** 2),
                ]
            elif np.isclose(px_y, ant_pos_y, atol=1e-8):
                y_roots = [px_y, px_y]
                x_roots = [
                    ox + np.sqrt(adi_rad**2 - (px_y - oy) ** 2),
                    ox - np.sqrt(adi_rad**2 - (px_y - oy) ** 2),
                ]
            else:
                # calculating coefficients of polynomial
                k = (ant_pos_y - px_y) / (ant_pos_x - px_x)
                a = k**2 + 1
                b = 2 * (k * px_y - k**2 * px_x - ox - k * oy)
                c = (
                    px_x**2 * k**2
                    - 2 * k * px_x * px_y
                    + px_y**2
                    - adi_rad**2
                    + ox**2
                    + 2 * k * px_x * oy
                    - 2 * px_y * oy
                    + oy**2
                )

                x_roots = np.roots([a, b, c])
                y_roots = k * (x_roots - px_x) + px_y

            # flag to determine whether there are two real roots
            are_roots = (
                ~np.iscomplex(x_roots)[0]
                & ~np.iscomplex(x_roots)[1]
                & np.isfinite(x_roots).all()
                & np.isfinite(y_roots).all()
            )

            if not are_roots:  # if no roots
                # pixel coords are stored at both front and back
                int_f_xs[a_pos, idx], int_f_ys[a_pos, idx] = px_x, px_y
                int_b_xs[a_pos, idx], int_b_ys[a_pos, idx] = px_x, px_y

            else:  # if two real roots
                # distance from centre to pixel
                distances_root_ant = np.sqrt(
                    (x_roots - ant_pos_x) ** 2 + (y_roots - ant_pos_y) ** 2
                )
                # initializing the list of tuples for easier sorting
                dtype = [("x", float), ("y", float), ("distance", float)]
                values = [
                    (x_roots[0], y_roots[0], distances_root_ant[0]),
                    (x_roots[1], y_roots[1], distances_root_ant[1]),
                ]
                roots = np.array(values, dtype=dtype)

                # sort in ascending order wrt x_values
                roots = np.sort(roots, order="distance")

                # store lower distance as front intersection
                int_f_xs[a_pos, idx], int_f_ys[a_pos, idx], _ = roots[0]
                # store higher distance as back intersection
                int_b_xs[a_pos, idx], int_b_ys[a_pos, idx], _ = roots[1]

    return int_f_xs, int_f_ys, int_b_xs, int_b_ys


def calculate_avg_speed(
    n_ant_pos,
    ant_xs,
    ant_ys,
    int_f_xs,
    int_f_ys,
    int_b_xs,
    int_b_ys,
    air_speed,
    breast_speed,
):
    """Calculates the average propagation speed along the path between
    two antennas

    """

    # Init array for storing speeds
    ant_speed = np.zeros([n_ant_pos, n_ant_pos - 1])

    for a_pos in range(n_ant_pos):
        # Calculate total distance
        ant_x, ant_y = ant_xs[a_pos], ant_ys[a_pos]
        recentered_ant_xs = np.roll(ant_xs, -a_pos)[1:]
        recentered_ant_ys = np.roll(ant_ys, -a_pos)[1:]
        total_distances = np.sqrt(
            (recentered_ant_xs - ant_x) ** 2 + (recentered_ant_ys - ant_y) ** 2
        )

        # Calculate total time

        fin_ant_to_back_xs = recentered_ant_xs - int_b_xs[a_pos]
        fin_ant_to_back_ys = recentered_ant_ys - int_b_ys[a_pos]

        back_to_front_ys = int_b_ys[a_pos] - int_f_ys[a_pos]
        back_to_front_xs = int_b_xs[a_pos] - int_f_xs[a_pos]

        front_to_ant_xs = int_f_xs[a_pos] - ant_x
        front_to_ant_ys = int_f_ys[a_pos] - ant_y

        air_time_back = (
            np.sqrt(fin_ant_to_back_xs**2 + fin_ant_to_back_ys**2) / air_speed
        )
        phantom_time = (
            np.sqrt(back_to_front_xs**2 + back_to_front_ys**2) / breast_speed
        )
        air_time_front = (
            np.sqrt(front_to_ant_xs**2 + front_to_ant_ys**2) / air_speed
        )

        total_time = air_time_front + phantom_time + air_time_back

        ant_speed[a_pos, :] = total_distances / total_time

    return ant_speed


if __name__ == "__main__":
    n_ant_pos = 72
    ant_rad = 0.21
    adi_rad = 0.055
    roi = adi_rad + 0.01
    xs, ys, phantom_mask = create_domain(roi=ant_rad, phantom_rad=adi_rad)
    x_ants, y_ants = create_antenna_array(ant_rad=ant_rad, n_ants=n_ant_pos)

    int_f_xs, int_f_ys, int_b_xs, int_b_ys = find_xy_ant_bound_circle(
        ant_xs=x_ants, ant_ys=y_ants, n_ant_pos=n_ant_pos, adi_rad=adi_rad
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

    # ant_speed[2:, :] = 0

    sino = plt.imshow(ant_speed.T, extent=(0, 360, 0, 24), aspect="auto")
    plt.xlabel(r"Detector angle ($^\circ$)")
    plt.ylabel("Receiving antenna")
    plt.colorbar(sino, label="Propagation speed (m/s)")

    # plt.plot(ant_speed[0, :])
    # plt.xlabel('Antenna position')
    # plt.ylabel('Propagation speed (m/s)')

    eng = matlab.engine.start_matlab()

    pixel_length = roi * 2 / 150
    D = ant_rad / pixel_length
    fan_rotation_increment = 360 / len(x_ants)
    fan_sensor_spacing = fan_rotation_increment / 2

    image = eng.ifanbeam(
        ant_speed.T,
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

    # image = eng.iradon(ant_speed.T, fan_sensor_spacing * 2)

    a = 3 / (np.max(image) - np.min(image))
    b = -a * np.min(image)
    image = a * image + b

    # img = plt.imshow(image)
    # plt.colorbar(img)

    # plt.plot(
    #     np.linspace(-1, 1, len(image[0])), image[np.size(image, axis=0) // 2, :]
    # )
    # plt.xlabel("Horizontal extent")
    # plt.ylabel(r"Propagation speed $10^8$ (m/s)")

    # circle_angles = np.linspace(0, 2*np.pi, 100)
    #
    # circle_xs = phantom_rad * np.cos(circle_angles)
    # circle_ys = phantom_rad * np.sin(circle_angles)
    # plt.plot(circle_xs, circle_ys)
    # plt.plot(int_f_xs[8, :], int_f_ys[8, :], 'ro')
    # plt.plot(int_b_xs[8, :], int_b_ys[8, :], 'ko')
    # to_ant_xs = np.roll(x_ants, -8)[1:]
    # to_ant_ys = np.roll(y_ants, -8)[1:]
    # for ant_pos in range(len(x_ants)-1):
    #     point_x, point_y = to_ant_xs[ant_pos], to_ant_ys[ant_pos]
    #     plt.plot([x_ants[8], point_x], [y_ants[8], point_y], 'b-')
    # plt.gca().set_aspect('equal')
    plt.tight_layout()
    # plt.show()

    # plt.savefig(
    #     os.path.join(
    #         os.path.expanduser("~"),
    #         "Desktop/slice_%d_ants.png" % n_ant_pos,
    #     )
    # )

    plt.savefig(
        os.path.join(
            os.path.expanduser("~"), "Desktop/backprojection_sinogram.png"
        )
    )

    eng.exit()
