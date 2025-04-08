"""
Illia Prykhodko
University of Manitoba
February 17th, 2023
"""

import numpy as np


###############################################################################


def find_xy_ant_bound_circle(
    ant_xs, ant_ys, n_ant_pos, pix_xs, pix_ys, adi_rad, *, ox=0.0, oy=0.0
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
    pix_xs : array_like
        Positions of x-coordinates of each pixel
    pix_ys : array_like
        Positions of y-coordinates of each pixel
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
    int_f_xs = np.empty([len(ant_xs), len(pix_xs), len(pix_ys)], dtype=float)
    int_f_ys = np.empty_like(int_f_xs)

    # back intersection - farther from antenna
    int_b_xs = np.empty_like(int_f_xs)
    int_b_ys = np.empty_like(int_f_xs)

    for a_pos in range(n_ant_pos):
        # singular antenna position
        ant_pos_x = ant_xs[a_pos]
        ant_pos_y = ant_ys[a_pos]

        # for each pixel
        for px_x, x in zip(pix_xs, len(pix_xs)):
            for px_y, y in zip(pix_ys, len(pix_ys)):
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

                # calculating the roots
                x_roots = np.roots([a, b, c])
                y_roots = k * (x_roots - px_x) + px_y

                # flag to determine whether there are two real roots
                are_roots = np.logical_and(
                    ~np.iscomplex(x_roots)[0], ~np.iscomplex(x_roots)[1]
                )

                if not are_roots:  # if no roots
                    # pixel coords are stored at both front and back
                    int_f_xs[a_pos, y, x], int_f_ys[a_pos, y, x] = px_x, px_y
                    int_b_xs[a_pos, y, x], int_b_ys[a_pos, y, x] = px_x, px_y

                else:  # if two real roots
                    # distance from centre to pixel
                    d_centre_pix = np.sqrt((px_x - ox) ** 2 + (px_y - oy) ** 2)

                    # initializing the list of tuples for easier sorting
                    dtype = [("x", float), ("y", float)]
                    values = [
                        (x_roots[0], y_roots[0]),
                        (x_roots[1], y_roots[1]),
                    ]
                    roots = np.array(values, dtype=dtype)

                    # sort in ascending order wrt x_values
                    roots = np.sort(roots, order="x")

                    is_inside = d_centre_pix <= adi_rad
                    is_left = ant_pos_x < px_x

                    if is_inside:  # if pixel is inside the breast
                        # store one intersection point
                        # sort in ascending order wrt x_values
                        roots = np.sort(roots, order="x")

                        if is_left:  # if antenna is to the left of pixel
                            # store lower x_value as front intersection
                            int_f_xs[a_pos, y, x], int_f_ys[a_pos, y, x] = (
                                roots[0]
                            )
                            # store pixel coords as back intersection
                            int_b_xs[a_pos, y, x], int_b_ys[a_pos, y, x] = (
                                px_x,
                                px_y,
                            )

                        else:
                            # store higher x_value as front intersection
                            int_f_xs[a_pos, y, x], int_f_ys[a_pos, y, x] = (
                                roots[1]
                            )
                            # store pixel coords as back intersection
                            int_b_xs[a_pos, y, x], int_b_ys[a_pos, y, x] = (
                                px_x,
                                px_y,
                            )

                    else:  # if pixel is outside the breast
                        # calculate distance from antenna to pixel
                        d_pix_ant = np.sqrt(
                            (ant_pos_x - px_x) ** 2 + (ant_pos_y - px_y) ** 2
                        )
                        # distance from antenna to adjacent point on circle
                        d_ant_adj = np.sqrt(
                            (ant_pos_x - ox) ** 2
                            + (ant_pos_y - oy) ** 2
                            - adi_rad**2
                        )

                        # flag to determine whether the pixel is
                        # in front of the breast
                        is_front = d_ant_adj >= d_pix_ant

                        if is_front:  # if pixel is in front
                            # store the same way as for no roots
                            int_f_xs[a_pos, y, x], int_f_ys[a_pos, y, x] = (
                                px_x,
                                px_y,
                            )
                            int_b_xs[a_pos, y, x], int_b_ys[a_pos, y, x] = (
                                px_x,
                                px_y,
                            )

                        else:  # if pixel is past the breast
                            roots = np.sort(roots, order="x")
                            if is_left:  # if antenna is to the left
                                # store lower x_value as front intersection
                                int_f_xs[a_pos, y, x], int_f_ys[a_pos, y, x] = (
                                    roots[0]
                                )
                                # store higher x_value as back intersection
                                int_b_xs[a_pos, y, x], int_b_ys[a_pos, y, x] = (
                                    roots[1]
                                )

                            else:  # if antenna is to the right
                                # store higher x_value as front intersection
                                int_f_xs[a_pos, y, x], int_f_ys[a_pos, y, x] = (
                                    roots[1]
                                )
                                # store lower x_value as back intersection
                                int_b_xs[a_pos, y, x], int_b_ys[a_pos, y, x] = (
                                    roots[0]
                                )

    return int_f_xs, int_f_ys, int_b_xs, int_b_ys


def find_xy_ant_bound_ellipse(
    ant_xs, ant_ys, n_ant_pos, pix_xs, pix_ys, mid_breast_max, mid_breast_min
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
    pix_xs : array_like
        Positions of x-coordinates of each pixel
    pix_ys : array_like
        Positions of y-coordinates of each pixel
    mid_breast_max : float
        major semi-axis of a phantom (b)
    mid_breast_min : float
        minor semi-axis of a phantom (a)

    Returns
    ----------
    int_f_xs : array-like mxNxN
        x-coordinates of each front intersection
    int_f_ys : array-like mxNxN
        y-coordinates of each front intersection
    int_b_xs : array-like mxNxN
        x-coordinates of each back intersection
    int_b_ys : array-like mxNxN
        y-coordinates of each back intersection
    """

    # initializing arrays for storing intersection coordinates
    # front intersection - closer to antenna
    int_f_xs = np.empty([len(ant_xs), len(pix_xs), len(pix_xs)], dtype=float)
    int_f_ys = np.empty_like(int_f_xs)

    # back intersection - farther from antenna
    int_b_xs = np.empty_like(int_f_xs)
    int_b_ys = np.empty_like(int_f_xs)

    # initializing semi-axes of an ellipse
    a = mid_breast_min
    b = mid_breast_max

    # for each antenna position
    for a_pos in range(n_ant_pos):
        # singular antenna position
        ant_pos_x = ant_xs[a_pos]
        ant_pos_y = ant_ys[a_pos]

        # for each pixel
        for px_x, x in zip(pix_xs, range(len(pix_xs))):
            for px_y, y in zip(pix_ys, range(len(pix_ys))):
                if ant_pos_x == px_x:  # if antenna is straight up of down
                    ant_pos_x += 1e-9

                # calculating roots
                k = (ant_pos_y - px_y) / (ant_pos_x - px_x)
                x_roots = np.roots(
                    [
                        a**2 * k**2 + b**2,
                        a**2 * (2 * k * px_y - 2 * k**2 * px_x),
                        a**2
                        * (
                            k**2 * px_x**2
                            - 2 * k * px_x * px_y
                            + px_y**2
                            - b**2
                        ),
                    ]
                )
                y_roots = k * (x_roots - px_x) + px_y

                # flag to determine whether there are roots
                are_roots = ~np.iscomplex(x_roots)[0]

                if not are_roots:  # if no intersection
                    # store pixel coords as intersection
                    int_f_xs[a_pos, y, x], int_f_ys[a_pos, y, x] = px_x, px_y
                    int_b_xs[a_pos, y, x], int_b_ys[a_pos, y, x] = px_x, px_y

                else:
                    # initializing an array of tuples for easier sorting
                    dtype = [("x", float), ("y", float)]
                    temp = [(x_roots[0], y_roots[0]), (x_roots[1], y_roots[1])]
                    roots = np.array(temp, dtype=dtype)

                    # flag to determine whether the pixel is inside the phantom
                    is_inside = px_x**2 / a**2 + px_y**2 / b**2 <= 1
                    # flag to determine the position of the antenna wrt
                    # to pixel and xy coords
                    is_left = ant_pos_x < px_x

                    if is_inside:
                        # sort roots in ascending order wrt to x_values
                        roots = np.sort(roots, order="x")
                        if is_left:  # if antenna is to the left of the pixel
                            # lower x - front
                            int_f_xs[a_pos, y, x], int_f_ys[a_pos, y, x] = (
                                roots[0][0],
                                roots[0][1],
                            )
                            # pixel - back
                            int_b_xs[a_pos, y, x], int_b_ys[a_pos, y, x] = (
                                px_x,
                                px_y,
                            )
                        else:
                            # higher x - front
                            int_f_xs[a_pos, y, x], int_f_ys[a_pos, y, x] = (
                                roots[1][0],
                                roots[1][1],
                            )
                            # pixel - back
                            int_b_xs[a_pos, y, x], int_b_ys[a_pos, y, x] = (
                                px_x,
                                px_y,
                            )
                    else:
                        # flag to determine whether the pixel is
                        # in front of the antenna
                        is_front = (
                            (
                                b**2 * ant_pos_x * px_x
                                + a**2 * ant_pos_y * px_y
                                - a**2 * b**2
                            )
                            / (a**2 * ant_pos_y)
                            < 0
                            and ant_pos_y <= 0
                        ) or (
                            (
                                b**2 * ant_pos_x * px_x
                                + a**2 * ant_pos_y * px_y
                                - a**2 * b**2
                            )
                            / (a**2 * ant_pos_y)
                            > 0
                            and ant_pos_y > 0
                        )

                        if is_front:
                            # store pixel coords as intersection
                            int_f_xs[a_pos, y, x], int_f_ys[a_pos, y, x] = (
                                px_x,
                                px_y,
                            )
                            int_b_xs[a_pos, y, x], int_b_ys[a_pos, y, x] = (
                                px_x,
                                px_y,
                            )
                        else:
                            # sort roots in ascending order wrt to x_values
                            roots = np.sort(roots, order="x")
                            if is_left:
                                # lower x - front
                                int_f_xs[a_pos, y, x], int_f_ys[a_pos, y, x] = (
                                    roots[0][0],
                                    roots[0][1],
                                )
                                # higher x - back
                                int_b_xs[a_pos, y, x], int_b_ys[a_pos, y, x] = (
                                    roots[1][0],
                                    roots[1][1],
                                )
                            else:
                                # higher x - front
                                int_f_xs[a_pos, y, x], int_f_ys[a_pos, y, x] = (
                                    roots[1][0],
                                    roots[1][1],
                                )
                                # lower x - back
                                int_b_xs[a_pos, y, x], int_b_ys[a_pos, y, x] = (
                                    roots[0][0],
                                    roots[0][1],
                                )

    return int_f_xs, int_f_ys, int_b_xs, int_b_ys


def _parallel_find_bound_circle_pix(
    ant_xs, ant_ys, n_ant_pos, pix_xs, pix_ys, adi_rad, ox, oy, idx
):
    """Finds breast boundary intersection coordinates
    with propagation trajectory from antenna position
    to corresponding pixel (for parallel calculation)

    Parameters
    ----------
    ant_xs : array_like Mx1
        Antenna x-coordinates
    ant_ys : array_like Mx1
        Antenna y-coordinates
    n_ant_pos : int
        Number of antenna positions
    pix_xs : array_like
        Positions of x-coordinates of each pixel
    pix_ys : array_like
        Positions of y-coordinates of each pixel
    adi_rad : float
        Approximate radius of a phantom
    ox : float
        x_coord of the centre of the circle
    oy : float
        y_coord of the centre of the circle
    idx : int
        Index of the current parallel iteration

    Returns
    ----------
    int_f_xs : float
        x-coordinates of each front intersection
    int_f_ys : float
        y-coordinates of each front intersection
    int_b_xs : float
        x-coordinates of each back intersection
    int_b_ys : float
        y-coordinates of each back intersection
    """

    a_pos, y, x = np.unravel_index(
        idx, [n_ant_pos, np.size(pix_xs), np.size(pix_ys)]
    )
    px_x = pix_xs[x]
    px_y = pix_ys[y]

    # singular antenna position
    ant_pos_x = ant_xs[a_pos]
    ant_pos_y = ant_ys[a_pos]

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

    # calculating the roots
    x_roots = np.roots([a, b, c])
    y_roots = k * (x_roots - px_x) + px_y

    # flag to determine whether there are two real roots
    are_roots = np.logical_and(
        ~np.iscomplex(x_roots)[0], ~np.iscomplex(x_roots)[1]
    )

    if not are_roots:  # if no roots
        # pixel coords are stored at both front and back
        int_f_xs, int_f_ys = px_x, px_y
        int_b_xs, int_b_ys = px_x, px_y

    else:  # if two real roots
        # distance from centre to pixel
        d_centre_pix = np.sqrt((px_x - ox) ** 2 + (px_y - oy) ** 2)

        # initializing the list of tuples for easier sorting
        dtype = [("x", float), ("y", float)]
        values = [(x_roots[0], y_roots[0]), (x_roots[1], y_roots[1])]
        roots = np.array(values, dtype=dtype)

        # sort in ascending order wrt x_values
        roots = np.sort(roots, order="x")

        is_inside = d_centre_pix <= adi_rad
        is_left = ant_pos_x < px_x

        if is_inside:  # if pixel is inside the breast
            # store one intersection point
            # sort in ascending order wrt x_values
            roots = np.sort(roots, order="x")

            if is_left:  # if antenna is to the left of pixel
                # store lower x_value as front intersection
                int_f_xs, int_f_ys = roots[0]
                # store pixel coords as back intersection
                int_b_xs, int_b_ys = px_x, px_y

            else:
                # store higher x_value as front intersection
                int_f_xs, int_f_ys = roots[1]
                # store pixel coords as back intersection
                int_b_xs, int_b_ys = px_x, px_y

        else:  # if pixel is outside the breast
            # calculate distance from antenna to pixel
            d_pix_ant = np.sqrt(
                (ant_pos_x - px_x) ** 2 + (ant_pos_y - px_y) ** 2
            )
            # distance from antenna to adjacent point on circle
            d_ant_adj = np.sqrt(
                (ant_pos_x - ox) ** 2 + (ant_pos_y - oy) ** 2 - adi_rad**2
            )

            # flag to determine whether the pixel is in front of the breast
            is_front = d_ant_adj >= d_pix_ant

            if is_front:  # if pixel is in front
                # store the same way as for no roots
                int_f_xs, int_f_ys = px_x, px_y
                int_b_xs, int_b_ys = px_x, px_y

            else:  # if pixel is past the breast
                roots = np.sort(roots, order="x")
                if is_left:  # if antenna is to the left
                    # store lower x_value as front intersection
                    int_f_xs, int_f_ys = roots[0]
                    # store higher x_value as back intersection
                    int_b_xs, int_b_ys = roots[1]

                else:  # if antenna is to the right
                    # store higher x_value as front intersection
                    int_f_xs, int_f_ys = roots[1]
                    # store lower x_value as back intersection
                    int_b_xs, int_b_ys = roots[0]

    return np.array([int_f_xs, int_f_ys, int_b_xs, int_b_ys])
