"""
Illia Prykhodko
University of Manitoba
July 6, 2023
"""

from scipy import optimize
import numpy as np


def shift_and_rotate(xs, ys, delta_x, delta_y, delta_phi):
    """Performs geometric shift and rotation on the given set of
    coordinates

    Parameters
    -----------
    xs : array_like
        x-coordinates of the boundary
    ys : array_like
        y-coordinates of the boundary
    delta_x : float
        Spatial shift in the direction of x-axis (m)
    delta_y : float
        Spatial shift in the direction of y-axis (m)
    delta_phi : float
        Spatial rotation counter-clockwise

    Returns
    -----------
    xs_shift_rot, ys_shift_rot : array_like
        New coordinates
    """

    xs_shift_rot = (xs + delta_x) * np.cos(delta_phi) - (ys + delta_y) * np.sin(
        delta_phi
    )

    ys_shift_rot = (xs + delta_x) * np.sin(delta_phi) + (ys + delta_y) * np.cos(
        delta_phi
    )

    return xs_shift_rot, ys_shift_rot


def MSE(shift, cs_left, cs_right):
    """Calculates the mean square error for two boundaries

    Assumption: left breast is fixed, right breast is moving

    parameters
    ----------
    shift : array_like 1x2
        x- and y-shift values as a 1-dimensional array
    cs_left : ppoly
        cubic spline of the boundary of the left breast
    cs_right : ppoly
        cubic spline of the boundary of the right breast

    returns
    ---------
    mse_value : float
        mean square error value
    """

    # Obtain polar coordinates
    phi = np.deg2rad(np.arange(0, 360, 5.0))
    rho_left = cs_left(phi)
    rho_right = cs_right(phi)
    # Size for mse normalization
    n = np.size(phi)

    # Transform into cartesian
    xs_left = rho_left * np.cos(phi)
    ys_left = rho_left * np.sin(phi)

    xs_right = rho_right * np.cos(phi)
    xs_right_shifted = xs_right + shift[0]

    ys_right = rho_right * np.sin(phi)
    ys_right_shifted = ys_right + shift[1]

    mse_x = (1 / n) * (np.sum((xs_right_shifted - xs_left) ** 2))
    mse_y = (1 / n) * (np.sum((ys_right_shifted - ys_left) ** 2))
    # mse_x = np.abs(np.sum((xs_right_shifted - xs_left)))
    # mse_y = np.abs(np.sum((ys_right_shifted - ys_left)))

    mse_value = mse_x + mse_y
    return mse_value


def MSE_shift_rot(shift, cs_left, cs_right):
    """Calculates the mean square error for two boundaries

    Assumption: left breast is fixed, right breast is moving,
    rotation permitted

    Parameters
    ----------
    shift : array_like 1x2
        x- and y-shift values as a 1-dimensional array
    cs_left : PPoly
        Cubic spline of the boundary of the left breast
    cs_right : PPoly
        Cubic spline of the boundary of the right breast

    Returns
    ---------
    mse_value : float
        Mean square error value
    """

    # Obtain polar coordinates
    phi = np.deg2rad(np.arange(0, 360, 5.0))
    rho_left = cs_left(phi)
    rho_right = cs_right(phi)
    # Size for mse normalization
    n = np.size(phi)

    # Transform into cartesian
    xs_left = rho_left * np.cos(phi)
    ys_left = rho_left * np.sin(phi)

    xs_right = rho_right * np.cos(phi)
    ys_right = rho_right * np.sin(phi)

    xs_right_shift_rot, ys_right_shift_rot = shift_and_rotate(
        xs=xs_right,
        ys=ys_right,
        delta_x=shift[0],
        delta_y=shift[1],
        delta_phi=shift[2],
    )

    mse_x = (1 / n) * (np.sum((xs_right_shift_rot - xs_left) ** 2))
    mse_y = (1 / n) * (np.sum((ys_right_shift_rot - ys_left) ** 2))
    # mse_x = np.abs(np.sum((xs_right_shifted - xs_left)))
    # mse_y = np.abs(np.sum((ys_right_shifted - ys_left)))

    mse_value = mse_x + mse_y
    return mse_value


def minimize_differential_shift(cs_left, cs_right):
    """Minimizes the MSE, outputs the needed shift

    Parameters
    ----------
    cs_left : PPoly
        Cubic spline of the boundary of the left breast
    cs_right : PPoly
        Cubic spline of the boundary of the right breast

    Returns
    ---------
    shift : array_like
        x- and y-shift of the right breast to align with left
    """
    x0 = np.array([0.0, 0.0])
    res = optimize.minimize(
        fun=MSE,
        x0=x0,
        args=(cs_left, cs_right),
        method="Nelder-Mead",
        tol=1e-11,
    )

    return res.x


def minimize_differential_shift_rot(cs_left, cs_right):
    """Minimizes the MSE, outputs the needed shift

    Parameters
    ----------
    cs_left : PPoly
        Cubic spline of the boundary of the left breast
    cs_right : PPoly
        Cubic spline of the boundary of the right breast

    Returns
    ---------
    shift : array_like
        x- and y-shift of the right breast to align with left
    """
    x0 = np.array([0.0, 0.0, 0.0])
    res = optimize.minimize(
        fun=MSE_shift_rot,
        x0=x0,
        args=(cs_left, cs_right),
        method="Nelder-Mead",
        tol=1e-11,
    )

    return res.x
