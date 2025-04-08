"""
Tyson Reimer
University of Manitoba
June 28th, 2019
"""

import os
import numpy as np
from stl import mesh

from umbms import get_proj_path

###############################################################################

__STL_FILE_PATH = os.path.join(get_proj_path(), "data/phantoms-stl/")

###############################################################################


def get_stl_vertices(shell_id):
    """Get the x,y,z coords of each vertex for each triangle in the stl
    file

    Returns the coordinates of the vertices that define the triangles of
    the .stl file mesh.

    Parameters
    ----------
    shell_id : str
        The ID for the adipose or fibroglandular shell to be loaded;
        ex: "A2"

    Returns
    -------
    shell_vertices : array_like
        The x,y,z coordinates, in millimeters, of the vertices that
        define the triangles that define the .stl mesh
    """

    # Import stl file
    shell_mesh = mesh.Mesh.from_file(
        os.path.join(__STL_FILE_PATH, "%s_inner_dense.stl" % shell_id)
    )

    shell_vertices = shell_mesh.vectors  # Get the x,y,z coordinates, in mm

    # Store the x,y,z positions in dummy vectors to convert to the
    # convenient x,y,z coordinate axes
    shell_vertices_xs = shell_vertices[:, :, 0]
    shell_vertices_ys = shell_vertices[:, :, 1]
    shell_vertices_zs = shell_vertices[:, :, 2]

    # Init arr to return
    true_shell_vertices = np.ones_like(shell_vertices)

    # Convert to new frame of reference, using standard axes definitions
    # with the BIRR system
    true_shell_vertices[:, :, 0] = shell_vertices_zs
    true_shell_vertices[:, :, 1] = shell_vertices_xs
    true_shell_vertices[:, :, 2] = shell_vertices_ys

    return true_shell_vertices


def get_shell_mesh(shell_id):
    """Load a Mesh object of the stl file for this shell_id

    Returns the coordinates of the vertices that define the triangles of
    the .stl file mesh.

    Parameters
    ----------
    shell_id : str
        The ID for the adipose or fibroglandular shell to be loaded;
        ex: "A2"

    Returns
    -------
    shell_mesh : mesh.Mesh() object
        The imported stl file as a Mesh object
    """

    # Load stl file
    shell_mesh = mesh.Mesh.from_file(
        os.path.join(__STL_FILE_PATH, "%s_inner_dense.stl" % shell_id)
    )

    return shell_mesh


def get_phantom_vertices(adipose_id, fibro_id):
    """Get the x,y,z coords of the vertices of each shell in the phantom
    stl

    Returns the coordinates of the vertices that define the triangles of
    the meshes that make the adipose and fibro shells (combines the
    shells into one stl object)

    Parameters
    ----------
    adipose_id : str
        The ID for the adipose shell; ex: "A1"
    fibro_id : str
        The ID for the fibroglandular shell; ex: "F4"

    Returns
    -------
    true_phantom_vertices : array_like
        The x,y,z coordinates, in millimeters, of the vertices that
        define the triangles that define the stl meshes
    """

    adi_mesh = get_shell_mesh(adipose_id)  # Get adipose mesh object
    fib_mesh = get_shell_mesh(fibro_id)  # Get fibroglandular mesh object

    # Combine the meshes
    phantom_mesh = mesh.Mesh(np.concatenate([adi_mesh.data, fib_mesh.data]))

    # Get the x,y,z positions of each vertex of each triangle in the mesh
    phantom_vertices = phantom_mesh.vectors

    # Store the x,y,z positions in dummy vectors to convert to the
    # convenient x,y,z coordinate axes
    phantom_vertices_xs = phantom_vertices[:, :, 0]
    phantom_vertices_ys = phantom_vertices[:, :, 1]
    phantom_vertices_zs = phantom_vertices[:, :, 2]

    # Init arr to return
    true_phantom_vertices = np.ones_like(phantom_vertices)

    # Convert to new frame of reference, using standard axes definitions
    # with the BIRR system
    true_phantom_vertices[:, :, 0] = phantom_vertices_zs
    true_phantom_vertices[:, :, 1] = phantom_vertices_xs
    true_phantom_vertices[:, :, 2] = phantom_vertices_ys

    return true_phantom_vertices


def get_xyz_min_mid_and_max(shell_vertices):
    """Returns the minimum, middle, and maximum x,y,z positions of a shell.

    Parameters
    ----------
    shell_vertices : array_like
        The vertices that define the triangles that define the stl mesh

    Returns
    -------
    x_positions : tuple
        Tuple containing (x_min, x_mid, x_max)
    y_positions : tuple
        Tuple containing (y_min, y_mid, y_max)
    z_positions : tuple
        Tuple containing (z_min, z_mid, z_max)
    """

    # By default, the coordinates are x,y,z; but to convert to the
    # coordinates used in the bed system, change:
    # x -> y, y -> z, z -> x

    # Find the minimum positions
    x_min, y_min, z_min = (
        np.min(shell_vertices[:, :, 0]),
        np.min(shell_vertices[:, :, 1]),
        np.min(shell_vertices[:, :, 2]),
    )

    # Find the maximum positions
    x_max, y_max, z_max = (
        np.max(shell_vertices[:, :, 0]),
        np.max(shell_vertices[:, :, 1]),
        np.max(shell_vertices[:, :, 2]),
    )

    # Find the middle-positions
    x_mid, y_mid, z_mid = (
        0.5 * (x_max - x_min) + x_min,
        0.5 * (y_max - y_min) + y_min,
        0.5 * (z_max - z_min) + z_min,
    )

    # Store the min/mid/max as tuples for ease-of-use
    x_positions = (x_min, x_mid, x_max)
    y_positions = (y_min, y_mid, y_max)
    z_positions = (z_min, z_mid, z_max)

    return x_positions, y_positions, z_positions


def get_shell_xy_for_z(shell_id, z_plane, slice_thickness=1.0):
    """Get the x,y coords of the stl file for the z_plane

    Returns the x,y coordinates of the shell for a coronal slice of
    thickness slice_thickness

    Parameters
    ----------
    shell_id : str
        The ID of the phantom shell, ex: "A3"
    z_plane : float
        The distance from the *top* of the get_phantom_vertices
        (where the base is) of the coronal slice, measured in
        millimeters
    slice_thickness : float
        The thickness of the coronal slice, in millimeters

    Returns
    -------
    x_points : array_like
        The x-positions of the phantom shell, in millimeters, with
        respect to the center of the base
    y_points : array_like
        The y-positions of the phantom shell, in millimeters, with
        respect to the center of the base
    """

    # Get the vertices for each triangle
    shell_vertices = get_stl_vertices(shell_id)

    # Find the min/mid/max of the x,y,z -positions for the shell
    xs, ys, zs = get_xyz_min_mid_and_max(shell_vertices)

    # Scale the z-positions of each vertex that defines the mesh so
    # that z=0 is at the base, and so that z > 0 at the nipple
    shell_vertices[:, :, 2] -= zs[2]  # Subtract the maximum z-value
    shell_vertices[:, :, 2] = np.abs(shell_vertices[:, :, 2])

    x_points, y_points = [], []  # Init lists to return

    # For every triangle
    for triangle_idx in range(np.size(shell_vertices, axis=0)):
        # Get the vertices for this triangle
        triangle_vertices = shell_vertices[triangle_idx, :, :]

        for vertex_idx in range(3):  # For each vertex
            # If the vertex is within half the slice_thickness from
            # the desired z_plane
            if (
                z_plane - slice_thickness / 2
                < triangle_vertices[vertex_idx, 2]
                < z_plane + slice_thickness
            ):
                # Store the x/y coordinates of this vertex
                x_points.append(triangle_vertices[vertex_idx, 0])
                y_points.append(triangle_vertices[vertex_idx, 1])

    # Convert to np arrays
    x_points, y_points = np.array(x_points), np.array(y_points)

    x_points -= xs[1]  # Scale so that x-points are centered at x=0
    y_points -= ys[1]  # Scale so that y-points are centered at y=0

    return x_points, y_points


def get_phantom_xy_for_z(adi_id, fib_id, z_plane, slice_thickness=1.0):
    """Get the x,y coords of the phantom stl file for a coronal slice

    Returns the x,y coordinates of the phantom for a coronal slice of
    thickness slice_thickness

    Parameters
    ----------
    adi_id : str
        The ID of the adipose shell in the phantom, ex: "A3"
    fib_id : str
        The ID of the fibroglandular shell in the phantom, ex: "F2"
    z_plane : float
        The distance from the *top* of the get_phantom_vertices
        (where the base is) of the coronal slice, measured in
        millimeters
    slice_thickness : float
        The thickness of the coronal slice, in millimeters

    Returns
    -------
    x_pts : array_like
        The x-positions of the phantom shell, in millimeters, with
        respect to the center of the base
    y_pts : array_like
        The y-positions of the phantom shell, in millimeters, with
        respect to the center of the base
    """

    # Get the vertices for each triangle
    phantom_vertices = get_phantom_vertices(adi_id, fib_id)

    # Find the min/mid/max of the x,y,z -positions for the shell
    xs, ys, zs = get_xyz_min_mid_and_max(phantom_vertices)

    # Scale the z-positions of each vertex that defines the mesh so
    # that z=0 is at the base, and so that z > 0 at the  nipple
    phantom_vertices[:, :, 2] -= zs[2]
    phantom_vertices[:, :, 2] = np.abs(phantom_vertices[:, :, 2])

    x_pts, y_pts = [], []  # Init lists to return

    # For every triangle
    for triangle_idx in range(np.size(phantom_vertices, axis=0)):
        # Get the vertices for this triangle
        triangle_vertices = phantom_vertices[triangle_idx, :, :]

        # For each vertex, see if it's in the coronal slice
        for vertex_idx in range(3):
            # If the vertex is within half the slice_thickness from
            # the desired z_plane
            if (
                z_plane - slice_thickness / 2
                < triangle_vertices[vertex_idx, 2]
                < z_plane + slice_thickness / 2
            ):
                # Store the x/y coordinates of this vertex
                x_pts.append(triangle_vertices[vertex_idx, 0])
                y_pts.append(triangle_vertices[vertex_idx, 1])

    x_pts, y_pts = np.array(x_pts), np.array(y_pts)  # Convert to np arrays

    x_pts -= xs[1]  # Scale so that x-points are centered at x=0
    y_pts -= ys[1]  # Scale so that y-points are centered at y=0

    return x_pts, y_pts
