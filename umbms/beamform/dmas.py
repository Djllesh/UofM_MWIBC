import numpy as np


def fd_dmas(fd_data, pix_ts, freqs):
    """Compute frequency-domain DMAS reconstruction

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    pix_ts : array_like
        One-way response times for each pixel in the domain, for each
        antenna position
    freqs : array_like
        The frequencies used in the scan

    Returns
    -------
    img : array_like, KxK
        Reconstructed image, K pixels by K pixels
    """
    n_ant_pos = np.size(fd_data, axis=1)

    # Init array for storing the individual back-projections, from
    # each antenna position
    back_projections = np.empty(
        [n_ant_pos, np.size(pix_ts, axis=1), np.size(pix_ts, axis=2)],
        dtype=complex,
    )

    # For each antenna position
    for aa in range(n_ant_pos):
        # Get the value to back-project
        back_proj_val = fd_data[:, aa, None, None] * np.exp(
            -2j * np.pi * freqs[:, None, None] * (-2 * pix_ts[aa, :, :])
        )
        # Sum over all frequencies
        back_proj_val = np.sum(back_proj_val, axis=0)

        # Store the back projection
        back_projections[aa, :, :] = back_proj_val

    # Init image to return
    img = np.empty(
        [np.size(pix_ts, axis=1), np.size(pix_ts, axis=1)], dtype=complex
    )

    # Loop over each antenna position
    for a_pos_frw in range(n_ant_pos):
        # For each other antenna position
        for a_pos_mult in range(a_pos_frw + 1, n_ant_pos):
            img += (
                back_projections[a_pos_frw, :, :]
                * back_projections[a_pos_mult, :, :]
            )

    return img
