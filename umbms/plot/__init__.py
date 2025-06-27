"""
Tyson Reimer
University of Manitoba
July 09th, 2020
"""

import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt


###############################################################################


def init_plt(figsize=(12, 6), labelsize=18):
    """

    Parameters
    ----------
    figsize : tuple
        The figure size
    labelsize : int
        The labelsize for the axis-ticks
    """

    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Libertinus Serif"]
    # For mathtext to match Libertinus Serif:
    # plt.rcParams["mathtext.fontset"] = "dejavuserif"

    plt.figure(figsize=figsize)
    plt.rc("font", family="Libertinus Serif")
    plt.tick_params(labelsize=labelsize)


###############################################################################
