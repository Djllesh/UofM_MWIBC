"""
Illia Prykhodko

University of Manitoba
January 17th, 2025
"""

from scipy.stats import pearsonr
import numpy as np


def ccc(x, y):
    cor = pearsonr(x, y)[0]

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    var_x = np.var(x)
    var_y = np.var(y)

    std_x = np.std(x)
    std_y = np.std(y)

    return (2 * cor * std_y * std_x) / (var_y + var_x + (mean_x - mean_y) ** 2)

