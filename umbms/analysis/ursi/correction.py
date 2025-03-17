"""
Illia Prykhodko

University of Manitoba
January 24th, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib.ticker as ticker
from umbms.analysis.stats import ccc

def fun(x, a):#, b):
    """ Linear correction function
    Parameters
    ------------
        x : data
        a : slope
        b : intercept

    Returns:
    ------------
        y : corrected data
    """
    return x * a #+ b


if __name__ == "__main__":

    # Opening the data
    data_dgbe90 = np.genfromtxt('data/dgbe90.csv', delimiter=',',
                                skip_header=1)

    data_dgbe95 = np.genfromtxt('data/dgbe95.csv', delimiter=',',
                                skip_header=1)

    data_dgbe70 = np.genfromtxt('data/dgbe70.csv', delimiter=',',
                                skip_header=1)

    data_glycerin = np.genfromtxt('data/glycerin.csv', delimiter=',',
                                skip_header=1)

    freqs = data_glycerin[:, 0]

    exp_dgbe90 = data_dgbe90[:, 1]
    phase_dgbe90 = data_dgbe90[:, 2]

    exp_dgbe95 = data_dgbe95[:, 1]
    phase_dgbe95 = data_dgbe95[:, 2]

    exp_dgbe70 = data_dgbe70[:, 1]
    phase_dgbe70 = data_dgbe70[:, 2]

    exp_glycerin = data_glycerin[:, 1]
    phase_glycerin = data_glycerin[:, 2]

    # Pick 4 colors uniformly from the inferno colormap
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 1, 4)]

    y_exp = [exp_glycerin, exp_dgbe95, exp_dgbe90, exp_dgbe70]
    y_phase = [phase_glycerin, phase_dgbe95, phase_dgbe90, phase_dgbe70]
    labels = ['Glycerin', 'DGBE 95%', 'DGBE 90%', 'DGBE 70%']

    params = []
    uncertainties = []
    for i in range(4):
        popt, pcov = curve_fit(fun, y_phase[i], y_exp[i],
                               # bounds=([0.7, -np.inf], [1.2, np.inf])
                               )

        params.append(popt)
        uncertainties.append(3*np.sqrt(np.diag(pcov)))
        print(f"{labels[i]}:\na = {popt[0]:.5e} +/- "
              f"{uncertainties[i][0]:.5e}"
              # f"\nb = {popt[1]:.5e} +/- {uncertainties[i][1]:.5e}"
               )
        cor = pearsonr(y_exp[i], fun(y_phase[i], *popt))[0]
        ccc_ = ccc(y_exp[i], fun(y_phase[i], *popt))
        print(f"CCC = {ccc_:.6f}, PCC = {cor:.6f}")
        print(f"Mean ratio = {np.mean(y_exp[i])/np.mean(y_phase[i])}")

    plt.rcParams['font.family'] = 'Times New Roman'
    # N = np.size(freqs)
    # delta = N * np.sum(phase_dgbe70**2) - np.sum(phase_dgbe70)**2
    # sigma_y2 = 1/(N-2) * np.sum((exp_dgbe70 - fun(phase_dgbe70, *popt))**2)
    # sigma_a2 = sigma_y2 / delta * np.sum(phase_dgbe70**2)
    # sigma_b2 = N * sigma_y2 / delta
    # print(sigma_a2)
    # print(sigma_b2)

    __MY_DPI = 120
    fig, ax = plt.subplots(**dict(figsize=(800 / __MY_DPI, 800 / __MY_DPI),
                                  dpi=__MY_DPI))
    plotfreqs = np.linspace(2, 9, 1001)
    mask = plotfreqs > 4
    plotfreqs = plotfreqs[mask]
    for i in range(4):
        # ax.plot(plotfreqs, params[i][0]*y_phase[i], label='Pre-correction')
        ax.plot(plotfreqs, y_exp[i], color=colors[i], label=labels[i])
        ax.plot(plotfreqs, fun(y_phase[i], *params[i]), color=colors[i],
                 linestyle='--')
        ax.plot(plotfreqs, fun(y_phase[i],
                            *[p - u for p, u in zip(params[i],
                                                      uncertainties[i])]),
                 color=colors[i], linewidth=0.4)

        ax.plot(plotfreqs, fun(y_phase[i],
                            *[p + u for p, u in zip(params[i],
                                                    uncertainties[i])]),
                 color=colors[i], linewidth=0.4)

        ax.fill_between(plotfreqs,
                         fun(y_phase[i],
                            *[p - u for p, u in zip(params[i],
                                                    uncertainties[i])]),
                         fun(y_phase[i],
                            *[p + u for p, u in zip(params[i],
                                                    uncertainties[i])]),
                         color=colors[i], alpha=0.2)

    def speed_formatter(value, pos):
        # value is the actual numeric data on the axis (like 2.3e7).
        # Convert to #.## × 10^7 style:
        plt_value = value / 1e7
        exp_str = r'$\cdot$ 10$^7$'
        return f"{plt_value:.1f}" + exp_str

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(speed_formatter))

    ax.set_xlim(3.9, 9.1)
    ax.set_ylim(7.2e7, 14.0e7)
    ax.grid(linewidth=0.5)
    plt.legend(loc='center right', fontsize=14)
    plt.xlabel('Frequency (GHz)', fontsize=16)
    plt.ylabel('Propagation speed (m/s)', fontsize=16)
    plt.tight_layout()
    # plt.show()
    # plt.savefig('C:/Users/prikh/Desktop/correction.png', dpi=__MY_DPI)
