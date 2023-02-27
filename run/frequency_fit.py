import numpy as np
from scipy.optimize import curve_fit
import pandas
import matplotlib.pyplot as plt

vac_permittivity = 8.85e-12


def complex_permittivity(freq, eps_inf, delta_eps, sigma, tau, alpha):
    return eps_inf + delta_eps / (1 + (1j * 2 * np.pi * freq * tau) **
            (1 - alpha)) + sigma / (1j * 2 * np.pi * freq * vac_permittivity)

def permittivity(freq, eps_inf, delta_eps, sigma, tau, alpha):
    return np.real(complex_permittivity(freq, eps_inf, delta_eps, sigma, tau,
                                                                        alpha))


def conductivity(freq, eps_inf, delta_eps, sigma, tau, alpha):
    return np.imag(complex_permittivity(freq, eps_inf, delta_eps, sigma, tau,
                                                    alpha)) * 2 * np.pi * freq

def make_complex_perm(freqs, permittivity, conductivity):
    return np.array(permittivity + 1j * conductivity / (2 * freqs * np.pi))

df = pandas.read_csv('C:/Users/illia/Desktop/ORR/ORR-Algorithm/'
            'data/freq_data/Triton X-100 30% DielectricMeasurementSummary.csv')

xdata = np.array(df["Freq"].values)
perm_ydata = np.array(df["Permittivity"].values)
conduct_ydata = np.array(df["Conductivity"].values)

fig, (perm_fit, cond_fit) = plt.subplots(2, sharex=True)

# for i in range(4):
#     popt_perm, _ = curve_fit(permittivity, xdata[0:801:50],
#                    perm_ydata[0:801:50], bounds=(float(-200 / (i + 1)),
#                    float(200 / (i + 1))), p0=popt_perm)
target_ydata = make_complex_perm(xdata, perm_ydata, conduct_ydata)

popt, _ = curve_fit(complex_permittivity, xdata, target_ydata)

popt_perm, _ = curve_fit(permittivity, xdata, perm_ydata)
popt_cond, _ = curve_fit(conductivity, xdata[0:801], conduct_ydata[0:801],
                                                            bounds=(-290, 290))


# perm_fit.plot(xdata[0:801], perm_ydata[0:801], 'b,',
#                                                   label="Manufacturer data")
# perm_fit.plot(xdata, permittivity(xdata, *popt_perm), 'g--',
#          label=r'fit: $\varepsilon_\infty$=%.4e, $\Delta\varepsilon$=%.4e, '
#                r'$\sigma_s$=%.4e, $\tau$=%.4e, $\alpha$=%.4e'
#                                         % tuple(popt_perm))
#
# perm_fit.set_title("Permittivity fit")
#
# cond_fit.plot(xdata[0:801:15], conduct_ydata[0:801:15], 'b+',
#                                                   label="Manufacturer data")
# cond_fit.plot(xdata, conductivity(xdata, *popt_cond), 'r--',
#           label=r'fit: $\varepsilon_\infty$=%.4e, $\Delta\varepsilon$=%.4e, '
#                 r'$\sigma_s$=%.4e, $\tau$=%.4e, $\alpha$=%.4e'
#                                             % tuple(popt_cond))

# cond_fit.set_title("Conductivity fit")

perm_fit.plot(xdata[0:801], perm_ydata[0:801], 'b,', label="Data")
perm_fit.plot(xdata, permittivity(xdata, *popt), 'g--',
              label=r'fit: $\varepsilon_\infty$=%.4e, $\Delta\varepsilon$=%.4e,'
                    r' $\sigma_s$=%.4e, $\tau$=%.4e, $\alpha$=%.4e'
                                                    % tuple(popt))

perm_fit.set_title("Permittivity fit")

cond_fit.plot(xdata[0:801:15], conduct_ydata[0:801:15], 'b+', label="Data")
cond_fit.plot(xdata, conductivity(xdata, *popt), 'r--',
              label=r'fit: $\varepsilon_\infty$=%.4e, $\Delta\varepsilon$=%.4e,'
                    r' $\sigma_s$=%.4e, $\tau$=%.4e, $\alpha$=%.4e'
                                                    % tuple(popt))

cond_fit.set_title("Conductivity fit")

perm_fit.legend()
cond_fit.legend()

plt.show()
#
# target_freqs = np.array(np.linspace(2e9, 9e9, 1001), dtype=int)
# fitted_perm = permittivity(target_freqs, *popt_perm)
# fitted_cond = conductivity(target_freqs, *popt_cond)
#
# data = {
#     'Freqs': target_freqs,
#     'Permittivity': fitted_perm,
#     'Conductivity': fitted_cond
# }
#
# df = pandas.DataFrame(data)
#
# df.to_csv('Fitted Dielectric Measurements.csv', index=False)
#
# print()
#
