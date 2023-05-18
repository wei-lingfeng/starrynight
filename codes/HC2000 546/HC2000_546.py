import os
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from datetime import date
from scipy.optimize import fsolve
from matplotlib.lines import Line2D
from thejoker.plot import plot_rv_curves
from thejoker import JokerPrior, TheJoker, RVData

user_path = os.path.expanduser('~')

sources = pd.read_csv('/home/l3wei/ONC/starrynight/catalogs/synthetic catalog.csv')
sources_epoch_combined = pd.read_csv('/home/l3wei/ONC/starrynight/catalogs/synthetic catalog - epoch combined.csv')
idx = sources[sources.HC2000=='546'].index.to_list()
M = sources_epoch_combined.loc[sources_epoch_combined.HC2000=='546', 'mass_MIST'].values[0]
M_err = sources_epoch_combined.loc[sources_epoch_combined.HC2000=='546', 'mass_e_MIST'].values[0]
rv = [sources.loc[idx[0], 'rv_apogee'], *sources.loc[idx, 'rv_helio']] * u.km/u.s
rv_err = [sources.loc[idx[0], 'rv_e_apogee'], *sources.loc[idx, 'rv_e_nirspec']] * u.km/u.s

# Observed
dates_observe = [
    date(2019, 2, 25),
    date(2020, 1, 21),
    date(2021, 10, 20)
]

t = [_.days for _ in np.diff(dates_observe)]
t.insert(0, 0)

data = RVData(t=t, rv=rv, rv_err=rv_err)
P_min = 10*u.day
P_max = 600*u.day
prior = JokerPrior.default(
    P_min=P_min, 
    P_max=P_max,
    sigma_K0=5*u.km/u.s, 
    sigma_v=100*u.km/u.s
)
joker = TheJoker(prior)

prior_samples = prior.sample(size=100000)
samples = joker.rejection_sample(data, prior_samples)

valid_idx = []
m = []
a = []
minimum_value = (4/M)**(1/3)
P = samples['P'].to(u.yr).value # yr
K = abs(samples['K'].to(u.au/u.yr).value)   # au/yr
e = samples['e'].value
for i, sample in enumerate(samples):
    value = 2*np.pi/(K[i] * P[i]**(1/3)*(1-e[i]**2)**(1/2))
    if value >= minimum_value:
        mi = fsolve(lambda m: (M + m)**(2/3) / m - value, x0=M/5)[0]
        ai = ((M + mi)*P[i]**2)**(1/3)
        if ai <= 50:
            m.append(mi)
            a.append(ai)
            valid_idx.append(i)

samples = samples[valid_idx]
P = P[valid_idx]    # yr
K = K[valid_idx]    # au/yr
e = e[valid_idx]
m = np.array(m)
a = np.array(a)

# plot sma-mass
lower_bound = lambda m: ((M + m) * P_min.to(u.yr).value**2)**(1/3) # a > a_P_min
upper_bound = lambda m: ((M + m) * P_max.to(u.yr).value**2)**(1/3) # a < a_P_max
left_bound = lambda m: 4*np.pi**2*m**2/(M + m) / (rv.ptp().to(u.au/u.yr).value/2)**2 # a < a_dv_min

m_intersection = fsolve(lambda m: left_bound(m) - upper_bound(m), x0=0.12)
m_grid_left = np.linspace(min(m), m_intersection, 100)
m_grid_right = np.linspace(m_intersection, M, 100)
m_grid = np.concatenate((m_grid_left, m_grid_right)).flatten()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(m, a, 1, color='C7', marker='.', label='Sampled Systems')
ax.plot(m_grid, lower_bound(m_grid), label=f'Minimum Period: {P_min.value:.0f} days')
ax.plot(m_grid, upper_bound(m_grid), linestyle='-.', label=f'Maximum Period: {P_max.value:.0f} days')
ax.plot(m_grid_left, left_bound(m_grid_left), linestyle='--', label=r'Minimum $\Delta v$ in circular orbit')
ax.vlines(M, lower_bound(M), upper_bound(M), color='C3', linestyle=':', label='Maximum Companion Mass')
ax.fill_between(
    m_grid, 
    lower_bound(m_grid),
    np.concatenate((left_bound(m_grid_left), upper_bound(m_grid_right))).flatten(),
    color='C7',
    alpha=0.1,
    label='Allowed Companion'
)
ax.set_xlabel(r'Companion Mass $\left(M_\odot\right)$')
ax.set_ylabel('Semi-Major Axis (au)')
handles, labels = ax.get_legend_handles_labels()
handles[0] = Line2D([], [], marker='.', color='C7', label='Sampled Systems', markersize=5, linestyle='None')
ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(1, -0.023))
plt.show()

# only plot orbits with period > 300 days
plot_orbit = P > (300/365.25)
samples = samples[plot_orbit]

fig, ax = plt.subplots(1, 1, figsize=(6,4)) # doctest: +SKIP
t_grid = np.linspace(-60, t[-1]+60, 1024)
plot_rv_curves(samples, t_grid, rv_unit=u.km/u.s, data=data, ax=ax,
               plot_kwargs=dict(color='C7'), data_plot_kwargs=dict(color='k', marker='.', markersize=5, elinewidth=1.2, capsize=3, capthick=1.2))
ax.errorbar(t[0], rv[0], yerr=rv_err[0], color='C0', fmt='.', markersize=5, elinewidth=1.2, capsize=3, capthick=1.2, label='APOGEE')
ax.errorbar(t[1:], rv[1:], yerr=rv_err[1:], color='C3', fmt='.', markersize=5, elinewidth=1.2, capsize=3, capthick=1.2, label='NIRSPAO')
ax.set_ylim((min(rv.value) - 5, max(rv.value) + 7))
ax.set_ylabel(r'RV ($\mathrm{km}\cdot\mathrm{s}^{-1}$)')
ax.set_xlabel(r'Days after 2019.2.25')
ax.legend(loc='lower right')
# plt.savefig(f'{user_path}/ONC/figures/HC2000 546.pdf', bbox_inches='tight')
plt.show()