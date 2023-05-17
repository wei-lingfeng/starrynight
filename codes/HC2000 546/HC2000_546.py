import os
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
from thejoker.plot import plot_rv_curves
from thejoker import JokerPrior, TheJoker, RVData

user_path = os.path.expanduser('~')

sources = pd.read_csv('/home/l3wei/ONC/starrynight/catalogs/synthetic catalog.csv')
idx = sources[sources.HC2000=='546'].index.to_list()
rv = [sources.loc[idx[0], 'rv_apogee'], *sources.loc[idx, 'rv_helio']] * u.km/u.s
err = [sources.loc[idx[0], 'rv_e_apogee'], *sources.loc[idx, 'rv_e_nirspec']] * u.km/u.s

# Observed
dates_observe = [
    date(2019, 2, 25),
    date(2020, 1, 21),
    date(2021, 10, 20)
]

t = [_.days for _ in np.diff(dates_observe)]
t.insert(0, 0)

data = RVData(t=t, rv=rv, rv_err=err)
prior = JokerPrior.default(P_min=300*u.day, P_max=600*u.day,
                           sigma_K0=5*u.km/u.s, sigma_v=20*u.km/u.s)
joker = TheJoker(prior)

prior_samples = prior.sample(size=10000)
samples = joker.rejection_sample(data, prior_samples)

fig, ax = plt.subplots(1, 1, figsize=(6,4)) # doctest: +SKIP
t_grid = np.linspace(-60, t[-1]+60, 1024)
plot_rv_curves(samples, t_grid, rv_unit=u.km/u.s, data=data, ax=ax,
               plot_kwargs=dict(color='C7'), data_plot_kwargs=dict(color='k', marker='.', markersize=5, elinewidth=1.2, capsize=3, capthick=1.2))
ax.errorbar(t[0], rv[0], yerr=err[0], color='C0', fmt='.', markersize=5, elinewidth=1.2, capsize=3, capthick=1.2, label='APOGEE')
ax.errorbar(t[1:], rv[1:], yerr=err[1:], color='C3', fmt='.', markersize=5, elinewidth=1.2, capsize=3, capthick=1.2, label='NIRSPAO')
ax.set_ylim((min(rv.value) - 5, max(rv.value) + 7))
ax.set_ylabel(r'RV ($\mathrm{km}\cdot\mathrm{s}^{-1}$)')
ax.set_xlabel(r'Days after 2019.2.25')
ax.legend(loc='lower right')
plt.savefig(f'{user_path}/ONC/figures/HC2000 546.pdf', bbox_inches='tight')
plt.show()