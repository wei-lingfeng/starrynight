import os
import numpy as np
import pandas as pd
import thejoker as tj
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pymc3 as pm
import exoplanet.units as xu
from datetime import date
from scipy.optimize import fsolve
from matplotlib.lines import Line2D
from thejoker.plot import plot_rv_curves
from thejoker import JokerPrior, TheJoker, RVData

user_path = os.path.expanduser('~')

new_sampling = True

sources = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog.csv')
sources_epoch_combined = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog - epoch combined.csv')
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
delta_t = t[-1] * u.day

# rv = [29.14, 21.337643, 25.994355] * u.km/u.s
# rv_err = [0.38, 0.27737549, 0.19132987] * u.km/u.s
# t = [0, 330, 638] * u.day

data = RVData(t=t, rv=rv, rv_err=rv_err)
P_min = 10*u.day
P_max = 2*t[-1]*u.day

if new_sampling:
    
    with pm.Model() as model:
        e = xu.with_unit(pm.Uniform('e', 0, 0.9), u.dimensionless_unscaled)

        prior = JokerPrior.default(
            P_min=P_min, 
            P_max=P_max,
            sigma_K0=5*u.km/u.s, 
            sigma_v=100*u.km/u.s,
            pars={'e': e}
        )
    joker = TheJoker(prior)
    prior_samples = prior.sample(size=100000)
    samples = joker.rejection_sample(data, prior_samples)
    samples.write(f'{user_path}/ONC/starrynight/codes/Parenago_1837/samples.hdf5', overwrite=True)
else:
    samples = tj.JokerSamples.read(f'{user_path}/ONC/starrynight/codes/Parenago_1837/samples.hdf5')

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

print(f'{len(valid_idx)} valid samples out of {len(samples)}.')
samples = samples[valid_idx]
P = P[valid_idx]    # yr
K = K[valid_idx]    # au/yr
e = e[valid_idx]
m = np.array(m)
a = np.array(a)

period = lambda m, P : ((M + m) * P.to(u.yr).value**2)**(1/3)   # define sma(mass, period), period constraint derived from Kepler's third law
lower_bound = lambda m : period(m, P_min) # a > sma(mass, P_min)
upper_bound = lambda m : period(m, P_max) # a < sma(mass, P_max)
left_bound_circular = lambda m : 4*np.pi**2*m**2/(M + m) / (rv.ptp().to(u.au/u.yr).value/2)**2 # a < sma(mass, e=0), Δv constraint derived from K (see Murray & Correia 2010 equation 66).
left_bound = lambda m, e : left_bound_circular(m) / (1 - e**2)  # a < sma(mass, e)

print(f'Maximum e for a < left_bound_circular: {max(e[a > left_bound_circular(m)]):.2f}')
e_max = 0.9

Ps = [P_min, delta_t/4, delta_t/3, delta_t/2, delta_t, P_max]
solve_intersection = lambda m, P, e : left_bound(m, e) - period(m, P) # solve for m intersections where Δv constraint = period constraint
m_intersections = np.array([fsolve(lambda m : solve_intersection(m, _, e_max), x0=0.03)[0] for _ in Ps]) # intersection between period and e_max
m_intersections_circular = np.array([fsolve(lambda m : solve_intersection(m, _, 0), x0=0.03)[0] for _ in [P_min, P_max]]) # intersection between period and e=0

# m_grid for boundaries
m_grids = [np.linspace(m_intersection, M, 100) for m_intersection in m_intersections]
m_grid_left = np.linspace(m_intersections[0], m_intersections[-1], 100) # m grid for left bound
m_grid_left_circular = np.linspace(m_intersections_circular[0], m_intersections_circular[-1], 100)  # m grid for Δv constraint with e=0
m_grid_fill = np.concatenate((m_grid_left, m_grids[-1])).flatten()   # m grid for fill


fig, ax = plt.subplots(1, 1, figsize=(6.18, 6.18))
ax.scatter(m, a, 1, color='C7', marker='.', label='Sampled Systems')
ax.plot(m_grids[0], lower_bound(m_grids[0]), linewidth=2, label=f'Min. Period: {P_min.value:.0f} days', zorder=5)
ax.plot(m_grids[-1], upper_bound(m_grids[-1]), linewidth=2, linestyle='-.', label=f'Max. Period: {P_max.value:.0f} days', zorder=4)
ax.plot(m_grid_left_circular, left_bound_circular(m_grid_left_circular), linewidth=2, linestyle='--', label=r'$\Delta v_\mathrm{{max}}\geq\Delta v_\mathrm{{obs}}$, $e = 0$', zorder=3)
ax.plot(m_grid_left, left_bound(m_grid_left, e_max), linewidth=2, linestyle='--', label=rf'$\Delta v_\mathrm{{max}}\geq\Delta v_\mathrm{{obs}}$, $e = {e_max:.1f}$', zorder=2)
ax.vlines(M, lower_bound(M), upper_bound(M), linewidth=2, color='k', label='Max. Companion Mass', zorder=1)
for i in range(1, len(m_grids)-1):
    if i==1:
        ax.plot(m_grids[i], period(m_grids[i], Ps[i]), color='C4', linewidth=1.5, linestyle=':', zorder=0, label='Forbidden Periods')
    else:
        ax.plot(m_grids[i], period(m_grids[i], Ps[i]), color='C4', linewidth=1.5, linestyle=':', zorder=0)
ax.fill_between(
    m_grid_fill, 
    lower_bound(m_grid_fill),
    np.concatenate((left_bound(m_grid_left, e_max), upper_bound(m_grids[-1]))).flatten(),
    color='C7',
    alpha=0.2,
    label='Allowed Companion',
    zorder=0
)
ax.set_xscale('log')
ax.set_xlim((0.0081, 0.78))
ax.set_ylim(bottom=-0.2)

log_percentile = lambda x, percentile : 10**(np.percentile(np.log10(np.array(x)[[0, -1]]), percentile))
log_middle = lambda x : log_percentile(x, 50)
ax.annotate(rf'$P={P_min.value:.0f}$ days', xy=(log_percentile(m_grids[0], 60), lower_bound(log_percentile(m_grids[0], 60)) - 0.05), horizontalalignment='center', verticalalignment='top', size=12)
ax.annotate(rf'$P={P_max.value:.0f}$ days', xy=(log_middle(m_grids[-1]) - 0.02, upper_bound(log_middle(m_grids[-1])) - 0.05), horizontalalignment='center', verticalalignment='bottom', size=12, rotation=15)
ax.annotate(rf'$\Delta v_\mathrm{{max}}\geq\Delta v_\mathrm{{obs}}$, $e={e_max:.1f}$', xy=(log_percentile(m_grid_left, 60) + 0.005, left_bound(log_percentile(m_grid_left, 60), e_max)), horizontalalignment='center', verticalalignment='bottom', size=12, rotation=71)
ax.annotate(rf'$\Delta v_\mathrm{{max}}\geq\Delta v_\mathrm{{obs}}$, $e=0$', xy=(log_percentile(m_grid_left_circular, 60) + 0.012, left_bound_circular(log_percentile(m_grid_left_circular, 60))), horizontalalignment='center', verticalalignment='bottom', size=12, rotation=70)
ax.annotate(rf'$m_\mathrm{{max}}={M:.2f}~M_\odot$', xy=(0.6, (lower_bound(M) + upper_bound(M))/2), horizontalalignment='center', verticalalignment='center', size=12, rotation=-90)

ax.annotate('$P=\Delta t$', xy=(0.3, period(0.3, Ps[-2])), horizontalalignment='center', verticalalignment='bottom', size=12, rotation=15)
ax.annotate('$P=\Delta t/2$', xy=(0.3, period(0.3, Ps[-3])), horizontalalignment='center', verticalalignment='bottom', size=12, rotation=10)
ax.annotate('$P=\Delta t/3$', xy=(0.3, period(0.3, Ps[-4]) - 0.015), horizontalalignment='center', verticalalignment='bottom', size=12, rotation=9)
ax.annotate('$P=\Delta t/4$', xy=(0.3, period(0.3, Ps[-5]) - 0.02), horizontalalignment='center', verticalalignment='top', size=12, rotation=7)

ax.xaxis.set_major_formatter(mticker.ScalarFormatter()) # set to regular format
ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel(r'Companion Mass $\left(M_\odot\right)$', fontsize=15, labelpad=10)
ax.set_ylabel('Semi-Major Axis (au)', fontsize=15, labelpad=10)
handles, labels = ax.get_legend_handles_labels()
handles[0] = Line2D([], [], marker='.', color='C7', label='Sampled Systems', markersize=5, linestyle='None')
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.24), fontsize=11, ncol=2)
plt.savefig(f'{user_path}/ONC/figures/Parenago 1837 - Allowed Param.pdf', bbox_inches='tight')
plt.show()

# only plot orbits with period > Δt/3
plot_orbit = P > (max(t)/3/365.25)
print(f'{sum(plot_orbit)} orbits plotted.')

fig, ax = plt.subplots(1, 1, figsize=(6, 4)) # doctest: +SKIP
t_grid = np.linspace(-60, t[-1]+60, 1024)
plot_rv_curves(samples[plot_orbit], t_grid, rv_unit=u.km/u.s, data=data, ax=ax,
               plot_kwargs=dict(color='C7', linewidth=0.4), data_plot_kwargs=dict(color='k', marker='.', markersize=5, elinewidth=1.2, capsize=3, capthick=1.2))
ax.errorbar(t[0], rv[0], yerr=rv_err[0], color='C0', fmt='.', markersize=5, elinewidth=1.2, capsize=3, capthick=1.2, label='APOGEE')
ax.errorbar(t[1:], rv[1:], yerr=rv_err[1:], color='C3', fmt='.', markersize=5, elinewidth=1.2, capsize=3, capthick=1.2, label='NIRSPAO')
ax.set_ylim((min(rv.value) - 5, max(rv.value) + 7))
ax.set_ylabel(r'RV ($\mathrm{km}\cdot\mathrm{s}^{-1}$)')
ax.set_xlabel(r'Days after 2019.2.25')
ax.legend(loc='lower right')
plt.savefig(f'{user_path}/ONC/figures/Parenago 1837 - Orbital Fits.pdf', bbox_inches='tight')
plt.show()