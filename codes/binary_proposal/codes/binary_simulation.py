from __future__ import division
import csv
import numpy as np
import velbin
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import pickle

new_simulation = False
save = True
data_path = 'binary_simulation.pkl'
n_stars = int(1e7)   # number of stars to generate
precision = 0.5 # km/s

target_mass = 0.53275
tolerance = 0.05  # percentage of tolerance

if new_simulation:
    fbin = 1
    mass_lo, mass_hi = 0.1, 1.5
    # mass_lo, mass_hi = target_mass * (1-tolerance), target_mass * (1+tolerance)
    
    n_binaries = 2*n_stars    # total number of binaries in the sample
    masses = np.random.uniform(mass_lo, mass_hi, n_binaries)
    all_binaries = velbin.solar(nbinaries=n_binaries)
    all_binaries.draw_mass_ratio('flat')
    
    index = (all_binaries.semi_major(masses) < 58.35)
    selected_binaries = all_binaries[index][:n_stars]
    masses = masses[index][:n_stars]
    
    
    times = [0, 3.]
    # times = np.array([0, 330, 968])/365.25
    
    is_bin = np.random.rand(n_stars) < fbin
    rvs = np.array([selected_binaries.velocity(mass=masses, time=time)[0,:] for time in times])
    delta_rvs = np.array([rvs[i+1] - rvs[i] for i in range(len(times)-1)])
    delta_rvs[:, ~is_bin] = 0
    
    if save:
        with open(data_path, 'wb') as file:
            pickle.dump([selected_binaries, rvs, delta_rvs, masses], file)
else:
    with open(data_path, 'rb') as file:
        selected_binaries, rvs, delta_rvs, masses = pickle.load(file)

print('Simulation Finished!')

sma = selected_binaries.semi_major(masses)
ecc = selected_binaries.eccentricity
inc = selected_binaries.inclination
period = selected_binaries.period * 365.25
mass_ratio = selected_binaries.mass_ratio

delta_rv = delta_rvs[0]

#################### Plot #################### 
alpha = 0.5
scatter_alpha = 0.1
marker_size = 0.1
lw = 2

# # delta rv distribution plot
# bins = 500
# limit = 10 # km/s


# fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
# counts = plt.hist(delta_rv, range=(-limit, limit), bins=bins, alpha=0.5, label='binary', log=True)[0]
# handle = ax.fill_between(1 * precision * np.array([-1, -1, 1, 1]), max(counts) * np.array([0, 1, 1, 0]), color='C1', alpha=0.2, edgecolor=None)
# handle.set_edgecolor(None)
# handle = ax.fill_between(1 * precision * np.array([-3, -3, -1, -1]), max(counts) * np.array([0, 1, 1, 0]), color='C3', alpha=0.2, edgecolor=None)
# handle.set_edgecolor(None)
# handle = ax.fill_between(1 * precision * np.array([1, 1, 3, 3]), max(counts) * np.array([0, 1, 1, 0]), color='C3', alpha=0.2)
# handle.set_edgecolor(None)
# ax.vlines(1 * precision * np.array([-1, 1]), 0, max(counts), colors='C1', linestyles='dashed', label='$1\sigma$ detection limit')
# ax.vlines(3 * precision * np.array([-1, 1]), 0, max(counts), colors='C3', linestyles='dashed', label='$3\sigma$ detection limit')
# ax.annotate('', 
#     xy=(7.5, 4e4), xycoords='data', 
#     xytext=(1.5, 4e4), textcoords='data',
#     arrowprops=dict(facecolor=[.5, .5, .5], linewidth=0),
#     zorder=0
# )
# ax.text(4.2, 5e4, 'Detectable binaries', fontsize=12, ha='center')
# ax.annotate('', 
#     xy=(-7.5, 4e4), xycoords='data', 
#     xytext=(-1.5, 4e4), textcoords='data',
#     arrowprops=dict(facecolor=[.5, .5, .5], linewidth=0),
#     zorder=0
# )
# ax.text(-4.2, 5e4, 'Detectable binaries', fontsize=12, ha='center')
# ax.hlines(4e4, 0.5, 1.5, colors=[.5, .5, .5], linestyle=(0, (0.5, 0.3)), lw=4, zorder=0)
# ax.hlines(4e4, -1.5, -0.5, colors=[.5, .5, .5], linestyle=(0, (0.5, 0.3)), lw=4, zorder=0)
# ax.set_ylim(bottom=1e3)
# ax.legend()
# ax.set_xlabel('$\Delta \mathrm{RV}$ (km/s)', fontsize=16)
# ax.set_ylabel('Number of Sources', fontsize=16)
# if save:
#     plt.savefig('RV Distribution.png')
# plt.show()


# fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
# counts = plt.hist(abs(delta_rv), range=(0, limit), bins=bins, alpha=0.5, label='binary', log=True)[0]
# handle = ax.fill_between(1 * precision * np.array([0, 0, 1, 1]), max(counts) * np.array([0, 1, 1, 0]), color='C1', alpha=0.2, edgecolor=None)
# handle.set_edgecolor(None)
# handle = ax.fill_between(1 * precision * np.array([1, 1, 3, 3]), max(counts) * np.array([0, 1, 1, 0]), color='C3', alpha=0.2, edgecolor=None)
# handle.set_edgecolor(None)
# ax.vlines(1 * precision, 0, max(counts), colors='C1', linestyles='dashed', label='$1\sigma$ detection limit')
# ax.vlines(3 * precision, 0, max(counts), colors='C3', linestyles='dashed', label='$3\sigma$ detection limit')
# ax.annotate('', 
#     xy=(4.8, 1e5), xycoords='data', 
#     xytext=(1.5, 1e5), textcoords='data',
#     arrowprops=dict(facecolor=[.5, .5, .5], linewidth=0),
#     zorder=0
# )
# ax.hlines(1e5, 0.5, 1.5, colors=[.5, .5, .5], lw=4, linestyle=(0, (0.7, 0.5)), zorder=0)
# ax.text(3, 12e4, 'Detectable binaries', fontsize=12, ha='center')
# ax.set_ylim(bottom=1e3)
# ax.legend()
# # ax.set_xticks(np.arange(0, limit+1, 3))
# ax.set_xlabel('$|\Delta \mathrm{RV}|$ (km/s)', fontsize=16)
# ax.set_ylabel('Number of Sources', fontsize=16)
# if save:
#     plt.savefig('RV Absolute Distribution.png')
# plt.show()


# detectable_1s = sum(abs(delta_rv) > 1 * precision) / n_stars
# detectable_3s = sum(abs(delta_rv) > 3 * precision) / n_stars
# print('Fraction of binaries above 1-sigma detection limit: {:.2f}%.'.format(detectable_1s * 100))
# print('Fraction of binaries above 3-sigma detection limit: {:.2f}%.'.format(detectable_3s * 100))
# print('Fraction of |delta rv| > {:d} km/s: {:.2f}%.'.format(limit, sum(abs(delta_rv) > limit) / n_stars * 100))
# print('Maximum delta rv: {:.2f} km/s'.format(max(abs(delta_rv))))


# # delta rv vs period plot
# fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
# ax.scatter(period, abs(delta_rv), marker_size, linewidths=0, alpha=scatter_alpha)
# ax.hlines(1 * precision, min(period), max(period), lw=lw, colors='C1', linestyles='dashed', label='$1\sigma$ detection limit')
# ax.hlines(3 * precision, min(period), max(period), lw=lw, colors='C3', linestyles='dashed', label='$3\sigma$ detection limit')
# ax.legend()
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_ylim(bottom=0.001)
# ax.set_xlabel('Period (days)', fontsize=12)
# ax.set_ylabel('$|\Delta \mathrm{RV}|$ (km/s)', fontsize=12)
# if save:
#     plt.savefig('RV vs Period.png')
# plt.show()

# # period fraction histogram
# hist_all, bin_edges = np.histogram(np.log10(period), bins=200, range=(min(np.log10(period)), 5))
# hist_detctable, bin_edges = np.histogram(np.log10(period[abs(delta_rv) > 3*precision]), bins=200, range=(min(np.log10(period)), 5))
# fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
# ax.bar((10**bin_edges[1:] + 10**bin_edges[:-1])/2, hist_detctable/hist_all, width=10**bin_edges[1:] - 10**bin_edges[:-1], alpha=alpha)
# ax.set_xscale('log')
# ax.set_xlabel('Period (day)', fontsize=12)
# ax.set_ylabel('Detectable fraction', fontsize=12)
# if save:
#     plt.savefig('Detectable fraction vs Period histogram.png')
# plt.show()


# delta rv vs semi-major axis plot
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
ax.scatter(sma, abs(delta_rv), marker_size, linewidths=0, alpha=scatter_alpha)
ax.hlines(1 * precision, min(sma), max(sma), lw=lw, colors='C1', linestyles='dashed', label='$1\sigma$ detection limit')
ax.hlines(3 * precision, min(sma), max(sma), lw=lw, colors='C3', linestyles='dashed', label='$3\sigma$ detection limit')
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(left=0.001, right=100)
ax.set_ylim(bottom=0.001)
ax.margins(0.05)
ax.set_xlabel('Semi-major axis (au)', fontsize=12)
ax.set_ylabel('$|\Delta \mathrm{RV}|$ (km/s)', fontsize=12)
if save:
    plt.savefig('RV vs Semi-major axis.png', bbox_inches='tight')
plt.show()

# # semi-major axis fraction histogram
# hist_all, bin_edges = np.histogram(np.log10(sma), bins=200)
# hist_detctable, bin_edges = np.histogram(np.log10(sma[abs(delta_rv) > 3*precision]), bins=200, range=(min(np.log10(sma)), max(np.log10(sma))))
# fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
# ax.bar((10**bin_edges[1:] + 10**bin_edges[:-1])/2, hist_detctable/hist_all, width=10**bin_edges[1:] - 10**bin_edges[:-1], alpha=alpha)
# ax.set_xscale('log')
# ax.set_ylim((0, 1))
# ax.set_xlabel('Semi-major axis (au)', fontsize=12)
# ax.set_ylabel('Detectable fraction', fontsize=12)
# if save:
#     plt.savefig('Detectable fraction vs Semi-major axis histogram.png')
# plt.show()


# # delta rv vs eccentricity plot
# fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
# ax.scatter(ecc, abs(delta_rv), marker_size, linewidths=0, alpha=scatter_alpha)
# ax.hlines(1 * precision, min(ecc), max(ecc), lw=lw, colors='C1', linestyles='dashed', label='$1\sigma$ detection limit')
# ax.hlines(3 * precision, min(ecc), max(ecc), lw=lw, colors='C3', linestyles='dashed', label='$3\sigma$ detection limit')
# ax.legend()
# ax.set_yscale('log')
# ax.set_xlim((0-0.05, 1+0.05))
# ax.set_ylim(bottom=0.001)
# ax.set_xlabel('Eccentricity', fontsize=12)
# ax.set_ylabel('$|\Delta \mathrm{RV}|$ (km/s)', fontsize=12)
# if save:
#     plt.savefig('RV vs Eccentricity.png')
# plt.show()


# # Eccentricity fraction histogram
# hist_all, bin_edges = np.histogram(ecc, bins=100, range=(0, 1))
# hist_detctable, bin_edges = np.histogram(ecc[abs(delta_rv) > 3*precision], bins=100, range=(0, 1))
# fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
# ax.bar((bin_edges[1:] + bin_edges[:-1])/2, hist_detctable/hist_all, width=(bin_edges[1] - bin_edges[0]), alpha=alpha)
# # ax.set_yscale('log')
# ax.set_xlim(right=1)
# ax.set_xlabel('Eccentricity', fontsize=12)
# ax.set_ylabel('Detectable fraction', fontsize=12)
# if save:
#     plt.savefig('Detectable fraction vs Eccentricity histogram.png')
# plt.show()


# # delta rv vs inclination plot
# fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
# ax.scatter(inc * 180/np.pi, abs(delta_rv), marker_size, linewidths=0, alpha=scatter_alpha)
# ax.hlines(1 * precision, min(inc * 180/np.pi), max(inc * 180/np.pi), lw=lw, colors='C1', linestyles='dashed', label='$1\sigma$ detection limit')
# ax.hlines(3 * precision, min(inc * 180/np.pi), max(inc * 180/np.pi), lw=lw, colors='C3', linestyles='dashed', label='$3\sigma$ detection limit')
# ax.legend()
# ax.set_xticks(range(0, 181, 30))
# ax.set_yscale('log')
# ax.set_xlim((-0.05*180, 1.05 * 180))
# ax.set_ylim(bottom=0.01, top=3e3)
# ax.set_xlabel('Inclination (degree)', fontsize=12)
# ax.set_ylabel('$|\Delta \mathrm{RV}|$ (km/s)', fontsize=12)
# if save:
#     plt.savefig('RV vs Inclination.png')
# plt.show()

# # Inclination fraction histogram
# hist_all, bin_edges = np.histogram(inc * 180/np.pi, bins=100, range=(0, 180))
# hist_detctable, bin_edges = np.histogram(inc[abs(delta_rv) > 3*precision] * 180/np.pi, bins=100, range=(0, 180))
# fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
# ax.bar((bin_edges[1:] + bin_edges[:-1])/2, hist_detctable/hist_all, width=(bin_edges[1] - bin_edges[0]), alpha=alpha)
# ax.set_xticks(range(0, 181, 30))
# ax.set_xlabel('Inclination (au)', fontsize=12)
# ax.set_ylabel('Detectable fraction', fontsize=12)
# if save:
#     plt.savefig('Detectable fraction vs Inclination histogram.png')
# plt.show()


# # delta rv vs mass
# fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
# ax.scatter(masses, abs(delta_rv), marker_size, linewidths=0, alpha=scatter_alpha)
# ax.hlines(1 * precision, min(masses), max(masses), lw=lw, colors='C1', linestyles='dashed', label='$1\sigma$ detection limit')
# ax.hlines(3 * precision, min(masses), max(masses), lw=lw, colors='C3', linestyles='dashed', label='$3\sigma$ detection limit')
# ax.legend()
# ax.set_xticks(np.arange(0.1, 1.6, 0.2))
# ax.set_yscale('log')
# ax.set_ylim(bottom=0.001, top=3e3)
# ax.set_xlabel('Primary Mass ($M_\odot$)', fontsize=12)
# ax.set_ylabel('$|\Delta \mathrm{RV}|$ (km/s)', fontsize=12)
# if save:
#     plt.savefig('RV vs Mass.png')
# plt.show()


# # Primary mass fraction histogram
# hist_all, bin_edges = np.histogram(masses, bins=100)
# hist_detctable, bin_edges = np.histogram(masses[abs(delta_rv) > 3*precision], bins=100, range=(min(masses), max(masses)))
# fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
# ax.bar((bin_edges[1:] + bin_edges[:-1])/2, hist_detctable/hist_all, width=(bin_edges[1] - bin_edges[0]), alpha=alpha)
# ax.set_ylim((0.2, 0.5))
# ax.set_xticks(np.arange(0.1, 1.6, 0.2))
# ax.set_xlabel('Primary Mass ($M_\odot$)', fontsize=12)
# ax.set_ylabel('Detectable fraction', fontsize=12)
# if save:
#     plt.savefig('Detectable fraction vs Primary mass histogram.png')
# plt.show()


# # delta rv vs mass ratio
# fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
# ax.scatter(mass_ratio, abs(delta_rv), marker_size, linewidths=0, alpha=scatter_alpha)
# ax.hlines(1 * precision, min(mass_ratio), max(mass_ratio), lw=lw,  colors='C1', linestyles='dashed', label='$1\sigma$ detection limit')
# ax.hlines(3 * precision, min(mass_ratio), max(mass_ratio), lw=lw, colors='C3', linestyles='dashed', label='$3\sigma$ detection limit')
# ax.legend()
# ax.set_yscale('log')
# ax.set_ylim(bottom=0.001, top=3e3)
# ax.set_xlabel('Mass Ratio', fontsize=12)
# ax.set_ylabel('$|\Delta \mathrm{RV}|$ (km/s)', fontsize=12)
# if save:
#     plt.savefig('RV vs Mass Ratio.png')
# plt.show()


# # Mass ratio fraction histogram
# hist_all, bin_edges = np.histogram(mass_ratio, bins=100, range=(0, 1))
# hist_detctable, bin_edges = np.histogram(mass_ratio[abs(delta_rv) > 3*precision], bins=100, range=(0, 1))
# fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
# ax.bar((bin_edges[1:] + bin_edges[:-1])/2, hist_detctable/hist_all, width=(bin_edges[1] - bin_edges[0]), alpha=alpha)
# ax.set_xlabel('Mass Ratio', fontsize=12)
# ax.set_ylabel('Detectable fraction', fontsize=12)
# if save:
#     plt.savefig('Detectable fraction vs Mass ratio histogram.png')
# plt.show()


## color mesh ##
seps      = np.logspace(np.log10(0.003), np.log10(60), 50)
mass_lims = np.logspace(np.log10(1e-3), np.log10(1), 50)
ratios = np.empty((len(seps), len(mass_lims)))


for i in range(len(mass_lims)-1):
    for j in range(len(seps)-1):
        RVs = delta_rv[np.where(
            (mass_ratio >= mass_lims[i]) & (mass_ratio < mass_lims[i+1]) & 
            (sma >= seps[j]) & (sma < seps[j+1])
        )]
        
        detectable = np.where(abs(RVs) >= 1.5)[0].size
        
        if RVs.size != 0:
            ratios[i, j] = detectable / float(RVs.size)
            # ax.text(seps[j], mass_lims[i],  '%s'%len(RVs), ha='left', va='bottom',  fontsize=7, color='k', zorder=100)
        else:
            ratios[i, j] = np.nan
        
        print('Running...{:.2f}%'.format((i*(len(mass_lims)-1) + j) / ( (len(mass_lims)-1) * (len(seps)-1) ) * 100))

fig, ax = plt.subplots(dpi=300)
cax = ax.pcolormesh(seps, mass_lims, ratios, vmin=0, vmax=1, cmap='coolwarm', zorder=-100)
ax.set_xscale('log')
ax.set_yscale('log')
cbar = plt.colorbar(cax)
cbar.set_label('Fraction of detectable binaries')
ax.axis('tight')
ax.set_xlabel('Separation (AU)')
ax.set_ylabel('Mass ratio')
if save:
    plt.savefig('mass ratio vs separation.pdf', bbox_inches='tight')
plt.show()



# # Match HC2000 546
# # APOGEE  19-02-25 29.14(Chris) 29.3595(APOGEE)
# # NIRSPEC 20-01-21 21.24937 (696 days after epoch1)
# # NIRSPEC 21-10-20 26.20320 (968 days after epoch1)


# # tolerance = 0.02  # percentage of tolerance
# rv_epoch1 = 29.3595 - 27.15
# rv_epoch2 = 21.24937 - 27.15
# rv_epoch3 = 26.20320 - 27.15
# # target_mass = 0.53275

# # target_delta_rv = np.diff([rv_epoch1, rv_epoch2, rv_epoch3])

# # possible_binaries = np.where(
# #     (
# #         (abs(delta_rv - target_delta_rv[0]) < tolerance * abs(target_delta_rv[0])) &
# #         (abs(delta_rv2 - target_delta_rv[1]) < tolerance * abs(target_delta_rv[1])) &
# #         (abs(masses - target_mass) < tolerance * target_mass)) |
# #     (
# #         (abs(delta_rv + target_delta_rv[0]) < tolerance * abs(target_delta_rv[0])) &
# #         (abs(delta_rv2 + target_delta_rv[1]) < tolerance * abs(target_delta_rv[1])) &
# #         (abs(masses - target_mass) < tolerance * target_mass))
# # )[0]

# rvs[0, ~is_bin] = np.nan
# rvs[1, ~is_bin] = np.nan
# rvs[2, ~is_bin] = np.nan

# possible_binaries = np.where(
#     (
#         (abs(rvs[0] - rv_epoch1) < tolerance * abs(rv_epoch1)) &
#         (abs(rvs[1] - rv_epoch2) < tolerance * abs(rv_epoch2)) &
#         (abs(rvs[2] - rv_epoch3) < tolerance * abs(rv_epoch3)) &
#         (abs(masses - target_mass) < tolerance * target_mass)) | 
#     (
#         (abs(rvs[0] + rv_epoch1) < tolerance * abs(rv_epoch1)) &
#         (abs(rvs[1] + rv_epoch2) < tolerance * abs(rv_epoch2)) &
#         (abs(rvs[2] + rv_epoch3) < tolerance * abs(rv_epoch3)) &
#         (abs(masses - target_mass) < tolerance * target_mass)) 
# )[0]

# print(len(possible_binaries))

# if save:
#     with open('HC2000 546.csv', 'w') as file:
#         writer = csv.writer(file)
#         writer.writerow(['rv 1', 'rv 2', 'rv 3', 'primary mass', 'secondary mass', 'semi-major axis', 'eccentricity', 'inclination', 'phase', 'theta'])
#         writer.writerows([[
#             rvs[0, i],
#             rvs[1, i],
#             rvs[2, i],
#             masses[i],
#             masses[i] * mass_ratio[i],
#             sma[i],
#             ecc[i],
#             inc[i],
#             selected_binaries.phase[i],
#             selected_binaries.theta[i]
#         ]
#         for i in possible_binaries])