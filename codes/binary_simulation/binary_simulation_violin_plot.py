# Generate violin plot in python 3.
# Requires to run binary_simulation_generate_data.py first.

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

user_path = os.path.expanduser('~')
fbins = np.linspace(0, 1, 5, endpoint=True)
v_disps = []
for fbin in fbins:
    with open(f'{user_path}/ONC/starrynight/codes/binary_simulation/v_disp fbin={fbin:.2f}.npy', 'rb') as file:
        v_disps.append(np.load(file))


quartile1, medians, quartile3 = np.percentile(v_disps, [25, 50, 75], axis=1)
hline_width = 0.2 / 3

with open(f'{user_path}/ONC/starrynight/codes/data_processing/vdisp_results/all/mcmc_params.txt', 'r') as file:
    raw = file.readlines()
vdisp_rv, vdisp_rv_e = eval([line for line in raw if line.startswith('σ_rv:')][0].strip('σ_rv:\t\n'))
vdisp_ra, vdisp_ra_e = eval([line for line in raw if line.startswith('σ_RA:')][0].strip('σ_RA:\t\n'))
vdisp_de, vdisp_de_e = eval([line for line in raw if line.startswith('σ_DE:')][0].strip('σ_DE:\t\n'))

# violin plot
fig, ax = plt.subplots(figsize=(6, 5))
violin_plot = ax.violinplot(v_disps, positions=fbins*100, widths=12, quantiles=[[0.25, 0.5, 0.75]]*len(fbins))
# f_inverse(median vdisps) = fbins
f_inverse = interp1d(medians, fbins)
print('Binary fraction needs to be {:.2%} to account for radial velocity dispersion.'.format(f_inverse(vdisp_rv)))

xmin, xmax = ax.get_xlim()

# measured velocity dispersions
alpha=0.2
colors = ['C3', 'C2', 'C4']
sigma_rv_line = ax.hlines(vdisp_rv, xmin=xmin, xmax=xmax, colors=colors[0], linestyles=':', zorder=0)
sigma_rv_fill = ax.fill_between([xmin, xmax], y1=[vdisp_rv - vdisp_rv_e]*2, y2=[vdisp_rv + vdisp_rv_e]*2, color=colors[0], alpha=alpha, zorder=0)
sigma_ra_line = ax.hlines(vdisp_ra, xmin=xmin, xmax=xmax, colors=colors[1], linestyles='-.', zorder=0)
sigma_ra_fill = ax.fill_between([xmin, xmax], y1=[vdisp_ra - vdisp_ra_e]*2, y2=[vdisp_ra + vdisp_ra_e]*2, color=colors[1], alpha=alpha, zorder=0)
sigma_de_line = ax.hlines(vdisp_de, xmin=xmin, xmax=xmax, colors=colors[2], linestyles='--', zorder=0)
sigma_de_fill = ax.fill_between([xmin, xmax], y1=[vdisp_de - vdisp_de_e]*2, y2=[vdisp_de + vdisp_de_e]*2, color=colors[2], alpha=alpha, zorder=0)

for part in violin_plot['bodies']:
    part.set_zorder(1)

ax.legend([
    (violin_plot['bodies'][0], violin_plot['cquantiles']),
    (sigma_rv_line, sigma_rv_fill),
    (sigma_ra_line, sigma_ra_fill),
    (sigma_de_line, sigma_de_fill)
], [
    r'$\sigma_{RV}$ - Simulated',
    r'$\sigma_{RV}$ - Measured',
    r'$\sigma_{RA}$ - Measured',
    r'$\sigma_{DE}$ - Measured'
], loc=(1.04, 0), fontsize=12)

ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xticks(fbins*100)
ax.set_xlabel('Binary Fraction (%)', fontsize=15, labelpad=10)
ax.set_ylabel(r'$\sigma_{RV}~\left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$', fontsize=15, labelpad=10)
# ax.set_title('Binary Simulation Result of Intrinsic Velocity Dispersion', fontsize=16, pad=15)
plt.savefig(f'{user_path}/ONC/figures/Binary Simulation Violin.pdf', bbox_inches='tight')
# plt.savefig(f'{user_path}/ONC/figures/Binary Simulation Violin.png', bbox_inches='tight', transparent=True)
plt.show()