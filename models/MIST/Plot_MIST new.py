from  Models.MIST import read_mist_models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import compress
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

colors = ['C0', 'C1', 'C2', 'C4']
linestyles = ['--', ':', '-.', (0, (6, 1, 1, 1, 1, 1))]

model = pd.read_csv('MIST_Model.csv')

# iso
iso = read_mist_models.ISO('/home/l3wei/ONC/Models/MIST/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_basic.iso')

# ages = iso.ages
ages = np.array([0.2, 0.5, 10, 20]) # Myr
ages *= 1e6

fig, ax = plt.subplots(figsize=(6, 4))

plot_2myr = False
handles = []
for idx, age in enumerate(ages):
    if (age > 1) & (~plot_2myr):
        plot_2myr = True
        line, = ax.plot(model.teff_2myr, model.mass, color='C3')
        fill, = ax.fill(np.append(model.teff_1myr, model.teff_3myr[::-1]), np.append(model.mass, model.mass[::-1]), facecolor='C3', alpha=0.3)
    i = iso.age_index(np.log10(age))
    teff = 10**iso.isos[i]['log_Teff']
    mass = iso.isos[i]['star_mass']
    handles.append(ax.plot(teff, mass, color=colors[idx%len(colors)], linestyle=linestyles[idx%len(linestyles)], label='{:.1f} Myr'.format(age/1e6))[0])

# Label reordering
_, labels = ax.get_legend_handles_labels()
ax.legend(
    [*list(compress(handles, ages < 1e6)), (line, fill), *list(compress(handles, ages > 3e6))], 
    [*list(compress(labels, ages < 1e6)), r'$2.0 \pm 1.0$ Myr', *list(compress(labels, ages>3e6))], 
    handlelength=2.7
)

ax.set_xlabel('Teff (K)', fontsize=12, labelpad=10)
ax.set_ylabel('MIST Mass ($M_\odot$)', fontsize=12)
ax.set_xlim([2800, 6200])
ax.set_ylim([-0.2, 6.5])
plt.savefig('/home/l3wei/ONC/Figures/MIST Evolutionary Models.pdf')
plt.show()
