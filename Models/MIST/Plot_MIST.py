from  Models.MIST import read_mist_models
import numpy as np
import matplotlib.pyplot as plt

colors = ['C0', 'C1', 'C3', 'C2', 'C4']
linestyles = ['--', ':', '-', '-.', (0, (6, 1, 1, 1, 1, 1))]

# # isocmd
# isocmd = read_mist_models.ISOCMD('/home/l3wei/ONC/Models/MIST/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd')

# ages = isocmd.ages
# fig, ax = plt.subplots(figsize=(6, 4))
# for i in range(0, 50, 10):
#     teff = 10**isocmd.isocmds[i]['log_Teff']
#     gmag = isocmd.isocmds[i]['Gaia_G_EDR3']
#     logg = isocmd.isocmds[i]['log_g']
#     mass = isocmd.isocmds[i]['star_mass']
#     ax.plot(teff, mass, label='{:.1f} Myr'.format(10**ages[i]/1e6))
#     ax.legend()


# ax.set_xlabel('Teff (K)')
# ax.set_ylabel('Mass ($M_\odot$)')
# ax.set_xlim([2800, 6200])
# ax.set_ylim([-1, 9])
# plt.show()


# iso
iso = read_mist_models.ISO('/home/l3wei/ONC/Models/MIST/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_basic.iso')

# ages = iso.ages
ages = np.array([0.1, 0.3, 1., 2., 3., 10.]) # Myr
ages *= 1e6

fig, ax = plt.subplots(figsize=(6, 4))
for idx, age in enumerate(ages):
    i = iso.age_index(np.log10(age))
    teff = 10**iso.isos[i]['log_Teff']
    mass = iso.isos[i]['star_mass']
    ax.plot(teff, mass, color=colors[idx%len(colors)], linestyle=linestyles[idx%len(linestyles)], label='{:.1f} Myr'.format(10**iso.ages[i]/1e6))

ax.legend(handlelength=2.7)
ax.set_xlabel('Teff (K)', fontsize=12)
ax.set_ylabel('MIST Mass ($M_\odot$)', fontsize=12)
ax.set_xlim([2800, 6200])
# ax.set_ylim([-0.2, 5.2])
ax.set_ylim([-0.2, 8])
plt.savefig('/home/l3wei/ONC/Figures/MIST Evolutionary Models.pdf')
plt.show()
