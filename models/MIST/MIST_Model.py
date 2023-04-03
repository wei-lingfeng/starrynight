import os 
import numpy as np 
import csv 
import matplotlib.pyplot as plt
from scipy import interpolate

user_path = os.path.expanduser('~') 

model_path = f"{user_path}/ONC/starrynight/models/MIST/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS"
save_path = 'MIST_Model.csv'

mass = []

for filename in os.listdir(model_path):
    if filename.endswith('.pdf'):
        continue
    mass.append(int(filename[:5]))

mass.sort()
mass = np.array(mass)
mass = mass[mass <= 2000]    # consider M < 20 Solar mass


logL    = []    # (L_sun)
logg    = []    # (cgs)
teff    = []    # (K)

for m in mass:
    # record the evolutionary track for each mass
    age         = []
    logL_evo    = []
    logg_evo    = []
    teff_evo    = []
    
    with open(model_path + '/{:05d}M.track.eep'.format(m), 'r') as file:
        # skip headers
        for _ in range(12):
            next(file)
        lines = file.readlines()
    
    for line in lines:
        data = [float(i) for i in line.split()]
        if data[0] > 1e7:
            break
        age.append(data[0])
        logL_evo.append(data[6])
        logg_evo.append(data[14])
        teff_evo.append(data[11])
    
    # age in Myr
    age = np.array(age) / 1e6
    
    if m == 100:
        print('Solar Mass Star:')
        fig, axs = plt.subplots(3, 1, figsize=(5, 6), sharex=True)
        axs[0].plot(age, np.power(10, teff_evo))
        axs[0].set_ylabel('Teff (K)')
        axs[1].plot(age, logg_evo)
        axs[1].set_ylabel('logg')
        axs[2].plot(age, logL_evo)
        axs[2].set_ylabel('Luminosity')
        axs[2].set_xlabel('Age (Myr)')
        plt.show()
    
    func = interpolate.interp1d(age, logL_evo)
    logL.append(func([2., 1., 3.]))
    func = interpolate.interp1d(age, teff_evo)
    teff.append(func([2., 1., 3.]))
    func = interpolate.interp1d(age, logg_evo)
    logg.append(func([2., 1., 3.]))

mass = mass/100
logL = np.array(logL)
teff = np.power(10, np.array(teff))
logg = np.array(logg)

model = np.hstack((mass[:, None], teff, logg, logL))


# Plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(teff[:, 0], mass, label='2 Myr', zorder=3)
ax.plot(teff[:, 1], mass, label='1 Myr', zorder=2)
ax.plot(teff[:, 2], mass, label='3 Myr', zorder=1)
ax.set_xlabel('Teff (K)')
ax.set_ylabel('Mass ($M_\odot$)')
ax.legend()
ax.set_xlim([2800, 6200])
ax.set_ylim([-1, 9])
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(teff[:, 0], logL[:, 0], label='2 Myr', zorder=3)
ax.plot(teff[:, 1], logL[:, 1], label='1 Myr', zorder=2)
ax.plot(teff[:, 2], logL[:, 2], label='3 Myr', zorder=1)
ax.set_xlabel('teff (K)')
ax.set_ylabel('Luminosity')
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(teff[:, 0], logg[:, 0], label='2 Myr', zorder=3)
ax.plot(teff[:, 1], logg[:, 1], label='1 Myr', zorder=2)
ax.plot(teff[:, 2], logg[:, 2], label='3 Myr', zorder=1)
ax.set_xlabel('teff (K)')
ax.set_ylabel('logg')
plt.show()


# with open(save_path, 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow([
#         'mass', 
#         'teff_2myr', 'teff_1myr', 'teff_3myr', 
#         'logg_2myr', 'logg_1myr', 'logg_3myr',
#         'logL_2myr', 'logL_1myr', 'logL_3myr'
#     ])
#     writer.writerows(model)