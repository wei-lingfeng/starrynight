# Extract Seiss Model
# mass  | teff |   L  |   logg
# teff = [teff_2myr, teff_1myr, teff_3myr]

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

save_path = 'Seiss_Model.csv'

masses = np.concatenate((
    np.arange(0.5, 2.1, 0.1),
    np.array([2.2, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0])
))

L       = []    # (L_sun)
logg    = []    # (cgs)
teff    = []    # (K)

for mass in masses:
    # record the evolutionary track for each mass
    age         = []
    L_evo       = []
    logg_evo    = []
    teff_evo    = []
    
    with open('siess_for_lingfeng/s{:.1f}_02m1.5.hrd'.format(mass), 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        data = [float(i) for i in line.split()]
        if data[-1] > 1e7:
            break
        age.append(data[-1])
        L_evo.append(data[2])
        logg_evo.append(data[5])
        teff_evo.append(data[6])
    
    if abs(mass - 1.0) < 1e-3:
        plt.plot(age, teff_evo)
        plt.ylabel('teff (K)')
        plt.show()
        plt.plot(age, logg_evo)
        plt.ylabel('logg')
        plt.show()
        plt.plot(age, L_evo)
        plt.ylabel('Luminosity')
        plt.show()
    
    func = interpolate.interp1d(age, L_evo)
    L.append(func([2e6, 1e6, 3e6]))
    func = interpolate.interp1d(age, teff_evo)
    teff.append(func([2e6, 1e6, 3e6]))
    func = interpolate.interp1d(age, logg_evo)
    logg.append(func([2e6, 1e6, 3e6]))

L = np.array(L)
teff = np.array(teff)
logg = np.array(logg)

model = np.hstack((masses[:, None], teff, logg, L))

plt.plot(teff[:, 0], masses)
plt.plot(teff[:, 1], masses)
plt.plot(teff[:, 2], masses)
plt.xlabel('teff (K)')
plt.ylabel('mass')
plt.show()
plt.plot(teff[:, 0], L[:, 0])
plt.plot(teff[:, 1], L[:, 1])
plt.plot(teff[:, 2], L[:, 2])
plt.xlabel('teff (K)')
plt.ylabel('Luminosity')
plt.show()
plt.plot(teff[:, 0], logg[:, 0])
plt.plot(teff[:, 1], logg[:, 1])
plt.plot(teff[:, 2], logg[:, 2])
plt.xlabel('teff (K)')
plt.ylabel('logg')
plt.show()


with open(save_path, 'w') as file:
    writer = csv.writer(file)
    writer.writerow([
        'mass', 
        'teff_2myr', 'teff_1myr', 'teff_3myr', 
        'logg_2myr', 'logg_1myr', 'logg_3myr',
        'L_2myr', 'L_1myr', 'L_3myr'
    ])
    writer.writerows(model)