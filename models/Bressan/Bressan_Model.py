# Extract Bressan Model
# mass  | teff |   L  |   logg
# teff = [teff_2myr, teff_1myr, teff_3myr]

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

save_path = 'Bressan_Model.csv'

masses = np.concatenate((
    np.array([.1, .12, .14, .16]),
    np.arange(0.2, 2.35, 0.05),
    np.arange(2.4, 6.6, 0.2),
    np.arange(7, 13, 1),
    np.array([14, 16, 18, 20])
))

logL    = []    # (log10 L)
teff    = []    # (log10 teff)

for mass in masses:
    # record the evolutionary track for each mass
    age         = []
    logL_evo    = []    # (log10 L)
    teff_evo    = []    # (log10 teff)
    
    with open('Z0.0001Y0.249/Z0.0001Y0.249OUTA1.74_F7_M{0:07.3f}.DAT'.format(mass), 'r') as file:
        next(file)
        lines = file.readlines()
    
    for line in lines:
        data = [float(i) for i in line.split()]
        if data[2] > 1e7:
            break
        age.append(data[2])
        logL_evo.append(data[3])
        teff_evo.append(data[4])
    
    if abs(mass - 1.0) < 1e-3:
        plt.plot(age, teff_evo)
        plt.ylabel('teff (K)')
        plt.show()
        plt.plot(age, logL_evo)
        plt.ylabel('Luminosity')
        plt.show()
    
    func = interpolate.interp1d(age, logL_evo)
    logL.append(func([2e6, 1e6, 3e6]))
    func = interpolate.interp1d(age, teff_evo)
    teff.append(func([2e6, 1e6, 3e6]))

logL = np.array(logL)
teff = np.power(10, np.array(teff))

model = np.hstack((masses[:, None], teff, logL))

plt.plot(teff[:, 0], masses)
plt.plot(teff[:, 1], masses)
plt.plot(teff[:, 2], masses)
plt.xlabel('teff (K)')
plt.ylabel('mass')
plt.show()
plt.plot(teff[:, 0], logL[:, 0])
plt.plot(teff[:, 1], logL[:, 1])
plt.plot(teff[:, 2], logL[:, 2])
plt.xlabel('teff (K)')
plt.ylabel('Luminosity')
plt.show()


with open(save_path, 'w') as file:
    writer = csv.writer(file)
    writer.writerow([
        'mass', 
        'teff_2myr', 'teff_1myr', 'teff_3myr', 
        'logL_2myr', 'logL_1myr', 'logL_3myr'
    ])
    writer.writerows(model)