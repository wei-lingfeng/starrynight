# Extract Pala Model
# mass  | teff |   L  |   logg
# teff = [teff_2myr, teff_1myr, teff_3myr]

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

save_path = 'Palla_Model.csv'

with open('palla.mass', 'r') as file:
    for _ in range(2):
        next(file)
    raw = file.read()

raw = raw.split('*** M=')[1:]

mass = []
L    = []    # (L)
teff    = []    # (log10 teff)

for lines in raw:
    lines = lines.split('\n')[:-1]
    m = float(lines[0][:3])
    
    # if m > 2.5:
    #     break
    
    mass.append(m)
    
    age         = []
    teff_evo    = []    # (teff)
    L_evo    = []    # (L/Lo)
    
    for line in lines[1:]:
        data = [float(i) for i in line.split()]
        age.append(data[2])
        L_evo.append(data[1])
        teff_evo.append(data[0])
    
    if abs(m - 1.0) < 1e-3:
        plt.plot(age, teff_evo)
        plt.ylabel('teff (K)')
        plt.show()
        plt.plot(age, L_evo)
        plt.ylabel('Luminosity')
        plt.show()
    
    func = interpolate.interp1d(age, L_evo, bounds_error=False, fill_value=None)
    L.append(func([2, 1, 3]))
    func = interpolate.interp1d(age, teff_evo, bounds_error=False, fill_value=None)
    teff.append(func([2, 1, 3]))

mass = np.array(mass)
L = np.array(L)
teff = np.array(teff)

model = np.hstack((mass[:, None], teff, L))


plt.plot(teff[:, 0], mass)
plt.plot(teff[:, 1], mass)
plt.plot(teff[:, 2], mass)
plt.xlabel('teff (K)')
plt.ylabel('mass')
plt.show()
plt.plot(teff[:, 0], L[:, 0])
plt.plot(teff[:, 1], L[:, 1])
plt.plot(teff[:, 2], L[:, 2])
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