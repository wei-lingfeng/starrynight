import numpy as np
import csv 

age = [2, 1, 3]

# [mass_2,1,3myr, teff_2,1,3myr, logg_2,1,3myr, L_2,1,3myr]

model_std = np.empty((144, 12))  # Standard
model_mag = np.empty((144, 12))  # Magnetic

for j in range(len(age)):
    # standard    
    with open('iso/std/dmestar_0000{:d}.0myr_z+0.00_a+0.00_phx.iso'.format(age[j]), 'r') as file:
        for _ in range(5):
            next(file)
        
        for i, line in enumerate(file.readlines()):
            model_std[i, 4*j:4*j+4] = [float(_) for _ in line.split()[:4]]
    
    # magnetic
    with open('iso/mag/dmestar_0000{:d}.0myr_z+0.00_a+0.00_phx_magBeq.iso'.format(age[j]), 'r') as file:
        for _ in range(5):
            next(file)
        
        for i, line in enumerate(file.readlines()):
            model_mag[i, 4*j:4*j+4] = [float(_) for _ in line.split()[:4]]


# rearrange models
model_std = model_std[:, [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]]
model_mag = model_mag[:, [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]]

# unify units
model_std[:, 3:6] = np.power(10, model_std[:, 3:6])
model_mag[:, 3:6] = np.power(10, model_mag[:, 3:6])

with open('Feiden_std_Model.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow([
        'mass_2myr', 'mass_1myr', 'mass_3myr',
        'teff_2myr', 'teff_1myr', 'teff_3myr',
        'logg_2myr', 'logg_1myr', 'logg_3myr',
        'logL_2myr', 'logL_1myr', 'logL_3myr'
    ])
    writer.writerows(model_std)

with open('Feiden_mag_Model.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow([
        'mass_2myr', 'mass_1myr', 'mass_3myr',
        'teff_2myr', 'teff_1myr', 'teff_3myr',
        'logg_2myr', 'logg_1myr', 'logg_3myr',
        'logL_2myr', 'logL_1myr', 'logL_3myr'
    ])
    writer.writerows(model_mag)
    