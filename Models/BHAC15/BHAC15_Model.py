import numpy as np
import csv 

save_path = 'Baraffe_Model.csv'

with open('BHAC15.txt', 'r') as file:
    raw = file.readlines()
    lines_1myr = raw[60:90]
    lines_2myr = raw[98:128]
    lines_3myr = raw[136:166]

# model[i] = [
#   M_2myr, Teff_2myr, L/Ls_2myr, logg_2myr, Mk_2myr,
#   M_1myr, Teff_1myr, L/Ls_1myr, logg_1myr, Mk_1myr,
#   M_3myr, Teff_3myr, L/Ls_3myr, logg_3myr, Mk_3myr
# ]
model = np.empty((30, 15))

for j, lines in enumerate([lines_2myr, lines_1myr, lines_3myr]):
    for i, line in enumerate(lines):
        data = np.array([float(_) for _ in line.split()])
        model[i, 5*j:5*j+5] = data[[0, 1, 2, 3, -1]]

# rearrange models
# model[i] = [
#   M_2myr, M_1myr, M_3myr,
#   Teff_2myr, Teff_1myr, Teff_3myr,
#   L/Ls_2myr, L/Ls_1myr, L/Ls_3myr,
#   logg_2myr, logg_1myr, logg_3myr, 
#   Mk_2myr, Mk_1myr, Mk_3myr, 
# ]
reorder = np.array([[n, n+5, n+10] for n in range(5)]).flatten()
model = model[:, reorder]

  
with open(save_path, 'w') as file:
    writer = csv.writer(file)
    writer.writerow([
        'mass_2myr', 'mass_1myr', 'mass_3myr', 
        'teff_2myr', 'teff_1myr', 'teff_3myr', 
        'logL_2myr', 'logL_1myr', 'logL_3myr',
        'logg_2myr', 'logg_1myr', 'logg_3myr',
        'Kmag_2myr', 'Kmag_1myr', 'Kmag_3myr',
    ])
    writer.writerows(model)