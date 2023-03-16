import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mass = np.concatenate((
    np.arange(0.4, 1.7, 0.1),
    np.array([1.8, 2.0, 2.5])
))

# 1myr, 2myr, 3myr
log_teff = np.array([
    [3.571, 3.560, 3.559],
    [3.601, 3.582, 3.576], 
    [3.621, 3.605, 3.595], 
    [3.638, 3.625, 3.616], 
    [3.652, 3.641, 3.633], 
    [3.663, 3.655, 3.647], 
    [3.673, 3.666, 3.661], 
    [3.680, 3.676, 3.673], 
    [3.686, 3.684, 3.682],
    [3.691, 3.690, 3.691],
    [3.696, 3.696, 3.698], 
    [3.700, 3.701, 3.705], 
    [3.703, 3.706, 3.711], 
    [3.709, 3.715, 3.724], 
    [3.714, 3.723, 3.737], 
    [3.725, 3.745, 3.849]
])

logL = np.array([
    [-.389, -.621, -.732],
    [-.211, -.473, -.598], 
    [-.076, -.325, -.465],
    [.045, -.199, -.339],
    [.149, -.092, -.229], 
    [.238, .005, -.132], 
    [.319, .090, -.042], 
    [.389, .166, .041], 
    [.450, .234, .115], 
    [.507, .296, .187], 
    [.558, .355, .254], 
    [.606, .409, .320], 
    [.651, .462, .383], 
    [.734, .567, .521], 
    [.810, .671, .683], 
    [.993, 1.029, 1.564]
])

teff = 10**log_teff

model_data = pd.DataFrame.from_dict({
    'mass': ['{:.1f}'.format(_) for _ in mass],
    'teff_2myr': teff[:, 1],
    'teff_1myr': teff[:, 0],
    'teff_3myr': teff[:, 2],
    'logL_2myr': logL[:, 1],
    'logL_1myr': logL[:, 0],
    'logL_3myr': logL[:, 2],
})

model_data.to_csv('DAntona_Model.csv', index=False)

fig, ax = plt.subplots()
ax.plot(teff[:, 0], mass, label='1myr')
ax.plot(teff[:, 1], mass, label='2myr')
ax.plot(teff[:, 2], mass, label='3myr')
ax.set_xlabel('Teff')
ax.set_ylabel('Mass')
plt.show()