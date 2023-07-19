import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D, axes3d

user_path = os.path.expanduser('~')

with open(f'{user_path}/ONC/starrynight/models/BHAC15/BHAC15.txt', 'r') as file:
    raw = file.readlines()

Age  = np.empty([8, 30])    # Myr
Mass = np.empty([8, 30])    # Solar Mass
Teff = np.empty([8, 30])    # K
Logg = np.empty([8, 30])

# Read the first 8 ages: up to 10 Myr
for n in range(8):
    Age[n] = float(raw[18 + n*38].strip('!t(Gyr)= \n')) * 1e3

    grid = np.loadtxt(raw[22 + n*38 : 51 + n*38 + 1])
    Mass[n] = grid[:, 0]
    Teff[n] = grid[:, 1]
    Logg[n] = grid[:, 3]

Teff = Teff.reshape(-1)
Logg = Logg.reshape(-1)
Mass = Mass.reshape(-1)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(Teff, Logg, Mass)
ax.set_xlabel('Teff (K)')
ax.set_ylabel('Logg')
ax.set_zlabel('Mass $(M_\u2609)$')
ax.view_init(elev=30, azim=240)
plt.show()

points = (Teff, Logg)
Teffq = np.linspace(min(Teff), 4647, num=50);
Loggq = np.linspace(min(Logg), max(Logg), num=50);
Teffq_grid, Loggq_grid = np.meshgrid(Teffq, Loggq)
Model_grid = interpolate.griddata(points, Mass, (Teffq_grid, Loggq_grid))
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(Teffq_grid, Loggq_grid, Model_grid)
ax.set_xlabel('Teff (K)')
ax.set_ylabel('Logg')
ax.set_zlabel('Mass $(M_\u2609)$')
ax.view_init(elev=30, azim=240)
plt.show()