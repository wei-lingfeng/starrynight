import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

with open('/home/weilingfeng/ONC/starrynight/codes/analysis/vrel_results/uniform_dist/linear-0.10pc/MIST-mass_vrel.pkl', 'rb') as file:
    mass, vrel, e_mass, e_vrel = pickle.load(file)

mass = mass.value
vrel = vrel.value
e_mass = e_mass.value
e_vrel = e_vrel.value

fig, ax = plt.subplots()
ax.errorbar(mass, vrel, xerr=e_mass, yerr=e_vrel, fmt='.', alpha=0.5, label='Data')

x = mass.reshape(-1, 1)
y = vrel.reshape(-1, 1)

lr = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = []
i = 0
for train_index, test_index in kf.split(x):
    i += 1
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lr.fit(x_train, y_train)
    ax.plot([min(x), max(x)], [lr.predict(min(x).reshape(-1, 1))[0], lr.predict(max(x).reshape(-1, 1))[0]], label=f'Fold {i}')
    y_pred = lr.predict(x_test)
    mse_scores.append(mean_squared_error(y_test, y_pred))

ax.set_xlabel('Mass $(M_\odot)$', fontsize=12)
ax.set_ylabel('Relative Velocity (km$\cdot$s$^{-1}$)', fontsize=12)
plt.legend()
plt.show()

print(f"Mean MSE: {np.mean(mse_scores)}")
print(f"STD MSE: {np.std(mse_scores)}")