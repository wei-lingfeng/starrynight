import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import linregress

nfold = 5
resampling = 100000

with open('/home/weilingfeng/ONC/starrynight/codes/analysis/vrel_results/uniform_dist/linear-0.10pc/MIST-mass_vrel.pkl', 'rb') as file:
    mass, vrel, e_mass, e_vrel = pickle.load(file)

mass = mass.value
vrel = vrel.value
e_mass = e_mass.value
e_vrel = e_vrel.value

np.random.seed(42)
idx = np.arange(len(mass))
np.random.shuffle(idx)
group_idxs = np.array_split(idx, 5)

k_fold = np.empty(nfold)
e_k_fold = np.empty(nfold)
b_fold = np.empty(nfold)
e_b_fold = np.empty(nfold)

mass_fold   = []
e_mass_fold = []
vrel_fold   = []
e_vrel_fold = []

for fold in range(nfold):
    groups = [x for x in range(5) if x!=fold] #e.g.: groups = [0, 1, 2, 4]
    group_idx = np.concatenate([group_idxs[group] for group in groups])
    mass_fold.append(mass[group_idx])
    e_mass_fold.append(e_mass[group_idx])
    vrel_fold.append(vrel[group_idx])
    e_vrel_fold.append(e_vrel[group_idx])
    
    ks = np.empty(resampling)
    bs = np.empty(resampling)
    Rs = np.empty(resampling)
    
    for i in range(resampling):
        mass_resample = np.random.normal(loc=mass[group_idx], scale=e_mass[group_idx])
        vrel_resample = np.random.normal(loc=vrel[group_idx], scale=e_vrel[group_idx])
        valid_resample_idx = (mass_resample > 0) & (vrel_resample > 0)
        mass_resample = mass_resample[valid_resample_idx]
        vrel_resample = vrel_resample[valid_resample_idx]
        result = linregress(mass_resample, vrel_resample)
        ks[i] = result.slope
        bs[i] = result.intercept

    k_fold[fold]    = np.median(ks)
    e_k_fold[fold]  = np.diff(np.percentile(ks, [16, 84]))[0]/2
    b_fold[fold]    = np.median(bs)
    e_b_fold[fold]  = np.diff(np.percentile(bs, [16, 84]))[0]/2


xs = np.linspace(mass.min(), mass.max(), 2)
fig, ax = plt.subplots()
ax.errorbar(mass, vrel, xerr=e_mass, yerr=e_vrel, fmt='.', alpha=0.5)
for fold in range(nfold):
    ax.plot(xs, k_fold[fold]*xs + b_fold[fold], label=f'$k_{fold+1}={k_fold[fold]:.2f}\pm{e_k_fold[fold]:.2f}$')

ax.set_xlabel('Mass $(M_\odot)$', fontsize=12)
ax.set_ylabel('Relative Velocity (km$\cdot$s$^{-1}$)', fontsize=12)
plt.legend()
plt.savefig('/home/weilingfeng/ONC/figures/KFold.pdf', bbox_inches='tight')
plt.show()

print(f'5-Fold k:{np.average(k_fold, weights=1/e_k_fold**2):.2f} ± {np.sqrt(1 / sum(1/e_k_fold**2)):.2f}')
print(f'Std: {np.std(k_fold):.2f}')