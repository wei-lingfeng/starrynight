import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import linregress
from collections.abc import Iterable
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import AnchoredText

user_path = os.path.expanduser('~')

resampling = 100000
nbins=7
kde_percentile = 84

def binary_mass(masses):
    if not isinstance(masses, Iterable):
        try:
            masses = np.array(masses)
        except:
            raise TypeError(f'Please provide a number or array for masses, instead of {type(masses)}.')
    
    binary_mass = np.empty_like(masses)
    for i, mass in enumerate(masses):
        if 0.075 <= mass < 0.15:
            cf = 0.16
            gamma = 0
        elif 0.15 <= mass < 0.3:
            cf = 0.14
            gamma = 0.7
        elif 0.3 <= mass < 0.675:
            cf = 0.15
            gamma = 0.1
        elif 0.675 <= mass <= 1.0875:
            cf = 0.2
            gamma = 0.2
        else:
            cf = 0.24
            gamma = 0.4
        
        binary_mass[i] = mass + cf*mass*(gamma+1)/(gamma+2)
    return binary_mass


for model_name in ['MIST', 'BHAC15', 'Feiden', 'Palla']:
# for model_name in ['MIST']:
    with open(f'{user_path}/ONC/starrynight/codes/analysis/vrel_results/uniform_dist/linear-0.10pc/{model_name}-mass_vrel.pkl', 'rb') as file:
        mass, vrel, e_mass, e_vrel = pickle.load(file)

    mass = mass.value
    vrel = vrel.value
    e_mass = e_mass.value
    e_vrel = e_vrel.value    
    
    
    # resampling
    ks = np.empty(resampling)
    bs = np.empty(resampling)
    Rs = np.empty(resampling)
    mass_new = binary_mass(mass)

    for i in range(resampling):
        mass_resample = np.random.normal(loc=mass_new, scale=e_mass)
        vrel_resample = np.random.normal(loc=vrel, scale=e_vrel)
        valid_resample_idx = (mass_resample > 0) & (vrel_resample > 0)
        mass_resample = mass_resample[valid_resample_idx]
        vrel_resample = vrel_resample[valid_resample_idx]
        result = linregress(mass_resample, vrel_resample)
        ks[i] = result.slope
        bs[i] = result.intercept
        Rs[i] = np.corrcoef(mass_resample, vrel_resample)[1, 0]

    # p-value
    result = linregress(mass, vrel)
    p = result.pvalue
    
    k_resample      = np.median(ks)
    e_k_resample    = np.diff(np.percentile(ks, [16, 84]))[0]/2
    b_resample      = np.median(bs)
    e_b_resample    = np.diff(np.percentile(bs, [16, 84]))[0]/2
    R_resample      = np.median(Rs)
    e_R_resample    = np.diff(np.percentile(Rs, [16, 84]))[0]/2
    
    
    # running average
    sources_in_bins = [len(_) for _ in np.array_split(np.arange(len(mass)), nbins)]
    division_idx = np.cumsum(sources_in_bins)[:-1] - 1
    mass_sorted = np.sort(mass)
    mass_borders = np.array([np.nanmin(mass) - 1e-3, *(mass_sorted[division_idx] + mass_sorted[division_idx + 1])/2, np.nanmax(mass)])
    
    mass_binned_avrg    = np.empty(nbins)
    e_mass_binned       = np.empty(nbins)
    mass_weight = 1 / e_mass**2
    vrel_binned_avrg    = np.empty(nbins)
    e_vrel_binned       = np.empty(nbins)
    vrel_weight = 1 / e_vrel**2
    
    for i, min_mass, max_mass in zip(range(nbins), mass_borders[:-1], mass_borders[1:]):
        idx = (mass > min_mass) & (mass <= max_mass)
        mass_weight_sum = sum(mass_weight[idx])
        mass_binned_avrg[i] = np.average(mass[idx], weights=mass_weight[idx])
        e_mass_binned[i] = 1/mass_weight_sum * sum(mass_weight[idx] * e_mass[idx])
        
        vrel_weight_sum = sum(vrel_weight[idx])
        vrel_binned_avrg[i] = np.average(vrel[idx], weights=vrel_weight[idx])
        e_vrel_binned[i] = 1/vrel_weight_sum * sum(vrel_weight[idx] * e_vrel[idx])
    
    
    # KDE
    resolution = 200
    X, Y = np.meshgrid(np.linspace(0, mass.max(), resolution), np.linspace(0, vrel.max(), resolution))
    positions = np.vstack([X.T.ravel(), Y.T.ravel()])
    values = np.vstack([mass, vrel])
    kernel = gaussian_kde(values)
    Z = np.rot90(np.reshape(kernel(positions).T, X.shape))
    
    
    # figure
    xs = np.linspace(mass.min(), mass.max(), 2)
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    h1 = ax.errorbar(
        mass, vrel, xerr=e_mass, yerr=e_vrel,
        fmt='.', 
        markersize=6, markeredgecolor='none', markerfacecolor='C0', 
        elinewidth=1, ecolor='C0', alpha=0.5, label=f'Sources - {model_name} Model',
        zorder=2
    )
        # Running Average
    h2 = ax.errorbar(
        mass_binned_avrg, vrel_binned_avrg, 
        xerr=e_mass_binned, 
        yerr=e_vrel_binned, 
        fmt='.', 
        markersize=8, markeredgecolor='none', markerfacecolor='C3', 
        elinewidth=1.2, ecolor='C3', 
        alpha=0.8,
        zorder=4
    )

    # Running Average Fill
    f2 = ax.fill_between(mass_binned_avrg, vrel_binned_avrg - e_vrel_binned, vrel_binned_avrg + e_vrel_binned, color='C3', edgecolor='none', alpha=0.5)
    
    # Model
    h4, = ax.plot(xs, k_resample*xs + b_resample, color='k', label='Best Fit', zorder=3)
    
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cmap = plt.cm.Blues
    my_cmap = cmap(np.linspace(0, 0.5, cmap.N))
    my_cmap = ListedColormap(my_cmap)
    ax.set_facecolor(cmap(0))
    ax.set_facecolor(cmap(0))
    im = ax.imshow(Z, cmap=my_cmap, alpha=0.8, extent=[0, mass.max(), 0, vrel.max()], zorder=0, aspect='auto')
    cs = ax.contour(X, Y, np.flipud(Z), levels=np.percentile(Z, [kde_percentile]), alpha=0.5, zorder=1)
    
    # contour label
    fmt = {cs.levels[0]: f'{kde_percentile}%'}
    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)
    h3 = Line2D([0], [0], color=cs.collections[0].get_edgecolor()[0])
    
    cax = fig.colorbar(im, fraction=0.1, shrink=1, pad=0.03)
    cax.set_label(label='KDE', size=12, labelpad=10)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    handles, labels = ax.get_legend_handles_labels()
    handles = [h1, (h2, f2), h3, h4]
    labels = [
        f'Sources - {model_name} Model',
        'Running Average',
        f"KDE's {kde_percentile}-th Percentile"
    ]
    labels.append(f'Best Fit:\n$k={k_resample:.2f}\pm{e_k_resample:.2f}$\n$b={b_resample:.2f}\pm{e_b_resample:.2f}$')
    ax.legend(handles, labels)
    
    def sci_notation(number, sig_fig=2):
        ret_string = "{0:.{1:d}e}".format(number, sig_fig)
        a, b = ret_string.split("e")
        # remove leading "+" and strip leading zeros
        b = int(b)
        return f'{a}*10^{{{str(b)}}}'
    
    at = AnchoredText(
        f'$p={sci_notation(p)}$\n$R={R_resample:.2f}\pm{e_R_resample:.2f}$', 
        prop=dict(size=10), frameon=True, loc='lower right'
    )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    at.patch.set_alpha(0.8)
    at.patch.set_edgecolor((0.8, 0.8, 0.8))
    ax.add_artist(at)
    
    ax.set_xlabel('Mass $(M_\odot)$', fontsize=12)
    ax.set_ylabel('Relative Velocity (km$\cdot$s$^{-1}$)', fontsize=12)


    plt.savefig(f'{user_path}/ONC/figures/Close Binary-{model_name}.pdf', bbox_inches='tight')
    plt.show()
