import pickle
import smart
import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_spectrum(date, name, order, spec_lines=None, save_path=None, mark_CO=True, show_figure=False):
    """Plot spectrum with model and CO lines in order 32 or 33.

    Parameters
    ----------
    date : tuple, optional
        Observation date of the source of the format (year, month, day).
    name : int | str
        Name of the object. e.g., 172, '522_A'
    order : int
        Order
    spec_lines: pd.DataFrame
        Lab spectral lines, by default None.
        {
            'name': str | List(str), e.g., 'CO', ['Si', 'Ti'].
            'wavelength': np.array. Wavelength in Angstrom.
            'alpha': float | np.array, optional. 1 by default.
            'label_offset': float | np.array, optional. 0 by default.
        }
    save_path : str, optional
        Save path, by default None
    mark_CO : bool, optional
        mark CO lines or not, by default True
    show_figure : bool, optional
        show figure or not, by default False

    Returns
    -------
    fig, (ax1, ax2)
        figure and axes objects
    """
    year, month, day = date
    
    if order in [32, 33]:
        orders = [32, 33]
    elif order==34:
        orders = [34]
    elif order==35:
        orders = [35]
    else:
        raise ValueError('Order should be 32, 33, 34, or 35.')

    year = str(year).zfill(2)
    month = str(month).zfill(2)
    day = str(day).zfill(2)
    month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    name = str(name)
    common_prefix = '/home/l3wei/ONC/Data/20{}{}{}/reduced/mcmc_median/{}_O{}_params'.format(year, month_list[int(month) - 1], day, name, orders)

    discard = 400
    params = ['teff', 'vsini', 'rv', 'airmass', 'pwv', 'veiling', 'lsf', 'noise']
    for _ in orders:
        params += ['wave_offset_O{}'.format(_), 'flux_offset_O{}'.format(_)]
    
    nparams = len(params)
    sampler = emcee.backends.HDFBackend('{}/sampler.h5'.format(common_prefix))
    flat_samples = sampler.get_chain(discard=discard,  flat=True)

    mcmc = np.empty((3, nparams))
    for i in range(nparams):
        mcmc[:, i] = np.percentile(flat_samples[:, i], [50, 16, 84])
    
    mcmc = np.array([mcmc[0, :], (mcmc[2, :] - mcmc[1, :])/2])
    mcmc = pd.DataFrame(mcmc, columns=params)
    
    c = 299792.458  # km/s
    beta = mcmc.rv[0]/c
    wave_offset = mcmc.loc[0, 'wave_offset_O{}'.format(order)]
    flux_offset = mcmc.loc[0, 'flux_offset_O{}'.format(order)]
    
    with open('{}/sci_specs.pkl'.format(common_prefix), 'rb') as file:
        sci_specs = pickle.load(file)
    sci_spec = sci_specs[[_.order for _ in sci_specs].index(order)]
    model, model_notel = smart.makeModel(mcmc.teff[0], order=str(order), data=sci_spec, logg=4.0, vsini=mcmc.vsini[0], rv=mcmc.rv[0], airmass=mcmc.airmass[0], pwv=mcmc.pwv[0], veiling=mcmc.veiling[0], lsf=mcmc.lsf[0], wave_offset=wave_offset, flux_offset=flux_offset, z=0, modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)

    if mark_CO:
        # Read CO lines
        co_lines = pd.read_csv('/home/l3wei/ONC/Codes/Plot Spectrum/CO lines.csv')
        co_lines.intensity = np.log10(co_lines.intensity)
        if order==32:
            co_lines = co_lines[co_lines.intensity >= -25].reset_index(drop=True)
        co_lines['wavelength'] = 1/co_lines.frequency * 1e8
        co_lines.wavelength *= np.sqrt((1 + beta)/(1 - beta))
        co_lines = co_lines.loc[(co_lines.wavelength >= model.wave[0]) & (co_lines.wavelength <= model.wave[-1])].reset_index(drop=True)
        co_lines['alpha'] = co_lines.intensity - min(co_lines.intensity)
        co_lines.alpha /= (max(co_lines.alpha) / 0.95)
        co_lines.alpha += 0.05
    
    alpha=0.7
    lw = 0.8

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    if mark_CO:
        if order==32:
            ax1.vlines(co_lines.wavelength + wave_offset, max(sci_spec.flux) + 0.02, max(sci_spec.flux) + 0.1, colors='k', alpha=co_lines.alpha, lw=1.2, label='CO lines')
            ax1.text(min(co_lines.wavelength) + wave_offset - 10, max(sci_spec.flux) + 0.055, 'CO', fontsize=12, horizontalalignment='center', verticalalignment='center')
            ax1.margins(x=0.03)
            ax2.margins(x=0.03)
        elif order==33:
            ax1.vlines(co_lines.wavelength + wave_offset, max(sci_spec.flux) + 0.02, max(sci_spec.flux) + 0.08, colors='k', alpha=co_lines.alpha, lw=1.2, label='CO lines')
            ax1.text(min(co_lines.wavelength) + wave_offset - 15, max(sci_spec.flux) + 0.048, 'CO', fontsize=12, horizontalalignment='center', verticalalignment='center')
            ax1.margins(x=0.05)
            ax2.margins(x=0.05)
        else:
            pass    # do not label.
    
    
    if spec_lines is not None:
        spec_lines.wavelength *= np.sqrt((1 + beta)/(1 - beta))
        if 'alpha' not in spec_lines.keys():
            spec_lines['alpha'] = 1
        if 'label_offset' not in spec_lines.keys():
            spec_lines['label_offset'] = 0
        
        ax1.vlines(spec_lines.wavelength + wave_offset, 0.75, 1.1, colors='k', linestyle='dashed', lw=1.2, label='Spectral Lines', alpha=spec_lines.alpha)
        for wavelength, label_offset, spec_name in zip(spec_lines.wavelength, spec_lines.label_offset, spec_lines.name):
            ax1.text(wavelength + label_offset + wave_offset - 4, 1.08, spec_name, fontsize=12, horizontalalignment='center', verticalalignment='bottom')
    
    # ax1.text(1.01, 0.98, '$\mathrm{{T}}_\mathrm{{eff}}$: ${:.2f}\pm{:.2f}$ K\nRV: ${:.2f}\pm{:.2f}$ km$\cdot$s$^{{-1}}$\nvsini: ${:.2f}\pm{:.2f}$ km$\cdot$s$^{{-1}}'.format(mcmc.teff[0], mcmc.teff[1], mcmc.rv[0], mcmc.rv[1], mcmc.vsini[0], mcmc.vsini[1]),
    #     verticalalignment='top', horizontalalignment='left',
    #     transform=ax1.transAxes,
    #     color='k', fontsize=12, bbox={'facecolor': 'none', 'edgecolor': 'C7'})
    
    # ax1.text(
    #     0.02, 0.02, 
    #     '$\mathrm{{T}}_\mathrm{{eff}}$: ${:.2f}\pm{:.2f}$ K, RV: ${:.2f}\pm{:.2f}$ km$\cdot$s$^{{-1}}$, vsini: ${:.2f}\pm{:.2f}$ km$\cdot$s$^{{-1}}$'.format(mcmc.teff[0], mcmc.teff[1], mcmc.rv[0], mcmc.rv[1], mcmc.vsini[0], mcmc.vsini[1]), 
    #     transform=ax1.transAxes,
    #     fontsize=12
    # )
    
    ax1.plot(sci_spec.wave, sci_spec.flux, color='C7', label='Data', alpha=alpha, lw=lw)
    ax1.plot(model_notel.wave, model_notel.flux, color='C3', label='Model', alpha=1, lw=1)
    ax1.plot(model.wave, model.flux, color='C0', label='Model + Telluric', alpha=alpha, lw=lw)
    ax1.minorticks_on()
    ax1.xaxis.tick_top()
    ax1.tick_params(axis='both', labelsize=12, labeltop=False)  # don't put tick labels at the top
    ax1.set_ylabel('Normalized Flux', fontsize=15)
    h1, l1 = ax1.get_legend_handles_labels()
    
    ax2.plot(sci_spec.wave, sci_spec.flux - model.flux, color='k', label='Residual', alpha=0.5, lw=lw)
    ax2.fill_between(sci_spec.wave, -sci_spec.noise, sci_spec.noise, facecolor='0.8', label='Noise')
    ax2.axhline(y=0, color='k', linestyle='--', dashes=(8, 2), alpha=alpha, lw=lw)
    ax2.minorticks_on()
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xlabel(r'$\lambda$ ($\AA$)', fontsize=15)
    ax2.set_ylabel('Residual', fontsize=15)
    h2, l2 = ax2.get_legend_handles_labels()
    custom_lines = [Line2D([], [], color='k', marker='|', linestyle='None',
                            markersize=14, markeredgewidth=1.2),
                    Line2D([], [], color='k', linestyle='dashed', lw=1.2),
                    Line2D([], [], color='C7', alpha=alpha, lw=1.2),
                    Line2D([], [], color='C3', lw=1.2),
                    Line2D([], [], color='C0', lw=1.2),
                    Line2D([], [], color='k', alpha=alpha, lw=1.2),
                    h2[-1]]
    if not mark_CO:
        del custom_lines[0]
    if spec_lines is None:
        del custom_lines[1]
    
    ax2.legend(custom_lines, [*l1, *l2], frameon=True, loc='lower left', bbox_to_anchor=(1, -0.08), fontsize=12, borderpad=0.5)
    fig.align_ylabels((ax1, ax2))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show_figure:
        plt.show()
    
    return fig, (ax1, ax2)


year    = 22
month   = 1
day     = 20
name    = 172
order   = 33

fig, (ax1, ax2) = plot_spectrum(
    date=(year, month, day),
    name=name,
    order=order,
    show_figure=True,
    # save_path='/home/l3wei/ONC/Figures/Spectrum O32.pdf'
)

# plt.close()