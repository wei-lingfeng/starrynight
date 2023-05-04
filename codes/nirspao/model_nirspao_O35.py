# Weighted average spectrums of each object

import os, sys, shutil
import pickle
import copy
os.environ["OPENBLAS_NUM_THREADS"] = "1" # Limit number of threads
os.environ["OMP_NUM_THREADS"] = "4" # Limit number of threads
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt 
import smart
import emcee
import corner
import plotly.graph_objects as go
from multiprocessing import Pool
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from collections.abc import Iterable
np.set_printoptions(threshold=sys.maxsize)


def plot_spectrum(sci_spec, result, save_path=None, mark_CO=True, show_figure=False):
    """Plot spectrum with model and CO lines in order 32 or 33.

    Parameters
    ----------
    sci_spec : smart.forward_model.classSpectrum.Spectrum
        science spectrum
    result : dictionary
        dictionary with teff, rv, wave_offset, etc.
    save_path : str, optional
        save path, by default None
    mark_CO : bool, optional
        mark CO lines or not, by default True
    show_figure : bool, optional
        show figure or not, by default False

    Returns
    -------
    fig, (ax1, ax2)
        figure and axes objects
    """
    
    order = sci_spec.header['ECHLORD']
    model, model_notel = smart.makeModel(result['teff'][0], data=sci_spec, order=order, logg=4.0, vsini=result['vsini'][0], rv=result['rv'][0], airmass=result['airmass'][0], pwv=result['pwv'][0], veiling=result['veiling'][0], lsf=result['lsf'][0], wave_offset=result[f'wave_offset_O{order}'][0], z=0, modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)
    if mark_CO:
        # Read CO lines
        co_lines = pd.read_csv('/home/weilingfeng/ONC/starrynight/codes/plot_spectrum/CO lines.csv')
        co_lines.intensity = np.log10(co_lines.intensity)
        if order==32:
            co_lines = co_lines[co_lines.intensity >= -25].reset_index(drop=True)
        co_lines['wavelength'] = 1/co_lines.frequency * 1e8
        c = 299792.458  # km/s
        beta = result['rv'][0]/c
        co_lines.wavelength *= np.sqrt((1 + beta)/(1 - beta))
        co_lines = co_lines[(co_lines.wavelength >= model.wave[0]) & (co_lines.wavelength <= model.wave[-1])].reset_index(drop=True)
        co_lines['alpha'] = co_lines.intensity - min(co_lines.intensity)
        co_lines.alpha /= (max(co_lines.alpha) / 0.95)
        co_lines.alpha += 0.05
    
    alpha=0.7
    lw = 0.8

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    if mark_CO:
        if order==32:
            ax1.vlines(co_lines.wavelength + result[f'wave_offset_O{order}'][0], ymin=max(sci_spec.flux) + 0.03*np.median(sci_spec.flux), ymax=max(sci_spec.flux) + 0.12*np.median(sci_spec.flux), colors='k', alpha=co_lines.alpha, lw=1.2)
            ax1.text(min(co_lines.wavelength) + result[f'wave_offset_O{order}'][0] - 10, max(sci_spec.flux) + 0.07*np.median(sci_spec.flux), 'CO', fontsize=12, horizontalalignment='center', verticalalignment='center')
            ax1.margins(x=0.03)
            ax2.margins(x=0.03)
        elif order==33:
            ax1.vlines(co_lines.wavelength + result[f'wave_offset_O{order}'][0], ymin=max(sci_spec.flux) + 0.02*np.median(sci_spec.flux), ymax=max(sci_spec.flux) + 0.07*np.median(sci_spec.flux), colors='k', alpha=co_lines.alpha, lw=1.2)
            ax1.text(min(co_lines.wavelength) + result[f'wave_offset_O{order}'][0] - 15, max(sci_spec.flux) + 0.043*np.median(sci_spec.flux), 'CO', fontsize=12, horizontalalignment='center', verticalalignment='center')
            ax1.margins(x=0.05)
            ax2.margins(x=0.05)
        else:
            pass    # do not label.
            
    ax1.plot(sci_spec.wave, sci_spec.flux, color='C7', alpha=alpha, lw=lw)
    ax1.plot(model_notel.wave, model_notel.flux, color='C3', alpha=1, lw=1)
    ax1.plot(model.wave, model.flux, color='C0', alpha=alpha, lw=lw)
    ax1.minorticks_on()
    ax1.xaxis.tick_top()
    ax1.tick_params(axis='both', labelsize=12, labeltop=False)  # don't put tick labels at the top
    ax1.set_ylabel('Flux (counts/s)', fontsize=15)

    ax2.plot(sci_spec.wave, sci_spec.flux - model.flux, color='k', alpha=0.5, lw=lw)
    ax2.fill_between(sci_spec.wave, -sci_spec.noise, sci_spec.noise, facecolor='0.8')
    ax2.axhline(y=0, color='k', linestyle='--', dashes=(8, 2), alpha=alpha, lw=lw)
    ax2.minorticks_on()
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xlabel(r'$\lambda$ ($\AA$)', fontsize=15)
    ax2.set_ylabel('Residual', fontsize=15)
    
    legend_elements = [
        Line2D([], [], color='k', marker='|', linestyle='None', markersize=14, markeredgewidth=1.2, label='CO Lines'),
        Line2D([], [], color='C7', alpha=alpha, lw=1.2, label='Combined Spectrum'),
        Line2D([], [], color='C3', lw=1.2, label='Model'),
        Line2D([], [], color='C0', lw=1.2, label='Model + Telluric'),
        Line2D([], [], color='k', alpha=alpha, lw=1.2, label='Residual'),
        Patch(facecolor='0.8', label='Noise')
    ]
    
    if not mark_CO:
        legend_elements.pop(0)
    
    ax2.legend(handles=legend_elements, frameon=True, loc='lower left', bbox_to_anchor=(1, -0.08), fontsize=12, borderpad=0.5)
    fig.align_ylabels((ax1, ax2))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    
    texts = '\n'.join((
        f"$T_\mathrm{{eff}}={result['teff'][0]:.2f}\pm{result['teff'][1]:.2f}$ K",
        f"$V_r={result['rv_helio']:.2f}\pm{result['rv'][1]:.2f}$ km$\cdot$s$^{{-1}}$",
        f"$v\sin i={result['vsini'][0]:.2f}\pm{result['vsini'][1]:.2f}$ km$\cdot$s$^{{-1}}$",
        f"$\mathrm{{AM}}={result['airmass'][0]:.2f}\pm{result['airmass'][1]:.2f}$",
        f"$\mathrm{{PWV}}={result['pwv'][0]:.2f}\pm{result['pwv'][1]:.2f}$ mm",
        f"$\Delta v_\mathrm{{inst}}={result['lsf'][0]:.2f}\pm{result['lsf'][1]:.2f}$ km$\cdot$s$^{{-1}}$",
        f"$C_\mathrm{{veil}}={result['veiling'][0]:.2f}\pm{result['veiling'][1]:.2f}$",
        f"$C_\mathrm{{noise}}={result['noise'][0]:.2f}\pm{result['noise'][1]:.2f}$",
        f"$C_\lambda={result[f'wave_offset_O{order}'][0]:.2f}\pm{result[f'wave_offset_O{order}'][1]:.2f}~\AA$"
    ))
        
    ax1.text(
        1.0142, 
        0.975, 
        texts, 
        fontsize=12, linespacing=1.5, horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, 
        bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.2", ec='0.8', fc='1')
    )

    if save_path is not None:
        if save_path.endswith('.png'):
            plt.savefig(save_path, bbox_inches='tight', transparent=True)
        else:
            plt.savefig(save_path, bbox_inches='tight')
    if show_figure:
        plt.show()
    
    return fig, (ax1, ax2)


##################################################
############## Probability Function ##############
##################################################

def lnprior(theta, orders, limits):
    
    teff, vsini, rv, airmass, pwv, veiling, lsf, noise = theta[:-len(orders)]
    wave_offsets = theta[-len(orders):]
    
    if      limits.teff[0]          < teff      < limits.teff[1]    \
    and     limits.vsini[0]         < vsini     < limits.vsini[1]   \
    and     limits.rv[0]            < rv        < limits.rv[1]      \
    and     limits.airmass[0]       < airmass   < limits.airmass[1] \
    and     limits.pwv[0]           < pwv       < limits.pwv[1]     \
    and     limits.veiling[0]       < veiling   < limits.veiling[1] \
    and     limits.lsf[0]           < lsf       < limits.lsf[1]     \
    and     limits.noise[0]         < noise     < limits.noise[1]   \
    and all(limits.wave_offset[0]   < _         < limits.wave_offset[1] for _ in wave_offsets):
        return 0.0
    else:
        return -np.inf


def lnlike(theta, sci_specs, orders):
    
    teff, vsini, rv, airmass, pwv, veiling, lsf, noise = theta[:-len(orders)]
    wave_offsets = theta[-len(orders):]
    
    sci_noises = np.array([])
    sci_fluxes = np.array([])
    model_fluxes = np.array([])

    for sci_spec, wave_offset, order in zip(sci_specs, wave_offsets, orders):
        model = smart.makeModel(teff, logg=4.0, vsini=vsini, rv=rv, airmass=airmass, pwv=pwv, veiling=veiling, lsf=lsf, z=0, wave_offset=wave_offset, order=order, data=sci_spec, modelset='phoenix-aces-agss-cond-2011')
        sci_noises = np.concatenate((sci_noises, sci_spec.noise * noise))
        sci_fluxes = np.concatenate((sci_fluxes, sci_spec.flux))
        model_fluxes = np.concatenate((model_fluxes, model.flux))

    sigma2 = sci_noises**2
    chi = -1/2 * np.sum( (sci_fluxes - model_fluxes)**2 / sigma2 + np.log(2*np.pi*sigma2) )
    if np.isnan(chi):
        return -np.inf
    else:
        return chi

    
def lnprob(theta, sci_specs, orders, limits):
    lp = lnprior(theta, orders, limits)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike(theta, sci_specs, orders)

##################################################
################## Model NIRSPAO #################
##################################################
def model_nirspao(infos, orders=[32, 33], initial_mcmc=True, finetune=True, finetune_mcmc=True, multiprocess=True, nwalkers=100, steps=300, **kwargs):
    """Fit teff and other params using emcee

    Parameters
    ----------
    infos : dict
        dictionary with keys 'date' (tuple), 'name' (int or str), 'sci_frames' (list of int), 'tel_frames' (list of int)
    orders : list, optional
        list of orders, by default [32, 33]
    initial_mcmc : bool, optional
        run a new emcee sampler or read existing results, by default True
    finetune : bool, optional
        run another finetuning mcmc sampler after removing pixels different from the model larger than 3σ or not, by default True
    finetune_mcmc : bool, optional
        run mcmc for finetune or read from previously saved file, by default True
    multiprocess : bool, optional
        use multiprocess or not, by default True
    nwalkers : int, optional
        number of walkers in emcee, by default 100
    steps : int, optional
        number of steps in emcee, by default 500
    kwargs : 
        limits : dict, optional
            dictonary with keys teff, vsini, rv, airmass, pwv, veiling, lsf, noise, wave_offset
        priors : dict, optional
            begining priors, same as limits if any key is unspecified by default
    Returns
    -------
    result : dict
        result.
    """
    
    date = infos['date']
    name = infos['name']
    name = str(name)
    sci_frames = infos['sci_frames']
    tel_frames = infos['tel_frames']
    
    print(f'Date:\t20{"-".join(str(_).zfill(2) for _ in date)}')
    print(f'Object:\t{name}')
    print(f'Science  Frames:\t{sci_frames}')
    print(f'Telluric Frames:\t{tel_frames}')
    print()
    
    sys.stdout.flush()
    
    # modify parameters
    params = ['teff', 'vsini', 'rv', 'airmass', 'pwv', 'veiling', 'lsf', 'noise']
    for order in orders:
        params += ['wave_offset_O{}'.format(order)]
    params_stripped = [_.strip('|'.join([f'_O{order}' for order in orders])) for _ in params]
    
    nparams = len(params)
    discard = steps - 100
    
    year = str(date[0]).zfill(2)
    month = str(date[1]).zfill(2)
    day = str(date[2]).zfill(2)
    
    month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    prefix = f'/home/weilingfeng/ONC/data/nirspao/20{year}{month_list[int(month) - 1]}{day}/reduced'
    save_path = f'{prefix}/mcmc_median/{name}_O{orders}_params/'
    
    if initial_mcmc:
        if os.path.exists(save_path): shutil.rmtree(save_path)
    
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    limits = kwargs.get('limits', {
        'teff':         (2300, 7000),
        'vsini':        (0, 100),
        'rv':           (-100, 100),
        'airmass':      (1, 3),
        'pwv':          (.5, 10),
        'veiling':      (0, 1e20),
        'lsf':          (1, 20),
        'noise':        (1, 50),
        'wave_offset':  (-1, 1)
    })
    
    priors = kwargs.get('priors', limits)
    for key, value in limits.items():
        if key not in priors.keys():
            priors[key] = value
    
    limits = pd.DataFrame(limits)
    priors = pd.DataFrame(priors)
    
        
    ##################################################
    ############### Construct Spectrums ##############
    ##################################################
    # sci_specs = [order 32 median combined sci_spec, order 33 median combined sci_spec]
    sci_specs = []
    barycorrs = []
    
    for order in orders:
        sci_abba = []
        tel_abba = []
        
        ##################################################
        ####### Construct Spectrums for each order #######
        ##################################################
        for sci_frame, tel_frame in zip(sci_frames, tel_frames):
            
            if int(year) > 18:
                # For data after 2018, sci_names = [nspec200118_0027, ...]
                sci_name = 'nspec' + year + month + day + '_' + str(sci_frame).zfill(4)
                tel_name = 'nspec' + year + month + day + '_' + str(tel_frame).zfill(4)
                pixel_start = 20
                pixel_end = -48
            
            else:
                # For data prior to 2018 (2018 included)
                sci_name = month_list[int(month) - 1] + day + 's' + str(sci_frame).zfill(4)
                tel_name = month_list[int(month) - 1] + day + 's' + str(tel_frame).zfill(4)
                pixel_start = 10
                pixel_end = -30
            
            if name.endswith('A'):
                sci_spec = smart.Spectrum(name=f'{sci_name}_A', order=order, path=f'{prefix}/extracted_binaries/{sci_name}/O{order}')
            elif name.endswith('B'):
                sci_spec = smart.Spectrum(name=f'{sci_name}_B', order=order, path=f'{prefix}/extracted_binaries/{sci_name}/O{order}')
            else:
                sci_spec = smart.Spectrum(name=sci_name, order=order, path=f'{prefix}/nsdrp_out/fits/all')
            
            sci_spec.pixel = np.arange(len(sci_spec.wave))
            sci_spec.snr = np.median(sci_spec.flux / sci_spec.noise)
            
            # if os.path.exists(f'{prefix}/{tel_name}_defringe/O{order}/{tel_name}_defringe_calibrated_{order}_all.fits'):
            #     tel_name = tel_name + '_defringe'
            
            tel_spec = smart.Spectrum(name=f'{tel_name}_calibrated', order=order, path=f'{prefix}/{tel_name}/O{order}')
            
            # Update the wavelength solution
            sci_spec.updateWaveSol(tel_spec)
            
            # Automatically mask out edge & flux < 0
            mask1 = [True if (sci_spec.flux[i] < 0) or (i < pixel_start) or (i >= len(sci_spec.wave) + pixel_end) else False for i in np.arange(len(sci_spec.wave))]
            sci_spec.pixel  = ma.MaskedArray(sci_spec.pixel, mask=mask1)
            sci_spec.wave   = ma.MaskedArray(sci_spec.wave,  mask=mask1)
            sci_spec.flux   = ma.MaskedArray(sci_spec.flux,  mask=mask1)
            sci_spec.noise  = ma.MaskedArray(sci_spec.noise, mask=mask1)

            # Mask flux > median + 3 sigma
            median_flux = ma.median(sci_spec.flux)
            upper_bound = median_flux + 3.*ma.std(sci_spec.flux - median_flux)
            mask2 = sci_spec.flux > upper_bound
            sci_spec.pixel  = ma.MaskedArray(sci_spec.pixel, mask=mask2)
            sci_spec.wave   = ma.MaskedArray(sci_spec.wave,  mask=mask2)
            sci_spec.flux   = ma.MaskedArray(sci_spec.flux,  mask=mask2)
            sci_spec.noise  = ma.MaskedArray(sci_spec.noise, mask=mask2)

            # Mask isolated bad pixels
            median_flux = ma.median(sci_spec.flux)
            lower_bound = median_flux - 3.5*ma.std(sci_spec.flux - median_flux)
            lowest_bound = median_flux - 5.*ma.std(sci_spec.flux - median_flux)
            mask3 = [False, *[True if (sci_spec.flux[i] < lowest_bound) and (sci_spec.flux[i-1] >= lower_bound) and (sci_spec.flux[i+1] >= lower_bound) else False for i in np.arange(1, len(sci_spec.wave)-1)], False]
            sci_spec.pixel  = ma.MaskedArray(sci_spec.pixel, mask=mask3)
            sci_spec.wave   = ma.MaskedArray(sci_spec.wave,  mask=mask3)
            sci_spec.flux   = ma.MaskedArray(sci_spec.flux,  mask=mask3)
            sci_spec.noise  = ma.MaskedArray(sci_spec.noise, mask=mask3)
            
            # Special Case
            if date == (19, 1, 12) and order == 32:
                mask4 = sci_spec.wave > 23980
                sci_spec.pixel  = ma.MaskedArray(sci_spec.pixel, mask=mask4)
                sci_spec.wave   = ma.MaskedArray(sci_spec.wave,  mask=mask4)
                sci_spec.flux   = ma.MaskedArray(sci_spec.flux,  mask=mask4)
                sci_spec.noise  = ma.MaskedArray(sci_spec.noise, mask=mask4)
            
            barycorrs.append(smart.barycorr(sci_spec.header).value)
            
            sci_abba.append(copy.deepcopy(sci_spec))
            tel_abba.append(copy.deepcopy(tel_spec))
        
        itime = sci_spec.header['ITIME']
        
        ##################################################
        ################ Weighted Average ################
        ##################################################
        # normalize to the highest snr frame
        median_flux = ma.median(sci_abba[np.argmax([_.snr for _ in sci_abba])].flux)
        for spec in sci_abba:
            normalize_factor = median_flux / ma.median(spec.flux)
            spec.flux   *= normalize_factor
            spec.noise  *= normalize_factor

        # tel_spec = tel_abba[np.argmin([_.header['RMS'] for _ in tel_abba])]
        
        # weighted average of abba frames
        sci_spec = copy.deepcopy(sci_abba[np.argmin([_.header['RMS'] for _ in tel_abba])])
        sci_spec.flux = ma.average(ma.array([_.flux for _ in sci_abba]), weights=1/ma.array([_.noise for _ in sci_abba])**2, axis=0)
        # sci_spec.noise = ma.sqrt(ma.std(ma.array([_.flux for _ in sci_abba]), axis=0)**2 + ma.sum(ma.array([_.noise for _ in sci_abba])**2, axis=0) / ma.sum(~ma.array([_.noise.mask for _ in sci_abba]), axis=0))
        sci_spec.noise = ma.sqrt(ma.sum(ma.array([_.noise for _ in sci_abba])**2, axis=0) / ma.sum(~ma.array([_.noise.mask for _ in sci_abba]), axis=0))
        sci_spec.pixel.mask = sci_spec.flux.mask
        sci_spec.wave.mask = sci_spec.flux.mask

        sci_spec.pixel  = sci_spec.pixel.compressed()
        sci_spec.wave   = sci_spec.wave.compressed()
        sci_spec.flux   = sci_spec.flux.compressed()
        sci_spec.noise  = sci_spec.noise.compressed()
        
        # numpy version instead of masked array:
        # fluxes = np.array([_.flux for _ in sci_abba])
        # noises = np.array([_.noise for _ in sci_abba])
        # weights = 1/noises**2
        # weighted_flux = fluxes*weights
        
        # sci_spec.flux = np.nansum(weighted_flux, axis=0) / np.nansum(weights, axis=0)
        # sci_spec.noise = np.sqrt(np.nansum(noises**2, axis=0) / np.sum(~np.isnan(noises), axis=0))
        
        # # drop nans
        # valid_idx = ~np.isnan(sci_spec.flux)
        # sci_spec.pixel = sci_spec.pixel[valid_idx]
        # sci_spec.wave  = sci_spec.wave [valid_idx]
        # sci_spec.flux  = sci_spec.flux [valid_idx]
        # sci_spec.noise = sci_spec.noise[valid_idx]
        
        # append sci_specs        
        sci_specs.append(copy.deepcopy(sci_spec))
        
        # plot coadded spectrum
        fig_data = []
        for i, spec in enumerate(sci_abba):
            spec.pixel  = spec.pixel.compressed()
            spec.wave   = spec.wave.compressed()
            spec.flux   = spec.flux.compressed()
            spec.noise  = spec.noise.compressed()
            fig_data.append(go.Scatter(x=spec.pixel, y=spec.flux, mode='lines+markers', name=f'Frame {i+1}', line=dict(width=1, color='#7f7f7f'), marker=dict(size=3)))
            fig_data.append(go.Scatter(x=spec.pixel, y=spec.noise, mode='lines+markers', name=f'Noise {i+1}', line=dict(width=1, color='#7f7f7f'), marker=dict(size=3)))

        fig_data.append(go.Scatter(x=sci_spec.pixel, y=sci_spec.flux, mode='lines+markers', name='Coadd Spectrum', line=dict(width=1, color='#1f77b4'), marker=dict(size=3)))
        fig_data.append(go.Scatter(x=sci_spec.pixel, y=sci_spec.noise, mode='lines+markers', name='Coadd Noise', line=dict(width=1, color='#1f77b4'), marker=dict(size=3)))

        fig = go.Figure()
        fig.add_traces(fig_data)
        fig.update_layout(width=1000, height=500, xaxis = dict(tickformat='000'))
        fig.update_layout(xaxis_title='Pixel', yaxis_title='Flux')
        fig.write_html(save_path + f'spectrum_coadd_plotly_O{order}.html')
        
        
        # plot coadd spectrum
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        for i, spec in enumerate(sci_abba):
            ax1.plot(spec.pixel, spec.flux, color='C7', lw=1, alpha=0.8)
            ax2.plot(spec.pixel, spec.noise, color='C7', lw=1, alpha=0.7)
        ax1.plot(sci_spec.pixel, sci_spec.flux, color='C0', lw=1)
        ax2.plot(sci_spec.pixel, sci_spec.noise, color='C0', lw=1)
        
        ax1.minorticks_on()
        ax1.xaxis.tick_top()
        ax1.tick_params(axis='both', labelsize=12, labeltop=False)  # don't put tick labels at the top
        ax1.set_ylabel('Flux (counts/s)', fontsize=15)
        
        ax2.minorticks_on()
        ax2.tick_params(axis='both', labelsize=12)
        ax2.set_xlabel('Pixel', fontsize=15)
        ax2.set_ylabel('Noise', fontsize=15)
        
        legend_elements = [
            Line2D([], [], color='C7', lw=0.8, alpha=0.7, label='Original Spectra'),
            Line2D([], [], color='C0', lw=1, label='Coadded Spectrum')
        ]
        
        ax2.legend(handles=legend_elements, frameon=True, loc='lower left', bbox_to_anchor=(1, -0.08), fontsize=12, borderpad=0.5)
        fig.align_ylabels((ax1, ax2))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(save_path + f'spectrum_coadd_O{order}.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    barycorr = np.median(barycorrs)
    
    # Save sci_specs
    with open(save_path + 'sci_specs.pkl', 'wb') as file:
        pickle.dump(sci_specs, file)
    
    ##################################################
    ################## MCMC Fitting ##################
    ##################################################
    initial_state = np.array([
        np.random.uniform(priors[f'{param}'][0], priors[f'{param}'][1], size=nwalkers) for param in params_stripped
    ]).transpose()
    
    move = [emcee.moves.KDEMove()]
    
    if initial_mcmc:
        print('MCMC...')
        print(f'Date:\t20{"-".join(str(_).zfill(2) for _ in date)}')
        print(f'Object:\t{name}')
        print(f'Science  Frames:\t{sci_frames}')
        print(f'Telluric Frames:\t{tel_frames}')
        print()
        backend = emcee.backends.HDFBackend(save_path + 'sampler1.h5')
        backend.reset(nwalkers, nparams)
        if multiprocess:
            with Pool(64) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders, limits), moves=move, backend=backend, pool=pool)
                sampler.run_mcmc(initial_state, steps, progress=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders, limits), moves=move, backend=backend)
            sampler.run_mcmc(initial_state, steps, progress=True)
        
        print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
        print(sampler.acceptance_fraction)
        print()
    else:
        sampler = emcee.backends.HDFBackend(save_path + 'sampler1.h5')
    
    
    ##################################################
    ################# Analyze Output #################
    ##################################################

    flat_samples = sampler.get_chain(discard=discard, flat=True)

    # mcmc[:, i] (2 by N) = [value, error]
    mcmc = np.empty((2, nparams))
    for i in range(nparams):
        mcmc[:, i] = np.array(np.median(flat_samples[:, i]), np.diff(np.percentile(flat_samples[:, i], [15.9, 84.1]))[0]/2)
    
    mcmc = pd.DataFrame(mcmc, columns=params)
        
    
    ##################################################
    ################ Construct Models ################
    ##################################################
    models = []
    models_notel = []
    other_params = {}
    for i, order in enumerate(orders):
        model, model_notel = smart.makeModel(mcmc.teff[0], data=sci_specs[i], order=order, logg=4.0, vsini=mcmc.vsini[0], rv=mcmc.rv[0], airmass=mcmc.airmass[0], pwv=mcmc.pwv[0], veiling=mcmc.veiling[0], lsf=mcmc.lsf[0], wave_offset=mcmc.loc[0, 'wave_offset_O{}'.format(order)], z=0, modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)
        models.append(copy.deepcopy(model))
        models_notel.append(copy.deepcopy(model_notel))
        
        # calculate veiling params and snrs        
        model_veiling = smart.Model(teff=mcmc.teff[0], logg=4., order=str(order), modelset='phoenix-aces-agss-cond-2011', instrument='nirspec')
        other_params[f'veiling_param_O{order}'] = mcmc.veiling[0] / np.median(model_veiling.flux)
        other_params[f'snr_O{order}'] = np.median(sci_specs[0].flux/sci_specs[0].noise)

        # calculate model dips and stds
        other_params[f'model_dip_O{order}'] = np.median(model_notel.flux) - min(model_notel.flux)
        other_params[f'model_std_O{order}'] = np.std(model_notel.flux)
        
    
    ##################################################
    ################# Writing Result #################
    ##################################################
    
    def get_result(mcmc, infos=infos, orders=orders):
        # teff, vsini, rv, airmass, pwv, veiling, lsf, noise, wave_offset1, wave_offset2
        result = {
            'HC2000':               infos['name'],
            'year':                 infos['date'][0],
            'month':                infos['date'][1],
            'day':                  infos['date'][2],
            'sci_frames':           infos['sci_frames'],
            'tel_frames':           infos['tel_frames'],
            'itime':                itime,
            'teff':                 mcmc.teff,
            'vsini':                mcmc.vsini,
            'rv':                   mcmc.rv, 
            'rv_helio':             mcmc.rv[0] + barycorr, 
            'airmass':              mcmc.airmass, 
            'pwv':                  mcmc.pwv, 
            'veiling':              mcmc.veiling,
            'lsf':                  mcmc.lsf, 
            'noise':                mcmc.noise
        }
        
        for order in orders:
            result[f'wave_offset_O{order}'] = mcmc[f'wave_offset_O{order}']
            
        for param in ['veiling_param', 'model_dip', 'model_std', 'snr']:
            for order in orders:
                result[f'{param}_O{order}'] = other_params[f'{param}_O{order}']
        
        ########## Write Parameters ##########
        with open(save_path + 'mcmc_params.txt', 'w') as file:
            for key, value in result.items():
                if isinstance(value, Iterable) and (not isinstance(value, str)):
                    file.write('{}: \t{}\n'.format(key, ", ".join(str(_) for _ in value)))
                else:
                    file.write('{}: \t{}\n'.format(key, value))
        
        return result
    
    result = get_result(mcmc)
    
    ##################################################
    #################### Fine Tune ###################
    ##################################################
    # Update spectrums for each order
    if finetune:
        print('Finetuning...')
        print(f'Date:\t20{"-".join(str(_).zfill(2) for _ in date)}')
        print(f'Object:\t{name}')
        print(f'Science  Frames:\t{sci_frames}')
        print(f'Telluric Frames:\t{tel_frames}')
        print()
        
        sci_specs_new = []
        for i in range(len(orders)):
            sci_spec = copy.deepcopy(sci_specs[i])
            # Update mask
            residual = sci_specs[i].flux - models[i].flux
            mask_finetune = np.where(abs(residual) > np.median(residual) + 3*np.std(residual))[0]
            
            # Mask out bad pixels after fine-tuning
            sci_spec.pixel   = np.delete(sci_spec.pixel, mask_finetune)
            sci_spec.wave   = np.delete(sci_spec.wave, mask_finetune)
            sci_spec.flux   = np.delete(sci_spec.flux, mask_finetune)
            sci_spec.noise  = np.delete(sci_spec.noise, mask_finetune)
            sci_specs_new.append(copy.deepcopy(sci_spec))

        # Update sci_specs
        sci_specs = copy.deepcopy(sci_specs_new)
        
        # Save sci_specs
        with open(save_path + 'sci_specs.pkl', 'wb') as file:
            pickle.dump(sci_specs, file)
        
        
        ##################################################
        ################### Re-run MCMC ##################
        ##################################################
        initial_state = np.array([
            np.random.uniform(priors[f'{param}'][0], priors[f'{param}'][1], size=nwalkers) for param in params_stripped
        ]).transpose()
        
        if finetune_mcmc:
            backend = emcee.backends.HDFBackend(save_path + 'sampler2.h5')
            backend.reset(nwalkers, nparams)
            if multiprocess:
                with Pool(64) as pool:
                    sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders, limits), moves=move, backend=backend, pool=pool)
                    sampler.run_mcmc(initial_state, steps, progress=True)
            else:
                sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders, limits), moves=move, backend=backend)
                sampler.run_mcmc(initial_state, steps, progress=True)
            
            print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
            print(sampler.acceptance_fraction)
            print()
        else:
            sampler = emcee.backends.HDFBackend(save_path + 'sampler2.h5')


        ##################################################
        ############### Re-Analyze Output ################
        ##################################################

        flat_samples = sampler.get_chain(discard=discard, flat=True)

        # mcmc[:, i] (2 by N) = [value, error]
        mcmc = np.empty((2, nparams))
        for i in range(nparams):
            mcmc[:, i] = np.array([np.median(flat_samples[:, i]), np.diff(np.percentile(flat_samples[:, i], [15.9, 84.1]))[0]/2])        
        
        mcmc = pd.DataFrame(mcmc, columns=params)

            
        ##################################################
        ############### Re-Construct Models ##############
        ##################################################
        models = []
        models_notel = []
        other_params = {}
        for i, order in enumerate(orders):
            model, model_notel = smart.makeModel(mcmc.teff[0], data=sci_specs[i], order=order, logg=4.0, vsini=mcmc.vsini[0], rv=mcmc.rv[0], airmass=mcmc.airmass[0], pwv=mcmc.pwv[0], veiling=mcmc.veiling[0], lsf=mcmc.lsf[0], wave_offset=mcmc.loc[0, 'wave_offset_O{}'.format(order)], z=0, modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)
            models.append(copy.deepcopy(model))
            models_notel.append(copy.deepcopy(model_notel))
            
            # calculate veiling params and snrs        
            model_veiling = smart.Model(teff=mcmc.teff[0], logg=4., order=str(order), modelset='phoenix-aces-agss-cond-2011', instrument='nirspec')
            other_params[f'veiling_param_O{order}'] = mcmc.veiling[0] / np.median(model_veiling.flux)
            other_params[f'snr_O{order}'] = np.median(sci_specs[0].flux/sci_specs[0].noise)

            # calculate model dips and stds
            other_params[f'model_dip_O{order}'] = np.median(model_notel.flux) - min(model_notel.flux)
            other_params[f'model_std_O{order}'] = np.std(model_notel.flux)


        ##################################################
        ################# Writing Result #################
        ##################################################
        result = get_result(mcmc)


    ##################################################
    ################## Create Plots ##################
    ##################################################

    ########## Walker Plot ##########
    fig, axes = plt.subplots(nrows=nparams, ncols=1, figsize=(10, 1.5*nparams), sharex=True)
    samples = sampler.get_chain()

    for i in range(nparams):
        ax = axes[i]
        ax.plot(samples[:, :, i], "C0", alpha=0.2)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(params[i])

    ax.set_xlabel("step number");
    plt.minorticks_on()
    fig.align_ylabels()
    plt.savefig(save_path + 'mcmc_walker.png', dpi=300, bbox_inches='tight')
    plt.close()

    ########## Corner Plot ##########
    fig = corner.corner(
        flat_samples, labels=params, truths=mcmc.loc[0].to_numpy(), quantiles=[0.16, 0.84]
    )
    plt.savefig(save_path + 'mcmc_corner.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    ########## Spectrum Plot ##########
    for i, order in enumerate(orders):
        
        fig, (ax1, ax2) = plot_spectrum(
            sci_spec=sci_specs[i], 
            result=result,
            save_path=save_path + f'spectrum_modeled_O{order}.pdf',
            mark_CO=False
        )
        plt.close()
    
    print('--------------------Finished--------------------')
    print('\n')
    
    return result


if __name__=='__main__':
    
    dates = [
        (15, 12, 24),
        (18, 2, 11)
    ]

    names = [
        '291_A', 
        337
    ]
    
    sci_frames = [
        [46, 47, 48, 49], 
        [47, 48, 49, 50]
    ]
    
    tel_frames = [
        [44, 45, 45, 44], 
        [51, 52, 52, 51]
    ]

    
    dim_check = [len(_) for _ in [dates, names, sci_frames, tel_frames]]
    if not all(_==dim_check[0] for _ in dim_check):
        sys.exit('Dimensions not agree! dates: {}, names: {}, sci_frames:{}, tel_frames:{}.'.format(*dim_check))
    
    priors = {
        'vsini':        (0, 40),
        'veiling':      (0, 1e5),
        'lsf':          (1, 10),
        'noise':        (1, 5),
        'wave_offset':  (-.2, .2)
    }
    
    for i in range(len(names)):
        infos = {
            'date':     dates[i],
            'name':     names[i],
            'sci_frames': sci_frames[i],
            'tel_frames': tel_frames[i],
        }
        
        result = model_nirspao(infos=infos, orders=[35], initial_mcmc=False, finetune=True, finetune_mcmc=False, multiprocess=True, steps=300, priors=priors)