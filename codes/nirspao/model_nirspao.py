# Median combine spectrums of each object

import os, sys, shutil
import pickle
import copy
os.environ["OMP_NUM_THREADS"] = "1" # Limit number of threads
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt 
import smart 
import emcee
import corner
import plotly.graph_objects as go
from itertools import repeat
from multiprocessing import Pool
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from collections.abc import Iterable
np.set_printoptions(threshold=sys.maxsize)


def plot_spectrum(sci_spec, model, model_notel, rv, wave_offset, save_path=None, mark_CO=True, show_figure=False):
    """Plot spectrum with model and CO lines in order 32 or 33.

    Parameters
    ----------
    sci_spec : smart.forward_model.classSpectrum.Spectrum
        science spectrum
    model_notel : smart.forward_model.classModel.Model
        model without telluric
    model : smart.forward_model.classModel.Model
        model with telluric
    rv : float
        radial velocity
    wave_offset : float
        wavelength offset
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
    if mark_CO:
        # Read CO lines
        co_lines = pd.read_csv('/home/l3wei/ONC/starrynight/codes/plot_spectrum/CO lines.csv')
        co_lines.intensity = np.log10(co_lines.intensity)
        if order==32:
            co_lines = co_lines[co_lines.intensity >= -25].reset_index(drop=True)
        co_lines['wavelength'] = 1/co_lines.frequency * 1e8
        c = 299792.458  # km/s
        beta = rv/c
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
            ax1.vlines(co_lines.wavelength + wave_offset, ymin=max(sci_spec.flux) + 0.02*np.median(sci_spec.flux), ymax=max(sci_spec.flux) + 0.12*np.median(sci_spec.flux), colors='k', alpha=co_lines.alpha, lw=1.2)
            ax1.text(min(co_lines.wavelength) + wave_offset - 10, max(sci_spec.flux) + 0.065*np.median(sci_spec.flux), 'CO', fontsize=12, horizontalalignment='center', verticalalignment='center')
            ax1.margins(x=0.03)
            ax2.margins(x=0.03)
        elif order==33:
            ax1.vlines(co_lines.wavelength + wave_offset, ymin=max(sci_spec.flux) + 0.02*np.median(sci_spec.flux), ymax=max(sci_spec.flux) + 0.07*np.median(sci_spec.flux), colors='k', alpha=co_lines.alpha, lw=1.2)
            ax1.text(min(co_lines.wavelength) + wave_offset - 15, max(sci_spec.flux) + 0.043*np.median(sci_spec.flux), 'CO', fontsize=12, horizontalalignment='center', verticalalignment='center')
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
    ax1.set_ylabel('Normalized Flux', fontsize=15)

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
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show_figure:
        plt.show()
    
    return fig, (ax1, ax2)


##################################################
############## Probability Function ##############
##################################################

def lnprior(theta, orders):
    
    teff, vsini, rv, airmass, pwv, veiling, lsf, noise = theta[:-len(orders)]
    wave_offsets = theta[-len(orders):]
    
    if  \
        2300    < teff              < 7000  \
    and 0       < vsini             < 100   \
    and -100    < rv                < 100   \
    and 1       < airmass           < 3     \
    and 0.5     < pwv               < 20    \
    and 0       < veiling           < 1e20  \
    and 1       < lsf               < 20    \
    and 1       < noise             < 50    \
    and -10     < all(wave_offsets) < 10:
        return 0.0
    else:
        return -np.inf


def lnlike(theta, sci_specs, orders):
    
    teff, vsini, rv, airmass, pwv, veiling, lsf, noise = theta[:-len(orders)]

    wave_offsets = theta[-len(orders):]
    
    sci_noises = np.array([])
    sci_fluxes = np.array([])
    model_fluxes = np.array([])

    for i, order in enumerate(orders):
        model = smart.makeModel(teff, logg=4.0, vsini=vsini, rv=rv, airmass=airmass, pwv=pwv, veiling=veiling, lsf=lsf, z=0, wave_offset=wave_offsets[i], order=str(order), data=sci_specs[i], modelset='phoenix-aces-agss-cond-2011')
        sci_noises = np.concatenate((sci_noises, sci_specs[i].noise * noise))
        sci_fluxes = np.concatenate((sci_fluxes, sci_specs[i].flux))
        model_fluxes = np.concatenate((model_fluxes, model.flux))

    sigma2 = sci_noises**2
    chi = -1/2 * np.sum( (sci_fluxes - model_fluxes)**2 / sigma2 + np.log(2*np.pi*sigma2) )
    if np.isnan(chi):
        return -np.inf
    else:
        return chi

    
def lnprob(theta, sci_specs, orders):
    lp = lnprior(theta, orders)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike(theta, sci_specs, orders)


##################################################
################## Model NIRSPAO #################
##################################################
def model_nirspao(infos, orders=[32, 33], initial_mcmc=True, finetune=True, finetune_mcmc=True, multiprocess=True, nwalkers=100, steps=500):
    """Fit teff and other params using emcee

    Parameters
    ----------
    infos : dict
        dictionary with keywords 'date' (tuple), 'name' (int or str), 'sci_frames' (list of int), 'tel_frames' (list of int)
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
    
    # Testing purposes
    # initial_mcmc = True
    # multiprocess=False
    # finetune=True
    # finetune_mcmc=True
    # orders=[32, 33]
    # nwalkers=100
    # steps=100

    # modify parameters
    params = ['teff', 'vsini', 'rv', 'airmass', 'pwv', 'veiling', 'lsf', 'noise']
    for order in orders:
        params += ['wave_offset_O{}'.format(order)]

    nparams = len(params)
    discard = steps - 100
    
    year = str(date[0]).zfill(2)
    month = str(date[1]).zfill(2)
    day = str(date[2]).zfill(2)
    
    month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    prefix = f'/home/l3wei/ONC/data/nirspao/20{year}{month_list[int(month) - 1]}{day}/reduced/'
    save_path = f'{prefix}mcmc_median/{name}_O{orders}_params/'
    
    if initial_mcmc:
        if os.path.exists(save_path): shutil.rmtree(save_path)
    
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    print(f'Date:\t20{"-".join(str(_).zfill(2) for _ in date)}')
    print(f'Object:\t{name}')
    print(f'Science  Frames:\t{sci_frames}')
    print(f'Telluric Frames:\t{tel_frames}')
    print()
    
    sys.stdout.flush()
    
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
                sci_spec = smart.Spectrum(name=f'{sci_name}_A', order=order, path=f'{prefix}extracted_binaries/{sci_name}/O{order}')
            elif name.endswith('B'):
                sci_spec = smart.Spectrum(name=f'{sci_name}_B', order=order, path=f'{prefix}extracted_binaries/{sci_name}/O{order}')
            else:
                sci_spec = smart.Spectrum(name=sci_name, order=order, path=f'{prefix}nsdrp_out/fits/all')
            
            sci_spec.pixel = np.arange(len(sci_spec.wave))
            sci_spec.snr = np.median(sci_spec.flux / sci_spec.noise)
            
            if os.path.exists(prefix + tel_name + '_defringe/O{}/'.format(order)):
                tel_name = tel_name + '_defringe'
            
            tel_spec = smart.Spectrum(name=f'{tel_name}_calibrated', order=order, path=f'{prefix}{tel_name}/O{order}/')
            
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
            
            sci_abba.append(copy.deepcopy(sci_spec))
            tel_abba.append(copy.deepcopy(tel_spec))

        # Normalize to the highest snr frame
        median_flux = ma.median(sci_abba[np.argmax([_.snr for _ in sci_abba])].flux)
        for spec in sci_abba:
            normalize_factor = median_flux / ma.median(spec.flux)
            spec.flux   *= normalize_factor
            spec.noise  *= normalize_factor

        # tel_spec = tel_abba[np.argmin([_.header['RMS'] for _ in tel_abba])]

        sci_spec = copy.deepcopy(sci_abba[np.argmin([_.header['RMS'] for _ in tel_abba])])
        sci_spec.flux = ma.average(ma.array([_.flux for _ in sci_abba]), weights=1/ma.array([_.noise for _ in sci_abba])**2, axis=0)
        sci_spec.noise = ma.sqrt(ma.std(ma.array([_.flux for _ in sci_abba]), axis=0)**2 + ma.sum(ma.array([_.noise for _ in sci_abba])**2, axis=0) / ma.sum(~ma.array([_.noise.mask for _ in sci_abba]), axis=0))
        # sci_spec.noise = ma.sqrt(ma.sum(ma.array([_.noise for _ in sci_abba])**2, axis=0) / ma.sum(~ma.array([_.noise.mask for _ in sci_abba]), axis=0))
        sci_spec.pixel.mask = sci_spec.flux.mask
        sci_spec.wave.mask = sci_spec.flux.mask

        sci_spec.pixel  = sci_spec.pixel.compressed()
        sci_spec.wave   = sci_spec.wave.compressed()
        sci_spec.flux   = sci_spec.flux.compressed()
        sci_spec.noise  = sci_spec.noise.compressed()

        # Special Case
        if date == (19, 1, 12) and order == 32:
            idx = sci_spec.wave < 23980
            sci_spec.pixel  = sci_spec.pixel[idx]
            sci_spec.flux   = sci_spec.flux [idx]
            sci_spec.noise  = sci_spec.noise[idx]
            sci_spec.wave   = sci_spec.wave [idx]

        barycorrs.append(smart.barycorr(sci_spec.header).value)
        itime = sci_spec.header['ITIME']
        
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
    
    # Modify parameters
    pos = [np.append([
        np.random.uniform(2300, 7000),  # Teff
        np.random.uniform(0, 40),       # vsini
        np.random.uniform(-100, 100),   # rv
        np.random.uniform(1, 3),        # airmass
        np.random.uniform(0.5, 20),     # pwv
        np.random.uniform(0, 1e5),      # veiling
        np.random.uniform(1, 10),       # lsf
        np.random.uniform(1, 5),        # noise
    ],
        list(np.reshape([[
        np.random.uniform(-5, 5)        # wave offset n
        ] for order in orders], -1))
    )
    for _ in range(nwalkers)]
    
    
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
                sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders), moves=move, backend=backend, pool=pool)
                sampler.run_mcmc(pos, steps, progress=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders), moves=move, backend=backend)
            sampler.run_mcmc(pos, steps, progress=True)
        
        print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
        print(sampler.acceptance_fraction)
        print()
    else:
        sampler = emcee.backends.HDFBackend(save_path + 'sampler1.h5')
    
    
    ##################################################
    ################# Analyze Output #################
    ##################################################

    flat_samples = sampler.get_chain(discard=discard, flat=True)

    # mcmc[:, i] (3 by N) = [value, lower, upper]
    mcmc = np.empty((3, nparams))
    for i in range(nparams):
        mcmc[:, i] = np.percentile(flat_samples[:, i], [50, 16, 84])
    
    mcmc = pd.DataFrame(mcmc, columns=params)
    
    # calculate veiling params
    veiling_params = []
    for order in orders:
        model_veiling = smart.Model(teff=mcmc.teff[0], logg=4., order=str(order), modelset='phoenix-aces-agss-cond-2011', instrument='nirspec')
        veiling_params.append(mcmc.veiling[0] / np.median(model_veiling.flux))
    
    
    ##################################################
    ################ Construct Models ################
    ##################################################
    
    models = []
    models_notel = []
    model_dips = []
    model_stds = []
    for i, order in enumerate(orders):
        model, model_notel = smart.makeModel(mcmc.teff[0], order=str(order), data=sci_specs[i], logg=4.0, vsini=mcmc.vsini[0], rv=mcmc.rv[0], airmass=mcmc.airmass[0], pwv=mcmc.pwv[0], veiling=mcmc.veiling[0], lsf=mcmc.lsf[0], wave_offset=mcmc.loc[0, 'wave_offset_O{}'.format(order)], z=0, modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)
        models.append(copy.deepcopy(model))
        models_notel.append(copy.deepcopy(model_notel))
        
        model_dips.append(np.median(model_notel.flux) - min(model_notel.flux))
        model_stds.append(np.std(model_notel.flux))
    
    
    ##################################################
    ################# Writing Result #################
    ##################################################
    
    # teff, vsini, rv, airmass, pwv, veiling, lsf, noise, wave_offset1, wave_offset2
    result = {
        'HC2000':               name,
        'year':                 date[0],
        'month':                date[1],
        'day':                  date[2],
        'sci_frames':           sci_frames,
        'tel_frames':           tel_frames,
        'itime':                itime,
        'teff':                 mcmc.teff,
        'vsini':                mcmc.vsini,
        'rv':                   mcmc.rv, 
        'rv_helio':             mcmc.rv[0] + barycorr, 
        'airmass':              mcmc.airmass, 
        'pwv':                  mcmc.pwv, 
        'veiling':              mcmc.veiling,
        'veiling_param_O32':    veiling_params[0],
        'veiling_param_O33':    veiling_params[1],
        'lsf':                  mcmc.lsf, 
        'noise':                mcmc.noise,
        'model_dip_O32':        model_dips[0],
        'model_std_O32':        model_stds[0],
        'model_dip_O33':        model_dips[1],
        'model_std_O33':        model_stds[1],
        'wave_offset_O32':      mcmc.wave_offset_O32, 
        'wave_offset_O33':      mcmc.wave_offset_O33, 
        'snr_O32':              np.median(sci_specs[0].flux/sci_specs[0].noise), 
        'snr_O33':              np.median(sci_specs[1].flux/sci_specs[1].noise)
    }
    
    ########## Write Parameters ##########
    with open(save_path + 'mcmc_params.txt', 'w') as file:
        for key, value in result.items():
            if isinstance(value, Iterable) and (not isinstance(value, str)):
                file.write('{}: \t{}\n'.format(key, ", ".join(str(_) for _ in value)))
            else:
                file.write('{}: \t{}\n'.format(key, value))
    
    
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
        for i, order in enumerate(orders):
            sci_spec = copy.deepcopy(sci_specs[i])
            # Update mask
            residual = sci_specs[i].flux - models[i].flux
            mask_finetune = np.where(abs(residual) > np.median(residual) + 3*np.std(residual))[0]
            
            # Mask out bad pixels after fine-tuning
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
        if finetune_mcmc:
            backend = emcee.backends.HDFBackend(save_path + 'sampler2.h5')
            backend.reset(nwalkers, nparams)
            if multiprocess:
                with Pool(64) as pool:
                    sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders), moves=move, backend=backend, pool=pool)
                    sampler.run_mcmc(pos, steps, progress=True)
            else:
                sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders), moves=move, backend=backend)
                sampler.run_mcmc(pos, steps, progress=True)
            
            print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
            print(sampler.acceptance_fraction)
            print()
        else:
            sampler = emcee.backends.HDFBackend(save_path + 'sampler2.h5')


        ##################################################
        ############### Re-Analyze Output ################
        ##################################################

        flat_samples = sampler.get_chain(discard=discard,  flat=True)
        
        # mcmc[:, i] (3 by N) = [value, lower, upper]
        mcmc = np.empty((3, nparams))
        for i in range(nparams):
            mcmc[:, i] = np.percentile(flat_samples[:, i], [50, 16, 84])
        
        mcmc = pd.DataFrame(mcmc, columns=params)
        
        # calculate veiling params
        veiling_params = []
        for order in orders:
            model_veiling = smart.Model(teff=mcmc.teff[0], logg=4., order=str(order), modelset='phoenix-aces-agss-cond-2011', instrument='nirspec')
            veiling_params.append(mcmc.veiling[0] / np.median(model_veiling.flux))

            
        ##################################################
        ############### Re-Construct Models ##############
        ##################################################
        models = []
        models_notel = []
        model_dips = []
        model_stds = []
        for i, order in enumerate(orders):
            model, model_notel = smart.makeModel(mcmc.teff[0], order=str(order), data=sci_specs[i], logg=4.0, vsini=mcmc.vsini[0], rv=mcmc.rv[0], airmass=mcmc.airmass[0], pwv=mcmc.pwv[0], veiling=mcmc.veiling[0], lsf=mcmc.lsf[0], wave_offset=mcmc.loc[0, 'wave_offset_O{}'.format(order)], z=0, modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)
            models.append(copy.deepcopy(model))
            models_notel.append(copy.deepcopy(model_notel))
            
            model_dips.append(np.median(model_notel.flux) - min(model_notel.flux))
            model_stds.append(np.std(model_notel.flux))


    ##################################################
    ################# Writing Result #################
    ##################################################
    # teff, vsini, rv, airmass, pwv, veiling, lsf, noise, wave_offset1, wave_offset2
    result = {
        'HC2000':               name,
        'year':                 date[0],
        'month':                date[1],
        'day':                  date[2],
        'sci_frames':           sci_frames,
        'tel_frames':           tel_frames,
        'itime':                itime,
        'teff':                 mcmc.teff,
        'vsini':                mcmc.vsini,
        'rv':                   mcmc.rv, 
        'rv_helio':             mcmc.rv[0] + barycorr, 
        'airmass':              mcmc.airmass, 
        'pwv':                  mcmc.pwv, 
        'veiling':              mcmc.veiling, 
        'veiling_param_O32':    veiling_params[0],
        'veiling_param_O33':    veiling_params[1],
        'lsf':                  mcmc.lsf, 
        'noise':                mcmc.noise,
        'model_dip_O32':        model_dips[0],
        'model_std_O32':        model_stds[0],
        'model_dip_O33':        model_dips[1],
        'model_std_O33':        model_stds[1], 
        'wave_offset_O32':      mcmc.wave_offset_O32, 
        'wave_offset_O33':      mcmc.wave_offset_O33, 
        'snr_O32':              np.median(sci_specs[0].flux/sci_specs[0].noise), 
        'snr_O33':              np.median(sci_specs[1].flux/sci_specs[1].noise)
    }
    
    ########## Write Parameters ##########
    with open(save_path + 'MCMC_Params.txt', 'w') as file:
        for key, value in result.items():
            if isinstance(value, Iterable) and (not isinstance(value, str)):
                file.write('{}: \t{}\n'.format(key, ", ".join(str(_) for _ in value)))
            else:
                file.write('{}: \t{}\n'.format(key, value))


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
            model_notel=models_notel[i], 
            model=models[i], 
            rv=mcmc.rv[0], 
            wave_offset=mcmc.loc[0, f'wave_offset_O{order}'], 
            save_path=save_path + f'spectrum_modeled_O{order}.pdf'
        )
        plt.close()
    
    print('--------------------Finished--------------------')
    print('\n')
    
    return result


if __name__=='__main__':
    
    test = False
    if not test:
        skip = 1
        Multiprocess=True
        # all data:
        dates = [
            *list(repeat((15, 12, 23), 4)),
            *list(repeat((15, 12, 24), 8)),
            *list(repeat((16, 12, 14), 4)),
            *list(repeat((18, 2, 11), 7)),
            *list(repeat((18, 2, 12), 5)),
            *list(repeat((18, 2, 13), 6)),
            *list(repeat((19, 1, 12), 5)),
            *list(repeat((19, 1, 13), 6)),
            *list(repeat((19, 1, 16), 6)),
            *list(repeat((19, 1, 17), 5)),
            *list(repeat((20, 1, 18), 2)),
            *list(repeat((20, 1, 19), 3)),
            *list(repeat((20, 1, 20), 6)),
            *list(repeat((20, 1, 21), 7)),
            *list(repeat((21, 2, 1), 2)),
            *list(repeat((21, 10, 20), 4)),
            *list(repeat((22, 1, 18), 6)),
            *list(repeat((22, 1, 19), 5)),
            *list(repeat((22, 1, 20), 7))
        ]

        names = [
            # 2015-12-23
            322, 296, 259, 213,
            # 2015-12-24
            '306_A', '306_B', '291_A', '291_B', 252, 250, 244, 261,
            # 2016-12-14
            248, 223, 219, 324,
            # 2018-2-11
            295, 313, 332, 331, 337, 375, 388,
            # 2018-2-12
            425, 713, 408, 410, 436,
            # 2018-2-13
            '354_B2', '354_B3', '354_B1', '354_B4', 442, 344,
            # 2019-1-12
            '522_A', '522_B', 145, 202, 188,
            # 2019-1-13
            302, 275, 245, 258, 220, 344,
            # 2019-1-16
            370, 389, 386, 398, 413, 253,
            # 2019-1-17
            288, 420, 412, 282, 217,
            # 2020-1-18
            217, 229,
            # 2020-1-19
            228, 224, 135,
            # 2020-1-20
            440, 450, 277, 204, 229, 214,
            # 2020-1-21
            215, 240, 546, 504, 703, 431, 229,
            # 2021-2-1
            484, 476,
            # 2021-10-20
            546, 217, 277, 435,
            # 2022-1-18
            457, 479, 490, 478, 456, 170, 
            # 2022-1-19
            453, 438, 530, 287, 171, 
            # 2022-1-20
            238, 266, 247, 165, 172, 177, 163
        ]
        
        sci_frames = [
            # 2015-12-23
            [75, 76, 77, 78], [81, 82, 83, 84], [87, 88, 89, 90], [91, 92, 93, 94],
            # 2015-12-24
            [40, 41, 42, 43], [40, 41, 42, 43], [46, 47, 48, 49], [46, 47, 48, 49], [52, 53, 54, 55], [56, 57, 58, 59], [62, 63, 64, 65], [66, 67, 68, 69],
            # 2016-12-14
            [40, 41, 42, 43], [46, 47, 48, 49], [50, 51, 52, 53], [56, 57, 58], 
            # 2018-2-11
            [26, 27, 29, 30], [33, 34, 35, 36], [37, 38, 39, 40], [43, 44, 45, 46], [47, 48, 49, 50], [53, 54, 55, 56], [57, 58, 59, 60],
            # 2018-2-12
            [47, 48, 49, 50], [54, 55, 56, 57], [58, 59, 60, 61], [64, 65, 66, 67], [70, 71, 72, 73],
            # 2018-2-13
            [25, 26, 27, 28], [25, 26, 27, 28], [32, 33, 34, 35], [36, 37, 38, 39], [42, 43, 44, 45], [48, 49, 50, 51], 
            # 2019-1-12
            [36, 37, 38, 39], [36, 37, 38, 39], [42, 43, 44, 45], [46, 47, 48, 49], [52, 53, 54, 55], 
            # 2019-1-13
            [41, 42, 43, 44], [47, 48], [49, 50, 51, 52], [55, 56, 57, 58], [59, 60, 61, 63], [66, 67, 68, 69], 
            # 2019-1-16
            [83, 84, 85, 86], [89, 90, 91, 92], [93, 94, 95, 96], [99, 100, 101, 102], [113, 114, 115, 116], [119, 120, 121, 122], 
            # 2019-1-17
            [38, 39, 40, 41], [44, 45, 46, 47], [48, 49, 50, 51], [54, 55, 56, 57], [58, 59, 60, 61], 
            # 2020-1-18
            [27, 28, 29, 30], [33, 34, 35, 36],
            # 2020-1-19
            [31, 32, 33, 34], [37, 38, 39, 40], [41, 42, 43, 44], 
            # 2020-1-20
            [32, 33, 34, 35], [36, 37, 38, 39], [42, 43, 44, 45], [46, 47, 48, 49], [50, 51, 52, 53], [54, 55, 56, 57],
            # 2020-1-21
            [29, 30, 31, 32], [33, 34, 35, 36], [39, 40, 41, 42], [43, 44, 45, 46], [47, 48, 49, 50], [53, 54, 55, 56], [58, 59, 60], 
            # 2021-2-1
            [27, 28, 29, 30], [31, 32, 33, 34, 35, 36], 
            # 2021-10-20
            [7, 8, 9, 10], [13, 14, 15, 16], [17, 18], [22, 23, 24],
            # 2022-1-18
            [24, 25, 26, 27], [31, 32, 33, 34], [35, 36, 37, 38], [41, 42, 43, 44], [45, 46, 47, 49], [52, 53, 54, 55], 
            # 2022-1-19
            [26, 27, 28, 29], [32, 33, 34, 35], [36, 37, 38, 39], [42, 43, 44, 45], [46, 47, 48, 49], 
            # 2022-1-20
            [23, 25, 27, 26], [30, 31, 33, 32], [34, 35, 36, 37], [40, 41, 42, 43], [44, 45, 46, 47], [51, 52, 53, 55], [56, 57, 58]
        ]
        
        tel_frames = [
            # 2015-12-23
            [79, 80, 80, 79], [79, 80, 80, 79], [95, 96, 96, 95], [95, 96, 96, 95],
            # 2015-12-24
            [44, 45, 45, 44], [44, 45, 45, 44], [44, 45, 45, 44], [44, 45, 45, 44], [50, 51, 51, 50], [60, 61, 61, 60], [60, 61, 61, 60], [44, 45, 45, 44], 
            # 2016-12-14
            [54, 55, 55, 54], [44, 45, 45, 44], [44, 45, 45, 44], [54, 55, 55], 
            # 2018-2-11
            [41, 42, 42, 41], [41, 42, 42, 41], [41, 42, 42, 41], [41, 42, 42, 41], [51, 52, 52, 51], [51, 52, 52, 51], [61, 62, 62, 61],
            # 2018-2-12
            [51, 52, 52, 51], [51, 52, 52, 51], [51, 52, 52, 51], [62, 63, 63, 62], [74, 75, 75, 74], 
            # 2018-2-13
            [40, 41, 41, 40], [40, 41, 41, 40], [40, 41, 41, 40], [40, 41, 41, 40], [46, 47, 47, 46], [52, 53, 53, 52], 
            # 2019-1-12
            [40, 41, 41, 40], [40, 41, 41, 40], [40, 41, 41, 40], [50, 51, 51, 50], [50, 51, 51, 50], 
            # 2019-1-13
            [39, 40, 40, 39], [45, 46], [53, 54, 54, 53], [53, 54, 54, 53], [64, 65, 65, 64], [64, 65, 65, 64], 
            # 2019-1-16
            [87, 88, 88, 87], [87, 88, 88, 87], [97, 98, 98, 97], [97, 98, 98, 97], [117, 118, 118, 117], [117, 118, 118, 117], 
            # 2019-1-17
            [42, 43, 43, 42], [42, 43, 43, 42], [62, 63, 63, 62], [52, 53, 53, 52], [62, 63, 63, 62], 
            # 2020-1-18
            [31, 32, 32, 31], [31, 32, 32, 31],
            # 2020-1-19
            [35, 36, 36, 35], [35, 36, 36, 35], [35, 36, 36, 35],
            # 2020-1-20
            [40, 41, 41, 40], [40, 41, 41, 40], [40, 41, 41, 40], [40, 41, 41, 40], [58, 59, 59, 58], [58, 59, 59, 58],
            # 2020-1-21
            [61, 62, 62, 61], [37, 38, 38, 37], [51, 52, 52, 51], [51, 52, 52, 51], [51, 52, 52, 51], [37, 38, 38, 37], [61, 62, 62], 
            # 2021-2-1
            [37, 38, 38, 37], [37, 38, 38, 37, 37, 38],
            # 2021-10-20
            [11, 12, 12, 11], [25, 26, 26, 25], [25, 26], [25, 26, 26],
            # 2022-1-18
            [28, 29, 29, 28], [28, 29, 29, 28], [39, 40, 40, 39], [39, 40, 40, 39], [50, 51, 51, 50], [50, 51, 51, 50], 
            # 2022-1-19
            [30, 31, 31, 30], [30, 31, 31, 30], [30, 31, 31, 30], [30, 31, 31, 30], [30, 31, 31, 30], 
            # 2022-1-20
            [28, 29, 29, 28], [28, 29, 29, 28], [49, 50, 50, 49], [49, 50, 50, 49], [49, 50, 50, 49], [28, 29, 29, 28], [28, 29, 29]
        ]
        
        if skip > 0:
            dates = dates[skip:]
            names = names[skip:]
            sci_frames = sci_frames[skip:]
            tel_frames = tel_frames[skip:]
    
    else:
        Multiprocess=True
        dates = [(15, 12, 23)]
        names = [322]
        sci_frames = [[75, 76, 77, 78]]
        tel_frames = [[79, 80, 80, 79]]

    
    dim_check = [len(_) for _ in [dates, names, sci_frames, tel_frames]]
    if not all(_==dim_check[0] for _ in dim_check):
        sys.exit('Dimensions not agree! dates: {}, names: {}, sci_frames:{}, tel_frames:{}.'.format(*dim_check))
    
    for i in range(len(names)):
        infos = {
            'date':     dates[i],
            'name':     names[i],
            'sci_frames': sci_frames[i],
            'tel_frames': tel_frames[i],
        }
        
        result = model_nirspao(infos=infos, initial_mcmc=True, finetune=True, finetune_mcmc=True, multiprocess=Multiprocess, steps=500)