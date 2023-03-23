# Median combine spectrums of each object

import os, sys, shutil
import pickle
import copy
os.environ["OMP_NUM_THREADS"] = "1" # Limit number of threads
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import smart 
import emcee
import corner
from functools import reduce
from multiprocessing import Pool
from scipy import interpolate
from collections.abc import Iterable
from itertools import repeat
np.set_printoptions(threshold=sys.maxsize)

def main(infos, orders=[34], Multiprocess=True, MCMC=True, Finetune=True):
    
    '''
    Fit teff and other params using mcmc.
    infos: dictionary with keywords "date", "name", "sci_frames", "tel_frames"
    '''
    date = infos['date']
    name = infos['name']
    sci_frames = infos['sci_frames']
    tel_frames = infos['tel_frames']
    
    # Testing purposes
    # MCMC = True          
    # Multiprocess=False 
    # Finetune=True

    # name = '145'
    # date = (19, 1, 12)
    # sci_frames = [42, 43, 44, 45]
    # tel_frames = [40, 41, 41, 40]
    # orders = [32, 33]
    
    # Modify Parameters
    params = ['logg', 'noise']
    for order in orders:
        params = params + ['wave_offset_O{}'.format(order), 'flux_offset_O{}'.format(order)]
    
    nparams, nwalkers, step = len(params), 100, 500
    discard = step - 100
    
    year = str(date[0]).zfill(2)
    month = str(date[1]).zfill(2)
    day = str(date[2]).zfill(2)
    
    month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    common_prefix = '/home/l3wei/ONC/Data/20{}{}{}/reduced/'.format(year, month_list[int(month) - 1], day)
    save_path = '{}mcmc_median/{}_O{}_logg/'.format(common_prefix, name, orders)
    
    if MCMC:
        if os.path.exists(save_path): shutil.rmtree(save_path)
    
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    print('\n\n')
    print('Date:\t20{}'.format("-".join(str(_).zfill(2) for _ in date)))
    print('Object:\t{}'.format(name))
    print('Science  Frames:\t{}'.format(sci_frames))
    print('Telluric Frames:\t{}'.format(tel_frames))
    print('\n\n')
    
    sys.stdout.flush()
    
    ##################################################
    ############### Construct Spectrums ##############
    ##################################################
    # sci_specs = [order 32 median combined sci_spec, order 33 median combined sci_spec]
    sci_specs = []
    barycorrs = []
    for order in orders:
        
        sci_abba = []
        sci_names = []
        tel_names = []
        
        
        ##################################################
        ####### Construct Spectrums for each order #######
        ##################################################
        for sci_frame, tel_frame in zip(sci_frames, tel_frames):        
            
            if int(year) > 18:
                # For data after 2018, sci_names = [nspec200118_0027, ...]
                sci_name = 'nspec' + year + month + day + '_' + str(sci_frame).zfill(4)
                tel_name = 'nspec' + year + month + day + '_' + str(tel_frame).zfill(4)
                pixel_start = 20
                pixel_end = -50
            
            else:
                # For data prior to 2018 (2018 included)
                sci_name = month_list[int(month) - 1] + day + 's' + str(sci_frame).zfill(4)
                tel_name = month_list[int(month) - 1] + day + 's' + str(tel_frame).zfill(4)
                pixel_start = 10
                pixel_end = -30
            
            sci_names.append(sci_name)
            tel_names.append(tel_name)
            
            if name.endswith('A'):
                sci_spec = smart.Spectrum(name=sci_name + '_A', order=order, path=common_prefix + 'extracted_binaries/' + sci_name + '/O' + str(order))
            elif name.endswith('B'):
                sci_spec = smart.Spectrum(name=sci_name + '_B', order=order, path=common_prefix + 'extracted_binaries/' + sci_name + '/O' + str(order))
            else:
                sci_spec = smart.Spectrum(name=sci_name, order=order, path=common_prefix + 'nsdrp_out/fits/all')
            
            # if os.path.exists(common_prefix + tel_name + '_defringe/O{}/'.format(order)):
            #     tel_name = tel_name + '_defringe'
            
            tel_spec = smart.Spectrum(name=tel_name + '_calibrated', order=order, path=common_prefix + tel_name + '/O{}/'.format(order))
            
            # Update the wavelength solution
            sci_spec.updateWaveSol(tel_spec)
            
            pixel = np.arange(len(sci_spec.wave))
            pixel = pixel.astype(float)
            
            # Plot original spectrums:
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.plot(sci_spec.wave, sci_spec.flux, alpha=0.7, lw=0.7, label='Original Spectrum')
            
            # Automatically mask out edge & flux < 0
            auto_mask = np.where(sci_spec.flux < 0)[0]
            mask_1 = reduce(np.union1d, (auto_mask, np.arange(0, pixel_start), np.arange(len(sci_spec.wave) + pixel_end, len(sci_spec.wave))))
            mask_1 = mask_1.astype(int)
            pixel           [mask_1] = np.nan
            sci_spec.wave   [mask_1] = np.nan
            sci_spec.flux   [mask_1] = np.nan
            sci_spec.noise  [mask_1] = np.nan
            
            # Mask flux > median + 3 sigma
            median_flux = np.nanmedian(sci_spec.flux)
            upper_bound = median_flux + 3.*np.nanstd(sci_spec.flux - median_flux)
            mask_2 = pixel[np.where(sci_spec.flux > upper_bound)]
            mask_2 = mask_2.astype(int)
            pixel           [mask_2] = np.nan
            sci_spec.wave   [mask_2] = np.nan
            sci_spec.flux   [mask_2] = np.nan
            sci_spec.noise  [mask_2] = np.nan
            
            # Mask isolated bad pixels
            median_flux = np.nanmedian(sci_spec.flux)
            lower_bound = median_flux - 3.5*np.nanstd(sci_spec.flux - median_flux)
            lowest_bound = median_flux - 5.*np.nanstd(sci_spec.flux - median_flux)
            mask_3 = np.array([i for i in np.arange(1, len(sci_spec.wave)-1) if (sci_spec.flux[i] < lowest_bound) and (sci_spec.flux[i-1] >= lower_bound) and (sci_spec.flux[i+1] >= lower_bound)], dtype=int)
            pixel           [mask_3] = np.nan
            sci_spec.wave   [mask_3] = np.nan
            sci_spec.flux   [mask_3] = np.nan
            sci_spec.noise  [mask_3] = np.nan
            
            # Plot masked spectrum
            ax.axhline(y=upper_bound, linestyle='--', label='Upper bound')
            ax.axhline(y=lower_bound, linestyle='--', label='Lower bound')
            ax.axhline(y=lowest_bound, linestyle='--', label='Lowest bound')
            ax.plot(sci_spec.wave, sci_spec.flux, color='C3', alpha=0.7, lw=0.7, label='Masked Spectrum')
            ax.minorticks_on()
            ax.legend()
            ax.set_xlabel(r'$\lambda$ ($\AA$)', fontsize=15)
            ax.set_ylabel('Flux (counts/s)', fontsize=15)
            plt.savefig(save_path + 'O{}_{}_Spectrum.png'.format(order, sci_frame), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Special Case
            if date == (19, 1, 12) and order == 32:
                sci_spec.flux = sci_spec.flux[sci_spec.wave < 23980]
                sci_spec.noise = sci_spec.noise[sci_spec.wave < 23980]
                sci_spec.wave = sci_spec.wave[sci_spec.wave < 23980]
            
            # Normalize
            sci_spec.noise = sci_spec.noise / np.nanmedian(sci_spec.flux)
            sci_spec.flux  = sci_spec.flux  / np.nanmedian(sci_spec.flux)
            
            
            sci_abba.append(sci_spec)  # Type: smart.Spectrum
            barycorrs.append(smart.barycorr(sci_spec.header).value)
        
        
        ##################################################
        ################# Median Combine #################
        ##################################################
        # sci spectrum flux table: 
        # flux_new = [
        #   flux 1
        #   flux 2
        #   ...
        # ]
        
        # interpolate spectrum
        wave_max = np.nanmin([np.nanmax(i.wave) for i in sci_abba])
        wave_min = np.nanmax([np.nanmin(i.wave) for i in sci_abba])
        resolution = 0.02
        
        wave_new = np.arange(wave_min, wave_max, resolution)
        flux_new = np.zeros((np.size(sci_frames), np.size(wave_new)))
        noise_new = np.zeros(np.shape(flux_new))
        
        for i in range(np.size(sci_frames)):
            # interpolate flux and noise function
            f_flux  = interpolate.interp1d(sci_abba[i].wave, sci_abba[i].flux)
            f_noise = interpolate.interp1d(sci_abba[i].wave, sci_abba[i].noise)
            flux_new[i, :] = f_flux(wave_new)
            noise_new[i, :] = f_noise(wave_new)
        
        # Median among all the frames     
        flux_med  = np.nanmedian(flux_new, axis=0)
        noise_med = np.nanmedian(noise_new, axis=0)
        
        sci_spec.wave = wave_new
        sci_spec.flux = flux_med
        sci_spec.noise = noise_med
        
        sci_specs.append(sci_spec)
        
        
        fig, ax = plt.subplots(figsize=(16, 6))
        # original interpolated spectrum
        for i in range(np.size(sci_frames)):
            ax.plot(wave_new, flux_new[i, :], color='C0', alpha=0.5, lw=0.5)
        # median combined spectrum
        ax.plot(sci_spec.wave, sci_spec.flux, 'C3', alpha=1, lw=0.5)
        ax.set_xlabel('$\lambda$ (Angstrom)', fontsize=15)
        ax.set_ylabel('Flux (count/s)', fontsize=15)
        ax.minorticks_on()
        plt.savefig(save_path + 'Spectrum_O{}.png'.format(order), dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(16, 6))
        for i in range(np.size(sci_frames)):
            ax.plot(wave_new, noise_new[i, :], color='C0', alpha=0.5, lw=0.5)
        ax.plot(sci_spec.wave, sci_spec.noise, 'C3', alpha=1, lw=0.5)
        ax.set_xlabel('$\lambda$ (Angstrom)', fontsize=15)
        ax.set_ylabel('Flux (count/s)', fontsize=15)
        ax.minorticks_on()
        plt.savefig(save_path + 'Noise_O{}.png'.format(order), dpi=300, bbox_inches='tight')
        plt.close()

    barycorr = np.median(barycorrs)
    
    
    ##################################################
    ################ Read MCMC Params ################
    ##################################################
    
    other_params = []
    others_path = common_prefix + 'mcmc_median/' + name + '_O[32, 33]_params/MCMC_Params.txt'
    
    with open(others_path, 'r') as file:
        lines = file.readlines()
    
    # teff, vsini, rv, airmass, pwv, veiling, lsf
    for line in lines:
        if line.startswith('teff:'):
            other_params.append(float(line.strip('teff: \t\n').split(', ')[0]))
        elif line.startswith('vsini:'):
            other_params.append(float(line.strip('vsini: \t\n').split(', ')[0]))
        elif line.startswith('rv:'):
            other_params.append(float(line.strip('rv: \t\n').split(', ')[0]))
        elif line.startswith('airmass:'):
            other_params.append(float(line.strip('airmass: \t\n').split(', ')[0]))
        elif line.startswith('pwv:'):
            other_params.append(float(line.strip('pwv: \t\n').split(', ')[0]))
        elif line.startswith('veiling:'):
            other_params.append(float(line.strip('veiling: \t\n').split(', ')[0]))
        elif line.startswith('lsf:'):
            other_params.append(float(line.strip('lsf: \t\n').split(', ')[0]))
    
    teff, vsini, rv, airmass, pwv, veiling, lsf = other_params
            
    ##################################################
    ################## MCMC Fitting ##################
    ##################################################
    
    pos = [np.append([
        np.random.uniform(3.0, 5.0),    # logg
        np.random.uniform(1, 5),        # noise
    ],
        list(np.reshape([[
        np.random.uniform(-5, 5),       # wave offset n
        np.random.uniform(-10, 10)      # flux offset n
        ] for j in range(np.size(orders))], -1))
    )
    for i in range(nwalkers)]
    
    
    move = [emcee.moves.KDEMove()]
    
    if MCMC:
        
        backend = emcee.backends.HDFBackend(save_path + 'sampler.h5')
        backend.reset(nwalkers, nparams)
        
        if Multiprocess:
            with Pool(32) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders, other_params), moves=move, backend=backend, pool=pool)
                sampler.run_mcmc(pos, step, progress=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders, other_params), moves=move, backend=backend)
            sampler.run_mcmc(pos, step, progress=True)
        
        print('\n\n')
        print(sampler.acceptance_fraction)
        print('\n\n')
        
    else:
        sampler = emcee.backends.HDFBackend(save_path + 'sampler.h5')
    
    
    
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
    model_veiling = smart.Model(teff=teff, logg=mcmc.logg[0], order=str(34), modelset='phoenix-aces-agss-cond-2011', instrument='nirspec')
    veiling_param = veiling / np.median(model_veiling.flux)
    
    
    ##################################################
    ################ Construct Models ################
    ##################################################
    
    models = []
    models_notel = []
    model_dips = []
    model_stds = []
    for i, order in enumerate(orders):
        model, model_notel = smart.makeModel(teff, order=str(order), data=sci_specs[i], logg=mcmc.logg[0], vsini=vsini, rv=rv, airmass=airmass, pwv=pwv, veiling=veiling, lsf=lsf, wave_offset=mcmc.loc[0, 'wave_offset_O{}'.format(order)], flux_offset=mcmc.loc[0, 'flux_offset_O{}'.format(order)], z=0, modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)
        models.append(model)
        models_notel.append(model_notel)
        
        model_dips.append(np.median(model_notel.flux) - min(model_notel.flux))
        model_stds.append(np.std(model_notel.flux))
    
    
    ##################################################
    ################# Writing Result #################
    ##################################################
    
    # teff, vsini, rv, airmass, pwv, veiling, lsf, noise, wave_offset1, flux_offset1, wave_offset2, flux_offset2
    result = {
        'HC2000':               name,
        'year':                 date[0],
        'month':                date[1],
        'day':                  date[2],
        'sci_frames':           sci_frames,
        'tel_frames':           tel_frames,
        'logg':                 mcmc.logg,
        'noise':                mcmc.noise,
        'model_dip_O34':        model_dips[0],
        'model_std_O34':        model_stds[0],
        'veiling_param_O34':    veiling_param,
        'wave_offset_O34':      mcmc.wave_offset_O34, 
        'flux_offset_O34':      mcmc.flux_offset_O34, 
        'snr_O34':              np.median(sci_specs[0].flux/sci_specs[0].noise),
    }
    
    ########## Write Parameters ##########
    with open(save_path + 'MCMC_Params.txt', 'w') as file:
        for key, value in zip(result.keys(), result.values()):
            if isinstance(value, Iterable) and (not isinstance(value, str)):
                file.write('{}: \t{}\n'.format(key, ", ".join(str(_) for _ in value)))
            else:
                file.write('{}: \t{}\n'.format(key, value))
    
    
    ##################################################
    #################### Fine Tune ###################
    ##################################################
    # Update spectrums for each order
    sci_specs_new = []
    for i, order in enumerate(orders):
        sci_spec = copy.deepcopy(sci_specs[i])
        pixel = np.arange(len(sci_spec.flux))
        pixel = pixel.astype(float)
        # Update mask
        residue = sci_specs[i].flux - models[i].flux
        mask_finetune = pixel[np.where(abs(residue) > np.nanmedian(residue) + 3*np.nanstd(residue))]
        mask_finetune = mask_finetune.astype(int)
        
        # Mask out bad pixels after fine-tuning
        pixel           = np.delete(pixel, mask_finetune)
        sci_spec.wave   = np.delete(sci_spec.wave, mask_finetune)
        sci_spec.flux   = np.delete(sci_spec.flux, mask_finetune)
        sci_spec.noise  = np.delete(sci_spec.noise, mask_finetune)
        sci_specs_new.append(sci_spec)

    # Update sci_specs
    sci_specs = sci_specs_new
    
    # Save sci_specs
    with open(save_path + 'sci_specs.pkl', 'wb') as file:
        pickle.dump(sci_specs, file)
    
    
    ##################################################
    ################### Re-run MCMC ##################
    ##################################################
    
    if Finetune:
        backend = emcee.backends.HDFBackend(save_path + 'sampler.h5')
        backend.reset(nwalkers, nparams)
        
        print('\n\n')
        print('Finetuning......')
        print('Date:\t20{}'.format("-".join(str(_).zfill(2) for _ in date)))
        print('Object:\t{}'.format(name))
        print('Science  Frames:\t%s' %str(sci_frames))
        print('Telluric Frames:\t%s' %str(tel_frames))
        print('\n\n')
        
        if Multiprocess:
            with Pool(32) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders, other_params), moves=move, backend=backend, pool=pool)
                sampler.run_mcmc(pos, step, progress=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders, other_params), moves=move, backend=backend)
            sampler.run_mcmc(pos, step, progress=True)
        
        print('\n\n')
        print(sampler.acceptance_fraction)
        print('\n\n')
        print('---------------------------------------------')
        print('\n\n')
        
    else:
        sampler = emcee.backends.HDFBackend(save_path + 'sampler.h5')


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
    model_veiling = smart.Model(teff=teff, logg=mcmc.logg[0], order=str(34), modelset='phoenix-aces-agss-cond-2011', instrument='nirspec')
    veiling_param = veiling / np.median(model_veiling.flux)
    
    
    ##################################################
    ############### Re-Construct Models ##############
    ##################################################
    
    models = []
    models_notel = []
    model_dips = []
    model_stds = []
    for i, order in enumerate(orders):
        model, model_notel = smart.makeModel(teff, order=str(order), data=sci_specs[i], logg=mcmc.logg[0], vsini=vsini, rv=rv, airmass=airmass, pwv=pwv, veiling=veiling, lsf=lsf, wave_offset=mcmc.loc[0, 'wave_offset_O{}'.format(order)], flux_offset=mcmc.loc[0, 'flux_offset_O{}'.format(order)], z=0, modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)
        models.append(model)
        models_notel.append(model_notel)
        
        model_dips.append(np.median(model_notel.flux) - min(model_notel.flux))
        model_stds.append(np.std(model_notel.flux))


    ##################################################
    ################## Create Plots ##################
    ##################################################

    ########## Walker Plot ##########
    fig, axes = plt.subplots(nrows=nparams, ncols=1, figsize=(10, 18), sharex=True)
    samples = sampler.get_chain()

    for i in range(nparams):
        ax = axes[i]
        ax.plot(samples[:, :, i], "C0", alpha=0.2)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(params[i])

    ax.set_xlabel("step number");
    plt.minorticks_on()
    fig.align_ylabels()
    plt.savefig(save_path + 'MCMC_Walker.png', dpi=300, bbox_inches='tight')
    plt.close()

    ########## Corner Plot ##########
    fig = corner.corner(
        flat_samples, labels=params, truths=mcmc.loc[0].to_numpy(), quantiles=[0.16, 0.84]
    )
    plt.savefig(save_path + 'MCMC_Corner.png', dpi=300, bbox_inches='tight')
    plt.close()

    ########## Spectrum Plot ##########
    for i, order in enumerate(orders):
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(sci_specs[i].wave, sci_specs[i].flux, color='C0', label='Data', alpha=0.7, lw=0.7)
        ax.plot(models_notel[i].wave, models_notel[i].flux, color='C4', label='Model', alpha=0.7, lw=0.7)
        ax.plot(models[i].wave, models[i].flux, color='C3', label='Model + Telluric', alpha=0.7, lw=0.7)
        ax.fill_between(sci_specs[i].wave, -sci_specs[i].noise, sci_specs[i].noise, facecolor='0.8', label='Noise')
        ax.plot(sci_specs[i].wave, sci_specs[i].flux - models[i].flux, color='k', label='Residual', alpha=0.7, lw=0.7)
        ax.axhline(y=0, color='k', linewidth=0.7)
        ax.set_xlabel(r'$\lambda$ ($\AA$)', fontsize=15)
        ax.set_ylabel('Normalized Flux', fontsize=15)
        ax.minorticks_on()
        ax.legend(frameon=True, loc='lower left', bbox_to_anchor=(1, 0))
        plt.savefig(save_path + 'Fitted_Spectrum_O%s' %order + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + 'Fitted_Spectrum_O%s' %order + '.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    
    ##################################################
    ################# Writing Result #################
    ##################################################
    
    # teff, vsini, rv, airmass, pwv, veiling, lsf, noise, wave_offset1, flux_offset1, wave_offset2, flux_offset2
    result = {
        'HC2000':               name,
        'year':                 date[0],
        'month':                date[1],
        'day':                  date[2],
        'sci_frames':           sci_frames,
        'tel_frames':           tel_frames,
        'logg':                 mcmc.logg,
        'noise':                mcmc.noise,
        'veiling_param_O34':    veiling_param,
        'model_dip_O34':        model_dips[0],
        'model_std_O34':        model_stds[0],
        'wave_offset_O34':      mcmc.wave_offset_O34,
        'flux_offset_O34':      mcmc.flux_offset_O34,
        'snr_O34':              np.median(sci_specs[0].flux/sci_specs[0].noise)
    }
    
    ########## Write Parameters ##########
    with open(save_path + 'MCMC_Params.txt', 'w') as file:
        for key, value in zip(result.keys(), result.values()):
            if isinstance(value, Iterable) and (not isinstance(value, str)):
                file.write('{}: \t{}\n'.format(key, ", ".join(str(_) for _ in value)))
            else:
                file.write('{}: \t{}\n'.format(key, value))
    
    return result


##################################################
############## Probability Function ##############
##################################################

def lnprior(theta):
    
    # Modify Orders
    logg, noise, wave_offset, flux_offset = theta
    
    # Modify Orders
    if  \
        3.0     < logg          < 5.0  \
    and 1       < noise         < 50    \
    and -10     < wave_offset   < 10   \
    and -1e5    < flux_offset   < 1e5   :
        return 0.0
    else:
        return -np.inf

def lnlike(theta, sci_specs, orders, other_params):
    
    # Modify Orders
    logg, noise, wave_offset, flux_offset = theta
    teff, vsini, rv, airmass, pwv, veiling, lsf = other_params
    # Modify Orders
    wave_offsets = [wave_offset]
    flux_offsets = [flux_offset]
    
    sci_noises = np.array([])
    sci_fluxes = np.array([])
    model_fluxes = np.array([])

    for i, order in enumerate(orders):
        model = smart.makeModel(teff, logg = logg, vsini = vsini, rv = rv, airmass = airmass, pwv = pwv, veiling = veiling, lsf = lsf, z = 0, wave_offset = wave_offsets[i], flux_offset = flux_offsets[i], order = str(order), data = sci_specs[i], modelset = 'phoenix-aces-agss-cond-2011')

        sci_noises = np.concatenate((sci_noises, sci_specs[i].noise * noise))
        sci_fluxes = np.concatenate((sci_fluxes, sci_specs[i].flux))
        model_fluxes = np.concatenate((model_fluxes, model.flux))

    sigma2 = sci_noises**2
    chi = -1/2 * np.sum( (sci_fluxes - model_fluxes)**2 / sigma2 + np.log(2*np.pi*sigma2) )
    
    if np.isnan(chi):
        print('sci_fluxes:\t{}'.format(sci_fluxes))
        print('model_fluxes:\t{}'.format(model_fluxes))
        return -np.inf
    return chi

    
def lnprob(theta, sci_specs, orders, other_params):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike(theta, sci_specs, orders, other_params)


if __name__=='__main__':
    
    # Previous Data:
    dates = [
        # *list(repeat((15, 12, 23), 4)),
        # *list(repeat((15, 12, 24), 8)),
        # *list(repeat((16, 12, 14), 4)),
        # *list(repeat((18, 2, 11), 7)),
        # *list(repeat((18, 2, 12), 5)),
        # *list(repeat((18, 2, 13), 6)),
        # *list(repeat((19, 1, 12), 5)),
        # *list(repeat((19, 1, 13), 6)),
        # *list(repeat((19, 1, 16), 6)),
        # *list(repeat((19, 1, 17), 5)),
        # *list(repeat((20, 1, 18), 2)),
        # *list(repeat((20, 1, 19), 3)),
        # *list(repeat((20, 1, 20), 6)),
        # *list(repeat((20, 1, 21), 7)),
        # *list(repeat((21, 2, 1), 2)),
        # *list(repeat((21, 10, 20), 4)),
        *list(repeat((22, 1, 18), 6)),
        *list(repeat((22, 1, 19), 5)),
        *list(repeat((22, 1, 20), 7))
    ]
    
    names = [
        # # 2015-12-23
        # 322, 296, 259, 213,
        # # 2015-12-24
        # '306A', '306B', '291A', '291B', 252, 250, 244, 261,
        # # 2016-12-14
        # 248, 223, 219, 324,
        # # 2018-2-11
        # 295, 313, 332, 331, 337, 375, 388,
        # # 2018-2-12
        # 425, 713, 408, 410, 436,
        # 2018-2-13
        # '354B2B3_A', '354B2B3_B', '354B1', '354B4', 442, 344,
        # # 2019-1-12
        # '522A', '522B', 145, 202, 188,
        # # 2019-1-13
        # 302, 275, 245, 258, 220, 344,
        # # 2019-1-16
        # 370, 389, 386, 398, 413, 253,
        # # 2019-1-17
        # 288, 420, 412, 282, 217,
        # # 2020-1-18
        # 217, 229,
        # # 2020-1-19
        # 228, 224, 135,
        # # 2020-1-20
        # 440, 450, 277, 204, 229, 214,
        # # 2020-1-21
        # 215, 240, 546, 504, 703, 431, 229,
        # # 2021-2-1
        # 484, 476,
        # # 2021-10-20
        # 546, 217, 277, 435,
        # 2022-1-18
        457, 479, 490, 478, 456, 170, 
        # 2022-1-19
        453, 438, 530, 287, 171, 
        # 2022-1-20
        238, 266, 247, 172, 165, 177, 163
    ]
    
    sci_frames = [
        # # 2015-12-23
        # [75, 76, 77, 78], [81, 82, 83, 84], [87, 88, 89, 90], [91, 92, 93, 94],
        # # 2015-12-24
        # [40, 41, 42, 43], [40, 41, 42, 43], [46, 47, 48, 49], [46, 47, 48, 49], [52, 53, 54, 55], [56, 57, 58, 59], [62, 63, 64, 65], [66, 67, 68, 69],
        # # 2016-12-14
        # [40, 41, 42, 43], [46, 47, 48, 49], [50, 51, 52, 53], [56, 57, 58], 
        # # 2018-2-11
        # [26, 27, 29, 30], [33, 34, 35, 36], [37, 38, 39, 40], [43, 44, 45, 46], [47, 48, 49, 50], [53, 54, 55, 56], [57, 58, 59, 60],
        # # 2018-2-12
        # [47, 48, 49, 50], [54, 55, 56, 57], [58, 59, 60, 61], [64, 65, 66, 67], [70, 71, 72, 73],
        # 2018-2-13
        # [25, 26, 27, 28], [25, 26, 27, 28], [32, 33, 34, 35], [36, 37, 38, 39], [42, 43, 44, 45], [48, 49, 50, 51], 
        # # 2019-1-12
        # [36, 37, 38, 39], [36, 37, 38, 39], [42, 43, 44, 45], [46, 47, 48, 49], [52, 53, 54, 55], 
        # # 2019-1-13
        # [41, 42, 43, 44], [47, 48], [49, 50, 51, 52], [55, 56, 57, 58], [59, 60, 61, 63], [66, 67, 68, 69], 
        # # 2019-1-16
        # [83, 84, 85, 86], [89, 90, 91, 92], [93, 94, 95, 96], [99, 100, 101, 102], [113, 114, 115, 116], [119, 120, 121, 122], 
        # # 2019-1-17
        # [38, 39, 40, 41], [44, 45, 46, 47], [48, 49, 50, 51], [54, 55, 56, 57], [58, 59, 60, 61], 
        # # 2020-1-18
        # [27, 28, 29, 30], [33, 34, 35, 36],
        # # 2020-1-19
        # [31, 32, 33, 34], [37, 38, 39, 40], [41, 42, 43, 44], 
        # # 2020-1-20
        # [32, 33, 34, 35], [36, 37, 38, 39], [42, 43, 44, 45], [46, 47, 48, 49], [50, 51, 52, 53], [54, 55, 56, 57],
        # # 2020-1-21
        # [29, 30, 31, 32], [33, 34, 35, 36], [39, 40, 41, 42], [43, 44, 45, 46], [47, 48, 49, 50], [53, 54, 55, 56], [58, 59, 60], 
        # # 2021-2-1
        # [27, 28, 29, 30], [31, 32, 33, 34, 35, 36], 
        # # 2021-10-20
        # [7, 8, 9, 10], [13, 14, 15, 16], [17, 18], [22, 23, 24],
        # 2022-1-18
        [24, 25, 26, 27], [31, 32, 33, 34], [35, 36, 37, 38], [41, 42, 43, 44], [45, 46, 47, 49], [52, 53, 54, 55], 
        # 2022-1-19
        [26, 27, 28, 29], [32, 33, 34, 35], [36, 37, 38, 39], [42, 43, 44, 45], [46, 47, 48, 49], 
        # 2022-1-20
        [23, 25, 27, 26], [30, 31, 33, 32], [34, 35, 36, 37], [40, 41, 42, 43], [44, 45, 46, 47], [51, 52, 53, 55], [56, 57, 58]
    ]
    
    tel_frames = [
        # # 2015-12-23
        # [79, 80, 80, 79], [79, 80, 80, 79], [95, 96, 96, 95], [95, 96, 96, 95],
        # # 2015-12-24
        # [44, 45, 45, 44], [44, 45, 45, 44], [44, 45, 45, 44], [44, 45, 45, 44], [50, 51, 51, 50], [60, 61, 61, 60], [60, 61, 61, 60], [44, 45, 45, 44], 
        # # 2016-12-14
        # [54, 55, 55, 54], [44, 45, 45, 44], [44, 45, 45, 44], [54, 55, 55], 
        # # 2018-2-11
        # [41, 42, 42, 41], [41, 42, 42, 41], [41, 42, 42, 41], [41, 42, 42, 41], [51, 52, 52, 51], [51, 52, 52, 51], [61, 62, 62, 61],
        # # 2018-2-12
        # [51, 52, 52, 51], [51, 52, 52, 51], [51, 52, 52, 51], [62, 63, 63, 62], [74, 75, 75, 74], 
        # 2018-2-13
        # [40, 41, 41, 40], [40, 41, 41, 40], [40, 41, 41, 40], [40, 41, 41, 40], [46, 47, 47, 46], [52, 53, 53, 52], 
        # # 2019-1-12
        # [40, 41, 41, 40], [40, 41, 41, 40], [40, 41, 41, 40], [50, 51, 51, 50], [50, 51, 51, 50], 
        # # 2019-1-13
        # [39, 40, 40, 39], [45, 46], [53, 54, 54, 53], [53, 54, 54, 53], [64, 65, 65, 64], [64, 65, 65, 64], 
        # # 2019-1-16
        # [87, 88, 88, 87], [87, 88, 88, 87], [97, 98, 98, 97], [97, 98, 98, 97], [117, 118, 118, 117], [117, 118, 118, 117], 
        # # 2019-1-17
        # [42, 43, 43, 42], [42, 43, 43, 42], [62, 63, 63, 62], [52, 53, 53, 52], [62, 63, 63, 62], 
        # # 2020-1-18
        # [31, 32, 32, 31], [31, 32, 32, 31],
        # # 2020-1-19
        # [35, 36, 36, 35], [35, 36, 36, 35], [35, 36, 36, 35],
        # # 2020-1-20
        # [40, 41, 41, 40], [40, 41, 41, 40], [40, 41, 41, 40], [40, 41, 41, 40], [58, 59, 59, 58], [58, 59, 59, 58],
        # # 2020-1-21
        # [61, 62, 62, 61], [37, 38, 38, 37], [51, 52, 52, 51], [51, 52, 52, 51], [51, 52, 52, 51], [37, 38, 38, 37], [61, 62, 62], 
        # # 2021-2-1
        # [37, 38, 38, 37], [37, 38, 38, 37, 37, 38],
        # # 2021-10-20
        # [11, 12, 12, 11], [25, 26, 26, 25], [25, 26], [25, 26, 26]
        # 2022-1-18
        [28, 29, 29, 28], [28, 29, 29, 28], [39, 40, 40, 39], [39, 40, 40, 39], [50, 51, 51, 50], [50, 51, 51, 50], 
        # 2022-1-19
        [30, 31, 31, 30], [30, 31, 31, 30], [30, 31, 31, 30], [30, 31, 31, 30], [30, 31, 31, 30], 
        # 2022-1-20
        [28, 29, 29, 28], [28, 29, 29, 28], [49, 50, 50, 49], [49, 50, 50, 49], [49, 50, 50, 49], [28, 29, 29, 28], [28, 29, 29]
    ]
    
    skip=0
    dim_check = [len(_) for _ in [dates, names, sci_frames, tel_frames]]
    
    if not all(_==dim_check[0] for _ in dim_check):
        sys.exit('Dimensions not agree! dates: {}, names: {}, sci_frames:{}, tel_frames:{}.'.format(*dim_check))
    
    
    names = [str(_) for _ in names]
    
    for i in range(skip, len(names)):
        
        infos = {
            'date':     dates[i],
            'name':     names[i],
            'sci_frames': sci_frames[i],
            'tel_frames': tel_frames[i],
        }
            
            
        result = main(infos=infos, MCMC=True, Finetune=True, Multiprocess=True)