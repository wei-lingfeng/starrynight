# Median combine spectrums of each object

import os, shutil
import csv
os.environ["OMP_NUM_THREADS"] = "1" # Limit number of threads
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import smart 
import emcee
import corner
from multiprocessing import Pool
from scipy import interpolate


def main(infos, orders=[32, 33], MCMC=True, Finetune=True, Multiprocess=True):
    
    name = infos['name']
    date = infos['date']
    sci_frames = infos['sci_frames']
    tel_frames = infos['tel_frames']

    # MCMC = True          
    # Multiprocess=False 
    # Finetune=True

    # name = '145'
    # date = (19, 1, 12)
    # sci_frames = [42, 43, 44, 45]
    # tel_frames = [40, 41, 41, 40]
    # orders = [32, 33]

    # Modify Parameters
    params = ['teff', 'vsini', 'rv', 'airmass', 'pwv', 'veiling', 'lsf', 'noise']
    for i in range(np.size(orders)):
            params = params + ['wave_offset_O{}'.format(order), 'flux_offset_O{}'.format(order)]

    nparams, nwalkers, step = len(params), 100, 500
    discard = step - 100

    Year = str(date[0]).zfill(2)
    Month = str(date[1]).zfill(2)
    Date = str(date[2]).zfill(2)

    Month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    common_prefix = '/home/l3wei/ONC/Data/20' + Year + Month_list[int(Month) - 1] + Date + '/reduced/'
    save_path = common_prefix + 'mcmc_median/' + name + '_O%s_params/' %orders

    if MCMC:
        if os.path.exists(save_path): shutil.rmtree(save_path)

    if not os.path.exists(save_path): os.makedirs(save_path)

    print('\n\n')
    print('Date:\t20%s-%s-%s' %(Year, Month, Date))
    print('Science  Frames:\t%s' %str(sci_frames))
    print('Telluric Frames:\t%s' %str(tel_frames))
    print('\n\n')


    ##################################################
    ############### Construct Spectrums ##############
    ##################################################
    # sci_specs = [32 sci_spec, 33 sci_spec]
    sci_specs = []
    barycorrs = []
    for order in orders:
        
        sci_abba = []
        sci_names = []
        tel_names = []
        pixels = []
        
        ##################################################
        ####### Construct Spectrums for each order #######
        ##################################################
        for sci_frame, tel_frame in zip(sci_frames, tel_frames):        
            
            if int(Year) > 18:
                # For data after 2018, sci_names = [nspec200118_0027, ...]
                sci_name = 'nspec' + Year + Month + Date + '_' + str(sci_frame).zfill(4)
                tel_name = 'nspec' + Year + Month + Date + '_' + str(tel_frame).zfill(4)
                BadPixMask  = np.concatenate([np.arange(0,20), np.arange(2000,2048)])

            else:
                # For data prior to 2018 (2018 included)
                sci_name = Month_list[int(Month) - 1] + Date + 's' + str(sci_frame).zfill(4)
                tel_name = Month_list[int(Month) - 1] + Date + 's' + str(tel_frame).zfill(4)
                BadPixMask  = np.concatenate([np.arange(0,10), np.arange(1000,1024)])
            
            sci_names.append(sci_name)
            tel_names.append(tel_name)
            
            if name.endswith('A'):
                sci_spec = smart.Spectrum(name=sci_name + '_A', order=order, path=common_prefix + 'extracted_binaries/' + sci_name + '/O' + str(order))
                tel_spec = smart.Spectrum(name=tel_name + '_calibrated', order=order, path=common_prefix + tel_name + '/O%s' %order)
            elif name.endswith('B'):
                sci_spec = smart.Spectrum(name=sci_name + '_B', order=order, path=common_prefix + 'extracted_binaries/' + sci_name + '/O' + str(order))
                tel_spec = smart.Spectrum(name=tel_name + '_calibrated', order=order, path=common_prefix + tel_name + '/O%s' %order)

            # update the wavelength solution
            sci_spec.updateWaveSol(tel_spec)

            pixel = np.arange(len(sci_spec.flux))

            # Automatically Mask out bad pixels: flux < 0
            BadPixMask_auto = np.concatenate([BadPixMask, pixel[np.where(sci_spec.flux < 0)]])
            pixel          = np.delete(pixel, BadPixMask_auto)
            sci_spec.wave  = np.delete(sci_spec.wave, BadPixMask_auto)
            sci_spec.flux  = np.delete(sci_spec.flux, BadPixMask_auto)
            sci_spec.noise = np.delete(sci_spec.noise, BadPixMask_auto)
            
            # Plot original spectrums:
            plt.figure(figsize=(16, 6))
            plt.plot(sci_spec.wave, sci_spec.flux, alpha=0.7, lw=0.7)
            plt.axhline(y=np.median(sci_spec.flux) + 3*np.std(sci_spec.flux), linestyle='--', color='C1')
            plt.ylabel('Flux (counts/s)', fontsize=15)
            plt.xlabel('$\lambda$ (\AA)', fontsize=15)
            plt.minorticks_on()
            plt.savefig(save_path + 'O%d_%d_Spectrum_Original.png' %(order, sci_frame), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Automatically Mask out bad pixels: flux > median + 3σ
            pixel = np.arange(len(sci_spec.flux))
            BadPixMask_auto = pixel[np.where(sci_spec.flux > np.median(sci_spec.flux) + 3.5*np.std(sci_spec.flux))]
            pixel          = np.delete(pixel, BadPixMask_auto)
            sci_spec.wave  = np.delete(sci_spec.wave, BadPixMask_auto)
            sci_spec.flux  = np.delete(sci_spec.flux, BadPixMask_auto)
            sci_spec.noise = np.delete(sci_spec.noise, BadPixMask_auto)
            
            # Special Case
            if date == (19, 1, 12) and order == 32:
                sci_spec.flux = sci_spec.flux[sci_spec.wave < 23980]
                sci_spec.noise = sci_spec.noise[sci_spec.wave < 23980]
                sci_spec.wave = sci_spec.wave[sci_spec.wave < 23980]
            
            
            # Renormalize
            sci_spec.noise = sci_spec.noise / np.median(sci_spec.flux)
            sci_spec.flux  = sci_spec.flux  / np.median(sci_spec.flux)
            
            
            pixels.append(pixel)
            sci_abba.append(sci_spec)  # Type: smart.Spectrum
            barycorrs.append(smart.barycorr(sci_spec.header).value)

        ##################################################
        ################# Median Combine #################
        ##################################################
        # sci spectrum flux table: 
        # flux1
        # flux2
        # ...
        wave_max = min([max(i.wave) for i in sci_abba])
        wave_min = max([min(i.wave) for i in sci_abba])
        resolution = 0.02

        wave_new = np.arange(wave_min, wave_max, resolution)
        flux_new = np.zeros((np.size(sci_frames), int(np.floor((wave_max - wave_min)/resolution + 1))))
        noise_new = np.zeros(np.shape(flux_new))

        for i in range(np.size(sci_frames)):
            f_flux  = interpolate.interp1d(sci_abba[i].wave, sci_abba[i].flux)
            f_noise = interpolate.interp1d(sci_abba[i].wave, sci_abba[i].noise)
            for j in range(np.size(wave_new)):
                try:
                    flux_new[i, j] = f_flux(wave_new[j])
                except:
                    flux_new[i, j] = np.nan
                
                try:
                    noise_new[i, j] = f_noise(wave_new[j])
                except:
                    noise_new[i, j] = np.nan

        # Adjust flux offset
        # flux_offset  = [np.nanmedian(line) for line in flux_new]
        # noise_offset = [np.nanmedian(line) for line in noise_new]
        # for i in range(np.size(sci_frames)):
        #     flux_new[i, :]  = flux_new[i, :]  - flux_offset[i]  + np.mean(flux_offset)
        #     noise_new[i, :] = noise_new[i, :] - noise_offset[i] + np.mean(noise_offset)


        flux_med  = np.nanmedian(flux_new, axis=0)
        noise_med = np.nanmedian(noise_new, axis=0)


        sci_spec.wave = wave_new
        sci_spec.flux = flux_med
        sci_spec.noise = noise_med
        
        # Median combined sci_spec
        sci_specs.append(sci_spec)


        plt.figure(figsize=(16, 6))
        for i in range(np.size(sci_frames)):
            plt.plot(wave_new, flux_new[i, :], color='C0', alpha=0.5, lw=0.5)
        plt.plot(sci_spec.wave, sci_spec.flux, 'C3', alpha=1, lw=0.5)
        plt.xlabel('$\lambda$ (Angstrom)', fontsize=15)
        plt.ylabel('Flux (count/s)', fontsize=15)
        plt.minorticks_on()
        plt.savefig(save_path + 'Spectrum_O%d.png' %order, dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(16, 6))
        for i in range(np.size(sci_frames)):
            plt.plot(wave_new, noise_new[i, :], color='C0', alpha=0.5, lw=0.5)
        plt.plot(sci_spec.wave, sci_spec.noise, 'C3', alpha=1, lw=0.5)
        plt.xlabel('$\lambda$ (Angstrom)', fontsize=15)
        plt.ylabel('Flux (count/s)', fontsize=15)
        plt.minorticks_on()
        plt.savefig(save_path + 'Noise_O%d.png' %order, dpi=300, bbox_inches='tight')
        plt.show()

    barycorr = np.median(barycorrs)
    
    ##################################################
    ################ Read MCMC Params ################
    ##################################################

    # empty


    ##################################################
    ################## MCMC Fitting ##################
    ##################################################

    pos = [np.append([
        np.random.uniform(2300, 7000),  # Teff
        # np.random.uniform(3.0, 5.0)    # logg
        np.random.uniform(0, 40),       # vsini
        np.random.uniform(-100, 100),   # rv
        np.random.uniform(1, 3),        # airmass
        np.random.uniform(0.5, 20),     # pwv
        np.random.uniform(0, 1e5),      # veiling
        np.random.uniform(1, 10),       # lsf
        np.random.uniform(1, 5),       # noise
    ],
        list(np.reshape([[
        np.random.uniform(-5, 5),       # wave offset n
        np.random.uniform(-10, 10)      # flux offset n
        ] for j in range(np.size(orders))], -1))
    )
    for i in range(nwalkers)]


    move = [emcee.moves.KDEMove()]

    if MCMC:
        
        backend = emcee.backends.HDFBackend(save_path + 'sampler1.h5')
        backend.reset(nwalkers, nparams)
        
        if Multiprocess:
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders), moves=move, backend=backend, pool=pool)
                sampler.run_mcmc(pos, step, progress=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders), moves=move, backend=backend)
            sampler.run_mcmc(pos, step, progress=True)
        
        print('\n\n')
        print(sampler.acceptance_fraction)
        print('\n\n')
        # np.save(save_path + 'sampler_chain', sampler.chain[:, :, :])
        # samples = sampler.chain[:, :, :].reshape((-1, nparams))
        # np.save(save_path + 'samples', samples)
        
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

    ##################################################
    ################ Construct Models ################
    ##################################################
    models = []
    models_notel = []
    for i, order in enumerate(orders):
        model, model_notel = smart.makeModel(mcmc.teff[0], order=order, data=sci_specs[i], logg=4.0, vsini=mcmc.vsini[0], rv=mcmc.rv[0], airmass=mcmc.airmass[0], pwv=mcmc.pwv[0], veiling=mcmc.veiling[0], lsf=mcmc.lsf[0], wave_offset=mcmc.loc[0, 'wave_offset_O{}'.format(order)], flux_offset=mcmc.loc[0, 'flux_offset_O{}'.format(order)], z=0, modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)
        models.append(model)
        models_notel.append(model_notel)

    ########## Write Parameters ##########
    with open(save_path + 'MCMC_Params.txt', 'w') as file:
        for idx in range(nparams):
            file.write(params[idx] + ": \t" + ", ".join(str(i) for i in mcmc[idx, :]) + '\n')


    ##################################################
    #################### Fine Tune ###################
    ##################################################
    # Update spectrums for each order
    sci_specs_new = []
    for i, order in enumerate(orders):
        sci_spec = sci_specs[i]
        pixel = np.arange(len(sci_spec.flux))
        
        # Update BadPixMask
        residue = sci_specs[i].flux - models[i].flux
        BadPixMask_new = pixel[np.where(abs(residue) > np.median(residue) + 3*np.std(residue))]
        
        # Mask out bad pixels after fine-tuning
        pixel           = np.delete(pixel, BadPixMask_new)
        sci_spec.flux   = np.delete(sci_spec.flux, BadPixMask_new)
        sci_spec.noise  = np.delete(sci_spec.noise, BadPixMask_new)
        sci_spec.wave   = np.delete(sci_spec.wave, BadPixMask_new)
        sci_specs_new.append(sci_spec)

    # Update sci_specs
    sci_specs = sci_specs_new


    ##################################################
    ################### Re-run MCMC ##################
    ##################################################
    if Finetune:
        backend = emcee.backends.HDFBackend(save_path + 'sampler2.h5')
        backend.reset(nwalkers, nparams)
        
        print('\n\n')
        print('Finetuning......')
        print('Date:\t20%s-%s-%s' %(Year, Month, Date))
        print('Science  Frames:\t%s' %str(sci_frames))
        print('Telluric Frames:\t%s' %str(tel_frames))
        print('\n\n')
        if Multiprocess:
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders), moves=move, backend=backend, pool=pool)
                sampler.run_mcmc(pos, step, progress=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(sci_specs, orders), moves=move, backend=backend)
            sampler.run_mcmc(pos, step, progress=True)
        
        print('\n\n')
        print(sampler.acceptance_fraction)
        print('\n\n')
        print('---------------------------------------------')
        print('\n\n')
        # np.save(save_path + 'sampler_chain', sampler.chain[:, :, :])
        # samples = sampler.chain[:, :, :].reshape((-1, nparams))
        # np.save(save_path + 'samples', samples)
        
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

    ##################################################
    ############### Re-Construct Models ##############
    ##################################################
    models = []
    models_notel = []
    for i, order in enumerate(orders):
        model, model_notel = smart.makeModel(mcmc.teff[0], order=order, data=sci_specs[i], logg=4.0, vsini=mcmc.vsini[0], rv=mcmc.rv[0], airmass=mcmc.airmass[0], pwv=mcmc.pwv[0], veiling=mcmc.veiling[0], lsf=mcmc.lsf[0], wave_offset=mcmc.loc[0, 'wave_offset_O{}'.format(order)], flux_offset=mcmc.loc[0, 'flux_offset_O{}'.format(order)], z=0, modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)
        models.append(model)
        models_notel.append(model_notel)


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
        
        plt.figure(figsize=(16, 6))
        plt.plot(sci_specs[i].wave, sci_specs[i].flux, color='C0', label='data', alpha=0.7, lw=0.7)
        plt.plot(models_notel[i].wave, models_notel[i].flux, color='C4', label='model', alpha=0.7, lw=0.7)
        plt.plot(models[i].wave, models[i].flux, color='C3', label='model + telluric', alpha=0.7, lw=0.7)

        plt.fill_between(sci_specs[i].wave, -sci_specs[i].noise, sci_specs[i].noise, facecolor='0.8', label='noise')
        plt.plot(sci_specs[i].wave, sci_specs[i].flux - models[i].flux, 'k-', label='residual', alpha=0.7, lw=0.7)
        plt.axhline(y=0, color='k', linewidth=0.7)
        plt.ylabel('Flux (counts/s)', fontsize=15)
        plt.xlabel('$\lambda$ (\AA)', fontsize=15)
        plt.minorticks_on()
        plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(1, 0))
        plt.savefig(save_path + 'Fitted_Spectrum_O%s' %order + '.png', dpi=300, bbox_inches='tight')
        plt.close()

    ########## Write Parameters ##########
    with open(save_path + 'MCMC_Params.txt', 'w') as file:
        for idx in range(nparams):
            file.write(params[idx] + ": \t" + ", ".join(str(i) for i in mcmc[idx, :]) + '\n')
    
    # teff, vsini, rv, airmass, pwv, veiling, lsf, noise, wave_offset1, flux_offset1, wave_offset2, flux_offset2
    result = {
        'teff':             mcmc.teff,
        'vsini':            mcmc.vsini,
        'rv':               mcmc.rv, 
        'rv_helio':         mcmc.rv[0] + barycorr, 
        'am':               mcmc.am, 
        'pwv':              mcmc.pwv, 
        'veiling':          mcmc.veiling,
        'lsf':              mcmc.lsf, 
        'noise':            mcmc.noise, 
        'wave_offset_O32':  mcmc.wave_offset_O32, 
        'flux_offset_O32':  mcmc.flux_offset_O32, 
        'wave_offset_O33':  mcmc.wave_offset_O33, 
        'flux_offset_O33':  mcmc.flux_offset_O33, 
        'S/N_O32':          np.median(sci_specs[0].flux/sci_specs[0].noise), 
        'S/N_O33':          np.median(sci_specs[1].flux/sci_specs[1].noise)
    }
    return result


##################################################
############## Probability Function ##############
##################################################

def lnprior(theta):
    
    # Modify Orders
    teff, vsini, rv, airmass, pwv, veiling, lsf, noise, wave_offset1, flux_offset1, wave_offset2, flux_offset2 = theta
    
    # Modify Orders
    if  \
        2300    < teff          < 7000  \
    and 0       < vsini         < 100   \
    and -100    < rv            < 100   \
    and 1       < airmass       < 3     \
    and 0.5     < pwv           < 20    \
    and 0       < veiling       < 1e20  \
    and 1       < lsf           < 20    \
    and 1       < noise         < 50    \
    and -10     < wave_offset1  < 10    \
    and -1e5    < flux_offset1  < 1e5   \
    and -10     < wave_offset2  < 10    \
    and -1e5    < flux_offset2  < 1e5   :
        return 0.0
    else:
        return -np.inf

def lnlike(theta, sci_specs, orders):
    
    # Modify Orders
    teff, vsini, rv, airmass, pwv, veiling, lsf, noise, wave_offset1, flux_offset1, wave_offset2, flux_offset2 = theta

    # Modify Orders
    wave_offsets = [wave_offset1, wave_offset2]
    flux_offsets = [flux_offset1, flux_offset2]
    
    sci_noises = np.array([])
    sci_fluxes = np.array([])
    model_fluxes = np.array([])

    for i, order in enumerate(orders):
        model = smart.makeModel(teff, logg = 4.0, vsini = vsini, rv = rv, airmass = airmass, pwv = pwv, veiling = veiling, lsf = lsf, z = 0, wave_offset = wave_offsets[i], flux_offset = flux_offsets[i], order = order, data = sci_specs[i], modelset = 'phoenix-aces-agss-cond-2011')

        sci_noises = np.concatenate((sci_noises, sci_specs[i].noise * noise))
        sci_fluxes = np.concatenate((sci_fluxes, sci_specs[i].flux))
        model_fluxes = np.concatenate((model_fluxes, model.flux))

    sigma2 = sci_noises**2
    chi = -1/2 * np.sum( (sci_fluxes - model_fluxes)**2 / sigma2 + np.log(2*np.pi*sigma2) )

    return chi

    
def lnprob(theta, sci_specs, orders):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + lnlike(theta, sci_specs, orders)



if __name__=='__main__':
    
    first_run = True
    # Skip the first skip_rows lines
    skip_rows = 0
    save_path = '/home/l3wei/ONC/Data/Fitted_Params_Binary.csv'
    
    
    names       = []
    dates       = []
    sci_frames  = []
    tel_frames  = []
    
    with open('/home/l3wei/ONC/Data/Fitted_Params_Combined_New.csv', 'r') as file:
        reader = csv.DictReader(file)
        
        for i in range(skip_rows):
            next(reader)
        
        for line in reader:
            if line['HC[2000]'].endswith('A') or line['HC[2000]'].endswith('B'):
                names.append(line['HC[2000]'])
                dates.append((int(line['year']), int(line['month']), int(line['day']) ))
                temp = [line['sci_frame1'], line['sci_frame2'], line['sci_frame3'], line['sci_frame4']]
                while '' in temp:
                    temp.remove('')
                sci_frames.append([int(i) for i in temp])
                
                temp = [line['tel_frame1'], line['tel_frame2'], line['tel_frame3'], line['tel_frame4']]
                while '' in temp:
                    temp.remove('')
                tel_frames.append([int(i) for i in temp])
            else:
                pass
    
    
    if first_run:
        skip_rows = 0
        with open(save_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow([
                'HC[2000]', 'year', 'month', 'day', 
                'sci_frame1', 'sci_frame2', 'sci_frame3', 'sci_frame4',
                'tel_frame1', 'tel_frame2', 'tel_frame3', 'tel_frame4',
                'teff', 'teff_lo', 'teff_hi', 
                'vsini', 'vsini_lo', 'vsini_hi', 
                'rv', 'rv_lo', 'rv_hi', 'rv_helio', 
                'am', 'am_lo', 'am_hi', 
                'pwv', 'pwv_lo', 'pwv_hi', 
                'veiling', 'veiling_lo', 'veiling_hi',
                'lsf', 'lsf_lo', 'lsf_hi',
                'noise', 'noise_lo', 'noise_hi',
                'wave_offset_O32', 'wave_offset_O32_lo', 'wave_offset_O32_hi', 
                'flux_offset_O32', 'flux_offset_O32_lo', 'flux_offset_O32_hi',
                'wave_offset_O33', 'wave_offset_O33_lo', 'wave_offset_O33_hi', 
                'flux_offset_O33', 'flux_offset_O33_lo', 'flux_offset_O33_hi', 
                'S/N_O32', 'S/N_O33'
            ])
    else:
        pass
    
    
    for i in range(len(sci_frames)):
    # for i in range(1):
        infos = {
            'name':     names[i],
            'date':     dates[i],
            'sci_frames': sci_frames[i],
            'tel_frames': tel_frames[i],
        }
        
        # while True:
        #     try:
        result = main(infos=infos, MCMC=True, Finetune=True, Multiprocess=True)
            # except:
            #     print('\n\nError encountered, retrying......\n')
            #     continue
            # break
                    
        with open(save_path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([
                names[i], *dates[i],
                *(sci_frames[i] + [ None for _ in range(4 - len(sci_frames[i])) ]),
                *(tel_frames[i] + [ None for _ in range(4 - len(tel_frames[i])) ]),
                *list(result['teff']),
                *list(result['vsini']),
                *list(result['rv']), 
                result['rv_helio'],
                *list(result['am']),
                *list(result['pwv']),
                *list(result['veiling']),
                *list(result['lsf']),
                *list(result['noise']),
                *list(result['wave_offset_O32']),
                *list(result['flux_offset_O32']),
                *list(result['wave_offset_O33']),
                *list(result['flux_offset_O33']),
                result['S/N_O32'], 
                result['S/N_O33'] 
            ])