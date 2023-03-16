# Fit a single frame using Multinest

import numpy as np
import matplotlib.pyplot as plt
import corner
import smart
import os, sys
from multiprocessing import Pool
import pymultinest
import json

def main(date, sci_frame, tel_frame, output_prefix = 'M_'):
    """Args:\n
    date: (Year, Month, Date). e.g. (20, 1, 18)\n
    sci_frame: scientific frame number. e.g. 27 for nspec200118_0027\n
    tel_frame: telluric frame number. e.g. 31 for nspec200118_0031\n
    output_prefix: output prefix
    """
    

    orders = [32, 33]
    parameters = ['teff', 'veiling', 'logg'] # , 'noise']

    def lnlike(cube, ndim, nparams):

        # teff, logg, veiling = cube[:3]
        teff = (7000 - 2300) * cube[0] + 2300
        veiling = np.power(10, cube[1])
        logg = (5.0 - 3.0) * cube[2] + 3.0
        # noise = (50-1) * cube[3] + 1
        
        sci_noises = np.array([])
        sci_fluxes = np.array([])
        model_fluxes = np.array([])
        
        for i, order in enumerate(orders):
            model = smart.makeModel(teff, logg = logg, veiling = veiling, z = 0, vsini = vsini, rv = rv, airmass = am, pwv = pwv, wave_offset = wave_offsets[i], flux_offset = flux_offsets[i], order = order, data = sci_specs[i], lsf = lsf, modelset = 'phoenix-aces-agss-cond-2011')
            
            sci_noises = np.concatenate((sci_noises, sci_specs[i].noise * noise))
            sci_fluxes = np.concatenate((sci_fluxes, sci_specs[i].flux))
            model_fluxes = np.concatenate((model_fluxes, model.flux))
        
        sigma2 = sci_noises**2
        chi = -0.5 * np.sum( (sci_fluxes - model_fluxes)**2 / sigma2 + np.log(2 * np.pi * sigma2) )

        return chi


    # cube = teff(0-1 -> 2300-7000), veiling, logg(0~1 -> 3.8~4.2)
    def prior(cube, ndim, nparams):
        # cube[0]  = (3.845098 - 3.361727836) * cube[0] + 3.361727836    # teff
        # cube[0] = (7000 - 2300) * cube[0] + 2300
        if np.log10(veiling_mcmc) < 0.5:
            cube[1] = cube[1] * 2
        elif veiling_mcmc < 1e2:
            cube[1] = 1.5 * round(np.log10(veiling_mcmc)) * cube[1]
        else:
            cube[1]  = 4 * cube[1] + round(np.log10(veiling_mcmc)) - 2
        # cube[2]  = (4.2 - 3.8)     * cube[2] + 3.8     # logg

    Year = str(date[0]).zfill(2)
    Month = str(date[1]).zfill(2)
    Date = str(date[2]).zfill(2)
    n_params = len(parameters)

    Month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    common_prefix = '/home/l3wei/ONC/Data/20' + Year + Month_list[int(Month) - 1] + Date + '/reduced/'
    
    
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
    
    # Line number -1 of: vsini, rv, am, pwv, veiling, wave_offset1, flux_offset1, lsf, noise, wave_offset2, flux_offset2
    if int(Year) > 19:
        mcmc_param_lines = [12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    else:
        mcmc_param_lines = [14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    
    # Special Case
    if int(Year)==20 and int(Month)==1 and int(Date)==19:
        mcmc_param_lines = [14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    
    save_path = common_prefix + 'multinest/' + sci_name + '_%s/' %orders
    if not os.path.exists(save_path): os.makedirs(save_path)

    sci_specs = []
    pixels = []
    print("Start Processing...\nScientific Frame:\t%s\nTelluric Frame:\t\t%s" %(sci_name, tel_name))
    
    ########################################################
    ########### Construct Spectrums of Each Order ##########
    ########################################################
    for order in orders:
        sci_spec = smart.Spectrum(name = sci_name, order = order, path = common_prefix + 'nsdrp_out/fits/all')

        tel_spec = smart.Spectrum(name = tel_name + '_calibrated', order = order, path = common_prefix + tel_name + '/O%s' %order)

        # Update the wavelength solution
        sci_spec.updateWaveSol(tel_spec)

        # Automatically Mask out bad pixels: flux < 0
        pixel = np.arange(len(sci_spec.flux))
        BadPixMask_auto = np.concatenate([BadPixMask, pixel[np.where(sci_spec.flux < 0)]])
        pixel          = np.delete(pixel, BadPixMask_auto)
        sci_spec.flux  = np.delete(sci_spec.flux, BadPixMask_auto)
        sci_spec.noise = np.delete(sci_spec.noise, BadPixMask_auto)
        sci_spec.wave  = np.delete(sci_spec.wave, BadPixMask_auto)
        
        # Plot original spectrums:
        plt.figure(figsize=(16, 6))
        plt.plot(sci_spec.wave, sci_spec.flux, alpha=0.7, lw=0.7)
        # plt.axhline(y=np.median(sci_spec.flux) + 3.5*np.std(sci_spec.flux), linestyle='--', color='C1')
        plt.ylabel('Flux (counts/s)', fontsize=15)
        plt.xlabel('$\lambda$ (\AA)', fontsize=15)
        plt.minorticks_on()
        plt.savefig(save_path + 'Original_Spectrum_O%s' %order + '.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Automatically Mask out bad pixels: flux > median + 3σ
        # pixel = np.arange(len(sci_spec.flux))
        # BadPixMask_auto = pixel[np.where(sci_spec.flux > np.median(sci_spec.flux) + 3.5*np.std(sci_spec.flux))]
        # pixel          = np.delete(pixel, BadPixMask_auto)
        # sci_spec.flux  = np.delete(sci_spec.flux, BadPixMask_auto)
        # sci_spec.noise = np.delete(sci_spec.noise, BadPixMask_auto)
        # sci_spec.wave  = np.delete(sci_spec.wave, BadPixMask_auto)
        
        # Append to the variables with s
        sci_specs.append(sci_spec)  # Type: smart.Spectrum
        pixels.append(pixel)

    barycorr = smart.barycorr(sci_spec.header).value
    
    ##################################################
    ################ Read MCMC Params ################
    ##################################################
    mcmc_param_path = common_prefix + 'mcmc/' + sci_name + '_MultipleOrders_fixedStellarParams_Veiling_addContinuum_KDEMove_' + str(orders) + '_badpix/mcmc_parameters.txt'
    
    if not os.path.exists(mcmc_param_path):
            mcmc_param_path = mcmc_param_path.replace("_badpix", "")
    
    # vsini, rv, am, pwv, veiling, wave_offset1, flux_offset1, lsf, noise, wave_offset2, flux_offset2
    # try opening the file: If not exist: return
    # try:
    with open(mcmc_param_path) as mcmc_param_file:
        mcmc_param_raw = mcmc_param_file.readlines()
        # try reading mcmc_parameters.txt: If empty, return
        # try:
        vsini = float(mcmc_param_raw[mcmc_param_lines[0]].strip("vsini_mcmc ()km/s").split(', ')[0]) # vsini
        rv      = float(mcmc_param_raw[mcmc_param_lines[1]].strip("rv_mcmc ()km/s").split(', ')[0])
        am      = float(mcmc_param_raw[mcmc_param_lines[2]].strip("am_mcmc ()").split(', ')[0])
        pwv     = float(mcmc_param_raw[mcmc_param_lines[3]].strip("pwv_mcmc ()").split(', ')[0])
        veiling_mcmc = float(mcmc_param_raw[mcmc_param_lines[4]].strip("veiling_mcmc ()").split(', ')[0])
        wave_offset1 = float(mcmc_param_raw[mcmc_param_lines[5]][12:-1].split(', ')[0])
        flux_offset1 = float(mcmc_param_raw[mcmc_param_lines[6]][9:-1].split(', ')[0])
        lsf     = float(mcmc_param_raw[mcmc_param_lines[7]].strip("lsf_mcmc ()").split(', ')[0])
        noise       = float(mcmc_param_raw[mcmc_param_lines[8]].strip("noise_mcmc ()").split(', ')[0])
        wave_offset2 = float(mcmc_param_raw[mcmc_param_lines[9]][12:-1].split(', ')[0])
        flux_offset2 = float(mcmc_param_raw[mcmc_param_lines[10]][9:-1].split(', ')[0])
        # except:
        #     print('Failed when trying to read mcmc parameters')
        #     print('Process ended with failure...\n\n')
        #     return
            
    # except:
    #     print('Failed when trying to open' + mcmc_param_path)
    #     print('Process ended with failure...\n\n')
    #     return
    
    wave_offsets = [wave_offset1, wave_offset2]
    flux_offsets = [flux_offset1, flux_offset2]
    
    
    ##################################################
    ################# Run Multinest ##################
    ##################################################
    pymultinest.run(lnlike, prior, n_params, outputfiles_basename = save_path + output_prefix, resume = False, verbose = False)

    print('Multinest Sampling Finished')
    json.dump(parameters, open(save_path + output_prefix + 'params.json', 'w'))

    
    ##################################################
    ################# Analyze Output #################
    ##################################################
    Analyzer = pymultinest.Analyzer(n_params=n_params, outputfiles_basename = save_path + output_prefix)
    param_chain = Analyzer.get_data()[:,2:]
    # param_chain[0,1,2] = teff, veiling, logg
    weights = Analyzer.get_data()[:, 0]
    mask = weights > 1e-4
    
    # Convert param_chain back to scale
    param_chain[:, 0] = param_chain[:, 0] * (7000-2300) + 2300  # teff
    param_chain[:, 1] = np.power(10, param_chain[:, 1])     # convert veiling back to the power of 10
    param_chain[:, 2] = param_chain[:, 2] * (4.2-3.8) + 3.8
    # param_chain[:, 3] = param_chain[:, 3] * (50 - 1) + 1
    np.savetxt(save_path + output_prefix + 'param_chain.txt', param_chain[mask, :])
    
    # parameter distribution: median & uncertainties:
    # e.g.: teff = 3000 -20 +10:
    # e.g.: [[3000, 20, 10], [veiling], [logg], noise]
    param_dist = np.array(list(map(lambda v: [v[1], v[1]-v[0], v[2]-v[1]], zip(*np.percentile(param_chain[mask, :], [16, 50, 84], axis=0)))))
    
    teff_nest    = param_dist[0, :]
    veiling_nest = param_dist[1, :]
    logg_nest    = param_dist[2, :]
    # noise_nest   = param_dist[3, :]
    
    ########################################################
    ############ Construct Models of Each Order ############
    ########################################################
    models = []
    models_notel = []
    for i, order in enumerate(orders):
        model, model_notel = smart.makeModel(teff_nest[0], veiling=veiling_nest[0], logg=logg_nest[0], order=order, data=sci_specs[i], vsini=vsini, rv=rv, airmass=am, pwv=pwv, lsf=lsf, wave_offset=wave_offsets[i], flux_offset=flux_offsets[i], modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)
        models.append(model)
        models_notel.append(model_notel)
    
    
    ##################################################
    ################### Fine Tuning ##################
    ##################################################
    print('Fine Tuning...\n')
    
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
    ################ Re-run Multinest ################
    ##################################################
    pymultinest.run(lnlike, prior, n_params, outputfiles_basename = save_path + output_prefix, resume = False, verbose = False)

    print('Fine-tuned Multinest Sampling Finished')
    json.dump(parameters, open(save_path + output_prefix + 'params.json', 'w'))
    
    
    ##################################################
    ############### Re-Analyze Output ################
    ##################################################
    Analyzer = pymultinest.Analyzer(n_params=n_params, outputfiles_basename = save_path + output_prefix)
    param_chain = Analyzer.get_data()[:,2:]
    # param_chain[0,1,2] = teff, veiling, logg, noise
    weights = Analyzer.get_data()[:, 0]
    mask = weights > 1e-4
    
    # Convert param_chain back to scale
    param_chain[:, 0] = param_chain[:, 0] * (7000-2300) + 2300  # teff
    param_chain[:, 1] = np.power(10, param_chain[:, 1])     # convert veiling back to the power of 10
    param_chain[:, 2] = param_chain[:, 2] * (4.2-3.8) + 3.8
    # param_chain[:, 3] = param_chain[:, 3] * (50 - 1) + 1
    np.savetxt(save_path + output_prefix + 'param_chain.txt', param_chain)
    
    # parameter distribution: median & uncertainties:
    # teff = 3000 -20 +10:
    # [[3000, 20, 10], [veiling], [logg]]
    param_dist = np.array(list(map(lambda v: [v[1], v[1]-v[0], v[2]-v[1]], zip(*np.percentile(param_chain[mask, :], [16, 50, 84], axis=0)))))
    
    teff_nest = param_dist[0, :]
    veiling_nest = param_dist[1, :]
    logg_nest = param_dist[2, :]


    ########################################################
    ########## Re-Construct Models of Each Order ###########
    ########################################################
    models = []
    models_notel = []
    for i, order in enumerate(orders):
        model, model_notel = smart.makeModel(teff_nest[0], veiling=veiling_nest[0], logg=logg_nest[0], order=order, data=sci_specs[i], vsini=vsini, rv=rv, airmass=am, pwv=pwv, lsf=lsf, wave_offset=wave_offsets[i], flux_offset=flux_offsets[i], modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)
        models.append(model)
        models_notel.append(model_notel)
    
    
    ##################################################
    ################## Create Plots ##################
    ##################################################
    # Create Walker Plot: Original
    for idx in range(n_params):
        plt.subplot(n_params, 1, idx+1)
        plt.plot(param_chain[:, idx])
        plt.ylabel(str(parameters[idx]))

    plt.savefig(save_path + output_prefix  + 'Multinest_Walker_Original.png')
    plt.close()

    # Create Walker Plot: Masked
    for idx in range(n_params):
        plt.subplot(n_params, 1, idx+1)
        plt.plot(param_chain[mask, idx])
        plt.ylabel(str(parameters[idx]))

    plt.savefig(save_path + output_prefix  + 'Multinest_Walker.png')
    plt.close()
    
    # Create Corner Plot
    print('Creating Fine-tuned Corner Plot...')
    figure = corner.corner(param_chain[mask, :], labels=parameters, weights=weights[mask], truths=np.median(param_chain[mask, :], axis=0), quantiles=[0.16, 0.84], show_titles=True)
    plt.savefig(save_path + output_prefix + 'Multinest_Corner.png')
    plt.close()
    
    print('Writing Fine-tuned Parameters...\n')
    with open(save_path + output_prefix + 'Multinest_Params.txt', 'w') as file:
        for idx in range(n_params):
            file.write(parameters[idx] + ": \t" + ", ".join(str(i) for i in param_dist[idx, :]) + '\n')
    
    # Create Spectrum Plot
    for i, order in enumerate(orders):
        
        plt.figure(figsize=(16, 6))
        plt.plot(models_notel[i].wave, models_notel[i].flux, 'r-', label='model', alpha=0.7, lw=0.7)
        plt.plot(models[i].wave, models[i].flux, 'm-', label='model + telluric', alpha=0.7, lw=0.7)
        plt.plot(sci_specs[i].wave, sci_specs[i].flux, color='0.5', label='data', alpha=0.7, lw=0.7)
        
        plt.fill_between(sci_specs[i].wave, -sci_specs[i].noise, sci_specs[i].noise, facecolor='0.8', label='noise')
        plt.plot(sci_specs[i].wave, sci_specs[i].flux-models[i].flux, 'k-', label='residual', alpha=0.7, lw=0.7)
        plt.axhline(y=0, color='k', linewidth=0.7)
        plt.ylabel('Flux (counts/s)', fontsize=15)
        plt.xlabel('$\lambda$ (\AA)', fontsize=15)
        plt.minorticks_on()
        plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(1, 0))
        plt.savefig(save_path + 'Spectrum_O%s' %order + '.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print('Process Completed!\n\n')



if __name__=='__main__':
    sci_frame = 31 #27
    tel_frame = 35 #31
    main(date=(20, 1, 19), sci_frame=sci_frame, tel_frame=tel_frame, output_prefix='T_')