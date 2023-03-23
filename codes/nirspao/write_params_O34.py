import os, sys
import copy
import smart
import emcee
import pickle
import numpy as np
from functools import reduce
from itertools import repeat
from scipy import interpolate
from collections.abc import Iterable

def write_params_O34(params):

    date = params['date']
    name = params['name']
    name = str(name)
    sci_frames = params['sci_frames']
    tel_frames = params['tel_frames']
    orders = [34]

    ylabels = ['logg', 'noise']
    for order in orders:
        ylabels = ylabels + ['wave offset O{}'.format(order), 'flux offset O{}'.format(order)]

    nparams = len(ylabels)
    discard = 400

    year = str(date[0]).zfill(2)
    month = str(date[1]).zfill(2)
    day = str(date[2]).zfill(2)

    month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    common_prefix = '/home/l3wei/ONC/Data/20{}{}{}/reduced/'.format(year, month_list[int(month) - 1], day)
    save_path = '{}mcmc_median/{}_O{}_logg/'.format(common_prefix, name, orders)
    
    
    ##################################################
    ############### Construct Spectrums ##############
    ##################################################
    # sci_specs = [order 34 median combined sci_spec]
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
                tel_spec = smart.Spectrum(name=tel_name + '_calibrated', order=order, path=common_prefix + tel_name + '/O%s' %order)
            elif name.endswith('B'):
                sci_spec = smart.Spectrum(name=sci_name + '_B', order=order, path=common_prefix + 'extracted_binaries/' + sci_name + '/O' + str(order))
                tel_spec = smart.Spectrum(name=tel_name + '_calibrated', order=order, path=common_prefix + tel_name + '/O%s' %order)
            else:
                sci_spec = smart.Spectrum(name=sci_name, order=order, path=common_prefix + 'nsdrp_out/fits/all')
                tel_spec = smart.Spectrum(name=tel_name + '_calibrated', order=order, path=common_prefix + tel_name + '/O%s' %order)
            
            # Update the wavelength solution
            sci_spec.updateWaveSol(tel_spec)
            
            pixel = np.arange(len(sci_spec.wave))
            pixel = pixel.astype(float)
                    
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
        resolution = 0.1
        
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
        elif line.startswith('pwv'):
            other_params.append(float(line.strip('pwv: \t\n').split(', ')[0]))
        elif line.startswith('veiling:'):
            other_params.append(float(line.strip('veiling: \t\n').split(', ')[0]))
        elif line.startswith('lsf:'):
            other_params.append(float(line.strip('lsf: \t\n').split(', ')[0]))
    
    teff, vsini, rv, airmass, pwv, veiling, lsf = other_params
    
    ##################################################
    ################ Read MCMC Result ################
    ##################################################
    
    sampler = emcee.backends.HDFBackend(save_path + 'sampler.h5')
    flat_samples = sampler.get_chain(discard=discard, flat=True)

    mcmc = np.empty((nparams, 3))
    for i in range(nparams):
        mcmc[i, :] = np.percentile(flat_samples[:, i], [16, 50, 84])

    # mcmc[i, :] = [value, lower, upper]
    mcmc = np.array([mcmc[:, 1], mcmc[:, 0], mcmc[:, 2]]).transpose()

    # calculate veiling params
    model_veiling = smart.Model(teff=teff, logg=mcmc[0, 0], order=str(34), modelset='phoenix-aces-agss-cond-2011', instrument='nirspec')
    veiling_param = veiling / np.median(model_veiling.flux)


    ##################################################
    ################ Construct Models ################
    ##################################################

    models = []
    models_notel = []
    model_dips = []
    model_stds = []
    for i, order in enumerate(orders):
        model, model_notel = smart.makeModel(teff, order=str(order), data=sci_specs[i], logg=mcmc[0, 0], vsini=vsini, rv=rv, airmass=airmass, pwv=pwv, veiling=veiling, lsf=lsf, wave_offset=mcmc[2*i - 2*np.size(orders), 0], flux_offset=mcmc[2*i - 2*np.size(orders) + 1, 0], z=0, modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)
        models.append(model)
        models_notel.append(model_notel)
        
        model_dips.append(np.median(model_notel.flux) - min(model_notel.flux))
        model_stds.append(np.std(model_notel.flux))


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
    sci_specs = copy.deepcopy(sci_specs_new)

    # Save sci_specs
    with open(save_path + 'sci_specs.pkl', 'wb') as file:
        pickle.dump(sci_specs, file)

    # teff, vsini, rv, airmass, pwv, veiling, lsf, noise, wave_offset1, flux_offset1, wave_offset2, flux_offset2
    result = {
        'HC2000':               name,
        'year':                 date[0],
        'month':                date[1],
        'day':                  date[2],
        'sci_frames':           sci_frames,
        'tel_frames':           tel_frames,             
        'logg':                 mcmc[0, :],
        'noise':                mcmc[1, :],
        'veiling_param_O34':    veiling_param,
        'model_dip_O34':        model_dips[0],
        'model_std_O34':        model_stds[0],
        'wave_offset_O34':      mcmc[2, :], 
        'flux_offset_O34':      mcmc[3, :], 
        'snr_O34':              np.median(sci_specs[0].flux/sci_specs[0].noise), 
    }

    ########## Write Parameters ##########
    with open(save_path + 'MCMC_Params_new.txt', 'w') as file:
        for key, value in result.items():
            if isinstance(value, Iterable) and (not isinstance(value, str)):
                file.write('{}: \t{}\n'.format(key, ", ".join(str(_) for _ in value)))
            else:
                file.write('{}: \t{}\n'.format(key, value))

    os.replace(save_path + 'MCMC_Params_new.txt', save_path + 'MCMC_Params.txt')


if __name__=='__main__':
    # Previous Data:
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
        238, 266, 247, 172, 165, 177, 163
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
    
    dim_check = [len(_) for _ in [dates, names, sci_frames, tel_frames]]
    if not all(_==dim_check[0] for _ in dim_check):
        sys.exit('Dimensions not agree! dates: {}, names: {}, sci_frames:{}, tel_frames:{}.'.format(*dim_check))
    
    
    for i in range(len(names)):
        
        params = {
            'date':     dates[i],
            'name':     names[i],
            'sci_frames': sci_frames[i],
            'tel_frames': tel_frames[i],
        }
        
        write_params_O34(params=params)