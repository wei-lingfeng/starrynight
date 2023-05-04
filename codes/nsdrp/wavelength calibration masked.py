####################################################
# The example to demonstrate the telluric wavelength
# calibration. The calibrated spectra will be save
# to the folder "tell_wave_cal" under the path
# 'data/reduced/fits/all'
####################################################
## Import the nirspec_pip package
import smart
import shutil
import os, sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from smart.utils.mask import generate_telluric_mask
from multiprocessing.pool import Pool

user_path = os.path.expanduser('~')

def wave_cal(date, frame, order, pwv='3.5', guess_pwv=True, xcorr_range=20, outlier_rej=3., save=True, test=False):
    year, month, day = date
    name = str(frame)
    
    year    = str(year).zfill(2)
    month   = str(month).zfill(2)
    day     = str(day).zfill(2)
    
    month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    if int(year) > 18:
        data_name = 'nspec' + year + month + day + '_' + name.zfill(4) # Filename of the telluric standard
        pixel_start = 10
        pixel_end = -50
    else:
        data_name = month_list[int(month)-1] + day + 's' + name.zfill(4)
        pixel_start = 10
        pixel_end = -30
    
    data_path = f'{user_path}/ONC/data/nirspao/20{year}{month_list[int(month)-1]}{day}/reduced/nsdrp_out/fits/all'
    
    print('Starting to calibrate {}, order {}.'.format(data_name, order))
    sys.stdout.flush()
    
    cal_param = {
        '32':{'xcorr_range': xcorr_range, 'outlier_rej': outlier_rej, 'pixel_range_start': pixel_start, 'pixel_range_end': pixel_end},
        '33':{'xcorr_range': xcorr_range, 'outlier_rej': outlier_rej, 'pixel_range_start': pixel_start, 'pixel_range_end': pixel_end},
        '34':{'xcorr_range': xcorr_range, 'outlier_rej': outlier_rej, 'pixel_range_start': pixel_start, 'pixel_range_end': pixel_end},
        '35':{'xcorr_range': xcorr_range, 'outlier_rej': outlier_rej, 'pixel_range_start': pixel_start, 'pixel_range_end': pixel_end},
        '36':{'xcorr_range': xcorr_range, 'outlier_rej': outlier_rej, 'pixel_range_start': pixel_start, 'pixel_range_end': pixel_end},
        '37':{'xcorr_range': xcorr_range, 'outlier_rej': outlier_rej, 'pixel_range_start': pixel_start, 'pixel_range_end': pixel_end}
    }
    save_path = f'{user_path}/ONC/data/nirspao/20{year}{month_list[int(month)-1]}{day}/reduced/{data_name}'
    
    if os.path.exists(save_path + '/O%s'.format(order)):
        shutil.rmtree(save_path + '/O%s'.format(order))
    
    data    = smart.Spectrum(name = data_name, order = order, path = data_path)
    auto_mask = np.where(np.logical_or(data.flux > np.inf, data.flux < 0))[0]
    
    # Mask out bad pixels
    # mask 1: mask edge & negative & infinity
    mask_1 = reduce(np.union1d, (auto_mask, np.arange(0,pixel_start), np.arange(len(data.wave) + pixel_end, len(data.wave))))
    mask_1 = mask_1.astype(int)
    fig, ax = plt.subplots(figsize=(15,5), dpi=300)
    ax.plot(data.wave, data.flux, lw=0.5, alpha=0.5, label='Original Spectrum') # Raw
    
    # Replace bad pixels with nan
    pixels = np.arange(0, len(data.wave))
    data.flux   [mask_1] = np.nan
    data.noise  [mask_1] = np.nan
    data.wave   [mask_1] = np.nan
    
    # mask 2: mask flux  > median + 3 sigma
    median_flux = np.nanmedian(data.flux)
    upper_bound = median_flux + outlier_rej*np.nanstd(data.flux - median_flux)
    mask_2 = pixels[np.where(data.flux > upper_bound)]
    data.flux   [mask_2] = np.nan
    data.noise  [mask_2] = np.nan
    data.wave   [mask_2] = np.nan
    
    # mask 3: mask isolated bad pixels
    median_flux = np.nanmedian(data.flux)
    lower_bound = median_flux - 3.*np.nanstd(data.flux - median_flux)
    lowest_bound = median_flux - 4.*np.nanstd(data.flux - median_flux)
    mask_3 = np.array([i for i in np.arange(1, len(data.wave)-1) if (data.flux[i] < lowest_bound) and (data.flux[i-1] >= lower_bound) and (data.flux[i+1] >= lower_bound)], dtype=int)
    data.flux   [mask_3] = np.nan
    data.noise  [mask_3] = np.nan
    data.wave   [mask_3] = np.nan
    
    ax.plot(data.wave, data.flux, lw=0.5, color='C3',alpha=0.5, label='Masked Spectrum') # Masked
    
    ax.axhline(y=upper_bound, linestyle='--', label='Upper bound')
    ax.axhline(y=lower_bound, linestyle='--', label='Lower bound')
    ax.axhline(y=lowest_bound, linestyle='--', label='Lowest bound')
    ax.legend()
    ax.set_xlabel(r'$\lambda$ ($\AA)$')
    ax.set_ylabel(r'$F_{\lambda}$')
    if not os.path.exists(f'{save_path}/O{order}'):
        os.makedirs(f'{save_path}/O{order}')
    plt.savefig(f'{save_path}/O{order}/Spectrum.png')
    plt.show()
    
    mask_custom = reduce(np.union1d, (mask_1, mask_2, mask_3))
    mask_custom = mask_custom.astype(int)
    
    # Run the telluric wavelength calibration
    smart.run_wave_cal(data_name, data_path, [order], save_path, mask_custom=mask_custom, apply_edge_mask=False, test=test, save=save, cal_param=cal_param, pwv=pwv)
    
    
    if guess_pwv:
        # Apply a telluric mask and adjust pwv
        mask_tel, pwv = generate_telluric_mask(
            name=f'{data_name}_calibrated', 
            order=order, 
            path=f'{save_path}/O{order}', 
            pixel_start=pixel_start, pixel_end=pixel_end, sigma=3.0, guess_pwv=True, diagnostic=True
        )
        print('Adjusted pwv={}'.format(pwv))
        mask = np.union1d(mask_tel, mask_custom)
        mask.sort()
        mask = mask.astype(int)
        
        # Rerun telluric wavelength calibration
        smart.run_wave_cal(data_name, data_path, [order], save_path, mask_custom=mask, apply_edge_mask=False, test=test, save=save, cal_param=cal_param, pwv=pwv)
    
    else:
        pass
    
    print('Finished calibrating {}, order {}.\n\n\n'.format(data_name, order))



if __name__ == '__main__':
    
    ## Set up the input paramters
    year    = 15
    month   = 12
    day     = 24
    frames  = [44, 45]
    orders  = [32, 35]
    guess_pwv = False
    pwvs = [1.5, 1.5] # [0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0]
    xcorr_range = 20
    
    date = (year, month, day)
    pwvs = [f'{pwv:.1f}' for pwv in pwvs]
    if (not guess_pwv) and (any([pwv not in [f'{_:.1f}' for _ in [0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0]] for pwv in pwvs])):
        raise ValueError(f'pwv must be one of {[0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0]}.')
    
    params = []
    for frame, pwv in zip(frames, pwvs):
        for order in orders:
            params.append((date, frame, order, pwv, guess_pwv, xcorr_range))
    
    if len(params) > 1:
        with Pool(min((len(params), 32))) as pool:
            pool.starmap(wave_cal, params)
    else:
        wave_cal(*params[0])