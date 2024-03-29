import os
import csv
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings

user_path = os.path.expanduser('~') 

def fit(teff):
    '''Fit for mass, logg, logL using Feiden standard model assuming a 2-Myr age.
    
    Parameters:
        Effective temperature, N-by-2 or N-by-3 array (teff; teff_e / teff; teff_lo; teff_hi).
    '''
    
    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    model_name = 'Feiden'
    model_path = f'{user_path}/ONC/starrynight/models/{model_name}/{model_name}_Model.csv'
    
    if np.shape(teff)[1] == 2:
        teff = np.array([teff[:, 0], teff[:, 0] - teff[:, 1], teff[:, 0] + teff[:, 1]]).transpose()
    else:
        pass
    
    # read std model
    model_data = pd.read_csv(model_path)
    
    # result = [mass, mass_lo, mass_hi, logg, logL]
    result = dict()
    
    # interpolate standard mass
    func_2 = interp1d(model_data.teff_2myr, model_data.mass_2myr, bounds_error=False, fill_value=None)
    func_1 = interp1d(model_data.teff_1myr, model_data.mass_1myr, bounds_error=False, fill_value=None)
    func_3 = interp1d(model_data.teff_3myr, model_data.mass_3myr, bounds_error=False, fill_value=None)

    points = np.hstack((func_1(teff), func_2(teff), func_3(teff)))

    result[f'mass_{model_name}'] = func_2(teff[:, 0])
    result[f'e_mass_{model_name}'] = (np.nanmax(points, axis=1) - np.nanmin(points, axis=1)) / 2

    # if truth = nan, then lo & hi = nan
    result[f'e_mass_{model_name}'][np.where( np.isnan(result[f'mass_{model_name}']) )] = np.nan
    
    
    
    # interpolate standard logg
    func_2 = interp1d(model_data.teff_2myr, model_data.logg_2myr, bounds_error=False, fill_value=None)
    func_1 = interp1d(model_data.teff_1myr, model_data.logg_1myr, bounds_error=False, fill_value=None)
    func_3 = interp1d(model_data.teff_3myr, model_data.logg_3myr, bounds_error=False, fill_value=None)

    points = np.hstack((func_1(teff), func_2(teff), func_3(teff)))

    result[f'logg_{model_name}']    = func_2(teff[:, 0])
    result[f'e_logg_{model_name}']  = (np.nanmax(points, axis=1) - np.nanmin(points, axis=1)) / 2

    # if truth = nan, then lo & hi = nan
    result[f'e_logg_{model_name}'][np.where( np.isnan(result[f'e_logg_{model_name}']) )] = np.nan
    
    
    
    # interpolate standard logL
    func_2 = interp1d(model_data.teff_2myr, model_data.logL_2myr, bounds_error=False, fill_value=None)
    func_1 = interp1d(model_data.teff_1myr, model_data.logL_1myr, bounds_error=False, fill_value=None)
    func_3 = interp1d(model_data.teff_3myr, model_data.logL_3myr, bounds_error=False, fill_value=None)

    points = np.hstack((func_1(teff), func_2(teff), func_3(teff)))

    result[f'logL_{model_name}'] = func_2(teff[:, 0])
    result[f'e_logL_{model_name}'] = (np.nanmax(points, axis=1) - np.nanmin(points, axis=1)) / 2

    # if truth = nan, then lo & hi = nan
    result[f'e_logL_{model_name}'][np.where( np.isnan(result[f'logL_{model_name}']) )] = np.nan
    
    
    return result



def fit_mag(teff):
    '''Fit for mass, logg, logL using Feiden magnetic model assuming a 2-Myr age.
    
    Parameters:
        Effective temperature, N-by-3 dict or array (keys: teff; teff_lo; teff_hi).
    '''
    
    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    model_name = 'Feiden'
    model_suffix = 'mag'
    model_path = f'{user_path}/ONC/starrynight/models/{model_name}/{model_name}_{model_suffix}_Model.csv'

    if type(teff) is dict:
        teff = np.array(list(teff.values())).transpose()
    else:
        pass
    
    
    # read mag model
    model_data = pd.read_csv(model_path)
    
    # result = [mass, mass_lo, mass_hi, logg, logL]
    result = dict()
    
    # interpolate standard mass
    func_2 = interp1d(model_data.teff_2myr, model_data.mass_2myr, bounds_error=False, fill_value=None)
    func_1 = interp1d(model_data.teff_1myr, model_data.mass_1myr, bounds_error=False, fill_value=None)
    func_3 = interp1d(model_data.teff_3myr, model_data.mass_3myr, bounds_error=False, fill_value=None)

    points = np.hstack((func_1(teff), func_2(teff), func_3(teff)))

    result[f'mass_{model_name}_{model_suffix}']     = func_2(teff[:, 0])
    result[f'e_mass_{model_name}_{model_suffix}']   = (np.nanmax(points, axis=1) - np.nanmin(points, axis=1)) / 2

    # if truth = nan, then lo & hi = nan
    result[f'e_mass_{model_name}_{model_suffix}'][np.where( np.isnan(result[f'mass_{model_name}_{model_suffix}']) )] = np.nan
    
    
    
    # interpolate magnetic logg
    func_2 = interp1d(model_data.teff_2myr, model_data.logg_2myr, bounds_error=False, fill_value=None)
    func_1 = interp1d(model_data.teff_1myr, model_data.logg_1myr, bounds_error=False, fill_value=None)
    func_3 = interp1d(model_data.teff_3myr, model_data.logg_3myr, bounds_error=False, fill_value=None)

    points = np.hstack((func_1(teff), func_2(teff), func_3(teff)))

    result[f'logg_{model_name}_{model_suffix}']     = func_2(teff[:, 0])
    result[f'e_logg_{model_name}_{model_suffix}']   = (np.nanmax(points, axis=1) - np.nanmin(points, axis=1)) / 2
    
    # if truth = nan, then lo & hi = nan
    result[f'e_logg_{model_name}_{model_suffix}'][np.where( np.isnan(result[f'logg_{model_name}_{model_suffix}']) )] = np.nan
    
    
    
    # interpolate magnetic logL
    func_2 = interp1d(model_data.teff_2myr, model_data.logL_2myr, bounds_error=False, fill_value=None)
    func_1 = interp1d(model_data.teff_1myr, model_data.logL_1myr, bounds_error=False, fill_value=None)
    func_3 = interp1d(model_data.teff_3myr, model_data.logL_3myr, bounds_error=False, fill_value=None)

    points = np.hstack((func_1(teff), func_2(teff), func_3(teff)))

    result[f'logL_{model_name}_{model_suffix}']     = func_2(teff[:, 0])
    result[f'logL_e_{model_name}_{model_suffix}']   = (np.nanmax(points, axis=1) - np.nanmin(points, axis=1)) / 2

    # if truth = nan, then lo & hi = nan
    result[f'logL_e_{model_name}_{model_suffix}'][np.where( np.isnan(result[f'logL_{model_name}_{model_suffix}']) )] = np.nan
    
    
    return result
    


if __name__ == '__main__':
    
    # read sample teff
    teff = []
    data_path = f'{user_path}/ONC/starrynight/catalogs/nirspec sources.csv'
    with open(data_path, 'r') as file:
        reader = csv.DictReader(file)
        for line in reader:
            teff.append([ float(i) for i in [line['teff'], line['teff_lo'], line['teff_hi']] ])
    teff = np.array(teff)

    result_std = fit(teff)
    result_mag = fit_mag(teff)