import os
import csv
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

user_path = os.path.expanduser('~') 

def fit(teff):
    '''Fit for mass, and logL using Palla model assuming a 2-Myr age.
    
    Parameters:
        Effective temperature, N-by-2 or N-by-3 array (teff; teff_e / teff; teff_lo; teff_hi).
    '''
    
    model_name = 'Palla'
    model_path = f'{user_path}/ONC/starrynight/models/{model_name}/{model_name}_Model.csv'
    
    if np.shape(teff)[1] == 2:
        teff = np.array([teff[:, 0], teff[:, 0] - teff[:, 1], teff[:, 0] + teff[:, 1]]).transpose()
    else:
        pass
    
    # read model
    model_data = pd.read_csv(model_path)
    
    # result = [mass, mass_lo, mass_hi, logL]
    result = dict()
    
    # interpolate mass
    func_2 = interp1d(model_data.teff_2myr, model_data.mass, bounds_error=False, fill_value=None)
    func_1 = interp1d(model_data.teff_1myr, model_data.mass, bounds_error=False, fill_value=None)
    func_3 = interp1d(model_data.teff_3myr, model_data.mass, bounds_error=False, fill_value=None)
    
    points = np.hstack((func_1(teff), func_2(teff), func_3(teff)))
    
    result[f'mass_{model_name}']    = func_2(teff[:, 0])
    result[f'e_mass_{model_name}']  = (np.nanmax(points, axis=1) - np.nanmin(points, axis=1)) / 2
    
    # if truth = nan, then lo & hi = nan
    result[f'e_mass_{model_name}'][np.where( np.isnan(result[f'mass_{model_name}']) )] = np.nan
    
    
    
    # interpolate logL
    func_2 = interp1d(model_data.teff_2myr, model_data.logL_2myr, bounds_error=False, fill_value=None)
    func_1 = interp1d(model_data.teff_1myr, model_data.logL_1myr, bounds_error=False, fill_value=None)
    func_3 = interp1d(model_data.teff_3myr, model_data.logL_3myr, bounds_error=False, fill_value=None)
    
    points = np.hstack((func_1(teff), func_2(teff), func_3(teff)))
    
    result[f'logL_{model_name}']    = func_2(teff[:, 0])
    result[f'e_logL_{model_name}']  = (np.nanmax(points, axis=1) - np.nanmin(points, axis=1)) / 2
    
    # if truth = nan, then lo & hi = nan
    result[f'e_logL_{model_name}'][np.where( np.isnan(result[f'logL_{model_name}']) )] = np.nan
    
    
    return result



if __name__ == '__main__':
    
    # read sample teff
    teff = []
    data_path = f'{user_path}/ONC/Data/nirspec sources.csv'
    with open(data_path, 'r') as file:
        reader = csv.DictReader(file)
        for line in reader:
            teff.append([ float(i) for i in [line['teff'], line['teff_lo'], line['teff_hi']] ])
    teff = np.array(teff)
    
    result = fit(teff)