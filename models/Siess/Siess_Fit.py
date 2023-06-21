import csv
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

user_path = os.path.expanduser('~') 

def fit(teff):
    '''Fit for mass, logg, and logL using Baraffe model assuming a 2-Myr age.
    
    Parameters:
        Effective temperature, N-by-2 or N-by-3 array (teff; teff_e / teff; teff_lo; teff_hi).
    '''
    
    model_name = 'Siess'
    model_path = f'{user_path}/ONC/Models/{model_name}/{model_name}_Model.csv'
    
    if np.shape(teff)[1] == 2:
        teff = np.array([teff[:, 0], teff[:, 0] - teff[:, 1], teff[:, 0] + teff[:, 1]]).transpose()
    else:
        pass
    
    # read model
    model_data = pd.read_csv(model_path)
    
    # result = [mass, mass_lo, mass_hi, logg, logL]
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
    
    
    
    # interpolate logg
    func_2 = interp1d(model_data.teff_2myr, model_data.logg_2myr, bounds_error=False, fill_value=None)
    func_1 = interp1d(model_data.teff_1myr, model_data.logg_1myr, bounds_error=False, fill_value=None)
    func_3 = interp1d(model_data.teff_3myr, model_data.logg_3myr, bounds_error=False, fill_value=None)

    points = np.hstack((func_1(teff), func_2(teff), func_3(teff)))

    result[f'logg_{model_name}']    = func_2(teff[:, 0])
    result[f'e_logg_{model_name}']  = (np.nanmax(points, axis=1) - np.nanmin(points, axis=1)) / 2
    
    # if truth = nan, then lo & hi = nan
    result[f'e_logg_{model_name}'][np.where( np.isnan(result[f'logg_{model_name}']) )] = np.nan
    
    
    
    # interpolate logL
    func_2 = interp1d(model_data.teff_2myr, model_data.L_2myr, bounds_error=False, fill_value=None)
    func_1 = interp1d(model_data.teff_1myr, model_data.L_1myr, bounds_error=False, fill_value=None)
    func_3 = interp1d(model_data.teff_3myr, model_data.L_3myr, bounds_error=False, fill_value=None)

    points = np.hstack((func_1(teff), func_2(teff), func_3(teff)))

    result[f'logL_{model_name}']    = np.log10(func_2(teff[:, 0]))
    result[f'e_logL_{model_name}']  = (np.log10(np.nanmax(points, axis=1)) - np.log10(np.nanmin(points, axis=1))) / 2
    
    # if truth = nan, then lo & hi = nan
    result[f'e_logL_{model_name}'][np.where( np.isnan(result[f'logL_{model_name}']) )] = np.nan
    
    
    return result



if __name__ == '__main__':
    
    save_path  = 'Siess_Fit.csv'
    
    # read sample teff
    teff = []
    data_path = '/home/l3wei/ONC/Data/nirspec sources.csv'
    with open(data_path, 'r') as file:
        reader = csv.DictReader(file)    
        for line in reader:
            teff.append([ float(i) for i in [line['teff'], line['teff_lo'], line['teff_hi']] ])
    teff = np.array(teff)
    
    result = fit(teff)
    
    # save result
    with open(save_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(list(result.keys()))
        writer.writerows( np.array(list(result.values())).transpose() )