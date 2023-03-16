import sys
import csv
import numpy as np
sys.path.insert(0, r'/home/l3wei/ONC')

from Models.BHAC15 import BHAC15_Fit
from Models.Feiden import Feiden_Fit
from Models.MIST import MIST_Fit
from Models.Palla import Palla_Fit


# fit nirspec
def fit_nirspec(data_path):
# read sample teff
    teff = []
    with open(data_path, 'r') as file:
        reader = csv.DictReader(file)    
        for line in reader:
            teff.append([ float(i) for i in [line['teff'], line['teff_e']] ])
    teff = np.array(teff)

    result = {
        **BHAC15_Fit.fit(teff),
        **MIST_Fit.fit(teff),
        **Feiden_Fit.fit(teff),
        # **Feiden_Fit.fit_mag(teff),
        **Palla_Fit.fit(teff)
    }
    
    return result


def fit_apogee(data_path):
# read sample teff
    teff = []
    with open(data_path, 'r') as file:
        reader = csv.DictReader(file)    
        for line in reader:
            teff.append([ float(i) for i in [line['teff'], line['teff_e']] ])
    teff = np.array(teff)
    
    result = {
        **BHAC15_Fit.fit(teff),
        **MIST_Fit.fit(teff),
        **Feiden_Fit.fit(teff),
        # **Feiden_Fit.fit_mag(teff),
        **Palla_Fit.fit(teff)
    }
    
    return result



if __name__ == '__main__':
    
    path_nirspec = '/home/l3wei/ONC/Data/nirspec sources.csv'
    result_nirspec = fit_nirspec(path_nirspec)
    
    # save result
    save_nirspec = '/home/l3wei/ONC/Data/nirspec interp mass.csv'
    with open(save_nirspec, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(list(result_nirspec.keys()))
        writer.writerows( np.array(list(result_nirspec.values())).transpose() )
    
    # path_apogee = '/home/l3wei/ONC/Data/apogee sources.csv'
    # result_apogee = fit_apogee(path_apogee)
    
    # # save result
    # save_apogee = '/home/l3wei/ONC/Data/apogee interp mass.csv'
    # with open(save_apogee, 'w') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(list(result_apogee.keys()))
    #     writer.writerows( np.array(list(result_apogee.values())).transpose() )
