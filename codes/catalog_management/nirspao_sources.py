# Grab median combined nirspao params & proper motion table & fitted mass
import os, sys
import pandas as pd
from numpy import ma
from itertools import repeat
from collections.abc import Iterable
from astropy.time import Time
from astropy import units as u
from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1

user_path = os.path.expanduser('~')

#################################################
################### Functions ###################
#################################################

def read_text(text):
    '''If iterable, read text to list. Otherwise, to value.
    '''
    value = eval(text.split(':')[1])
    if isinstance(value, Iterable):
        value = list(value)
    
    return value


#################################################
################ Parameter setup ################
#################################################

def nirspao_sources(dates, names, exceptions, save_path=None, overwrite=False):
    '''Generate dataframe for NIRSPAO sources.
    - Parameters:
        dates: list of astropy Times.
        names: list of HC2000 IDs, integer or string. e.g. [322, '522A']
        exceptions: dictionary of exceptions using O35.
    - Returns:
        result: pandas dataframe of NIRSPAO sources.
    '''
    
    dim_check = [len(_) for _ in [dates, names]]
    
    if dim_check[0] != dim_check[1]:
        sys.exit('Dimensions not agree: dates {}, names {}'.format(*dim_check))
    
    for i in range(len(exceptions['dates'])):
        exceptions['dates'][i].out_subfmt = 'date'
    
    result = {
        'HC2000':               [],
        'RAJ2000':              [],
        'DEJ2000':              [],
        '_RAJ2000':             [],
        '_DEJ2000':             [],
        'date':                 [],
        'itime':                [],
        'sci_frames':           [],
        'tel_frames':           [],
        'teff':                 [],
        'teff_e':               [],
        'vsini':                [],
        'vsini_e':              [],
        'rv':                   [],
        'rv_helio':             [],
        'rv_e':                 [],
        'airmass':              [],
        'airmass_e':            [],
        'pwv':                  [],
        'pwv_e':                [],
        'veiling':              [],
        'veiling_e':            [],
        'veiling_param_O32':    [],
        'veiling_param_O33':    [],
        'veiling_param_O35':    [],
        'lsf':                  [],
        'lsf_e':                [],
        'noise':                [],
        'noise_e':              [],
        'model_dip_O32':        [],
        'model_std_O32':        [],
        'model_dip_O33':        [],
        'model_std_O33':        [],
        'model_dip_O35':        [],
        'model_std_O35':        [],
        'wave_offset_O32':      [],
        'wave_offset_O32_e':    [],
        'wave_offset_O33':      [],
        'wave_offset_O33_e':    [],
        'wave_offset_O35':      [],
        'wave_offset_O35_e':    [],
        'snr_O32':              [],
        'snr_O33':              [],
        'snr_O35':              [],
        'Kmag':                 [],
        'Kmag_e':               [],
        'Hmag':                 [],
        'Hmag_e':               []
    }
    
    names = [str(_) for _ in names]
    exceptions['names'] = [str(_) for _ in exceptions['names']]
    month_list = ['jan', 'feb', 'mar', 'apr', 'may',
                  'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # read catalogs
    hc2000 = Vizier.get_catalogs('J/ApJ/540/236')['J/ApJ/540/236/table1']
    coords = SkyCoord([f'{ra} {dec}' for ra, dec in zip(hc2000['RAJ2000'], hc2000['DEJ2000'])], unit=(u.hourangle, u.deg))
    
    #################################################
    ############### Construct Catalog ###############
    #################################################
    
    for date, name in zip(dates, names):
        date.out_subfmt = 'date'
        year    = date.datetime.year    # int
        month   = date.datetime.month   # int
        day     = date.datetime.day     # int
        
        data_path = f'{user_path}/ONC/data/nirspao/{year}{month_list[month-1]}{str(day).zfill(2)}/reduced/mcmc_median/{name}_O{[32, 33]}_params/mcmc_params.txt'
        with open(data_path, 'r') as file:
            lines = file.readlines()
        
        index = list(hc2000['__HC2000_']).index(int(name.split('_')[0]))
        
        result['HC2000'].append(name)
        result['RAJ2000'].append(coords[index].ra.degree)
        result['DEJ2000'].append(coords[index].dec.degree)
        result['_RAJ2000'].append(hc2000['RAJ2000'][index])
        result['_DEJ2000'].append(hc2000['DEJ2000'][index])
        if hc2000['Kmag'].mask[index]:
            result['Kmag'].append(None)
            result['Kmag_e'].append(None)
        else:
            result['Kmag'].append(hc2000['Kmag'][index])
            result['Kmag_e'].append(hc2000['e_Kmag'][index])
        if hc2000['Hmag'].mask[index]:
            result['Hmag'].append(None)
            result['Hmag_e'].append(None)
        else:
            result['Hmag'].append(hc2000['Hmag'][index])
            result['Hmag_e'].append(hc2000['e_Hmag'][index])
        result['date'].append(date.value)
        
        for line in lines:
            if line.startswith('itime:'):
                result['itime'].append(read_text(line))
            
            elif line.startswith('sci_frames:'):
                result['sci_frames'].append(read_text(line))
            
            elif line.startswith('tel_frames:'):
                result['tel_frames'].append(read_text(line))
            
            elif line.startswith('teff:'):
                value, error = read_text(line)
                result['teff'].append(value)
                result['teff_e'].append(error)
            
            elif line.startswith('vsini:'):
                value, error = read_text(line)
                result['vsini'].append(value)
                result['vsini_e'].append(error)
            
            elif line.startswith('rv:'):
                value, error = read_text(line)
                result['rv'].append(value)
                result['rv_e'].append(error)
            
            elif line.startswith('rv_helio:'):
                result['rv_helio'].append(read_text(line))
            
            elif line.startswith('airmass:'):
                value, error = read_text(line)
                result['airmass'].append(value)
                result['airmass_e'].append(error)
            
            elif line.startswith('pwv:'):
                value, error = read_text(line)
                result['pwv'].append(value)
                result['pwv_e'].append(error)
            
            elif line.startswith('veiling:'):
                value, error = read_text(line)
                result['veiling'].append(value)
                result['veiling_e'].append(error)
            
            elif line.startswith('veiling_param_O32:'):
                result['veiling_param_O32'].append(read_text(line))
            
            elif line.startswith('veiling_param_O33:'):
                result['veiling_param_O33'].append(read_text(line))
            
            elif line.startswith('lsf:'):
                value, error = read_text(line)
                result['lsf'].append(value)
                result['lsf_e'].append(error)
            
            elif line.startswith('noise:'):
                value, error = read_text(line)
                result['noise'].append(value)
                result['noise_e'].append(error)
            
            elif line.startswith('model_dip_O32:'):
                result['model_dip_O32'].append(read_text(line))
            
            elif line.startswith('model_std_O32:'):
                result['model_std_O32'].append(read_text(line))
            
            elif line.startswith('model_dip_O33:'):
                result['model_dip_O33'].append(read_text(line))
            
            elif line.startswith('model_std_O33:'):
                result['model_std_O33'].append(read_text(line))
            
            elif line.startswith('wave_offset_O32:'):
                value, error = read_text(line)
                result['wave_offset_O32'].append(value)
                result['wave_offset_O32_e'].append(error)
            
            elif line.startswith('wave_offset_O33:'):
                value, error = read_text(line)
                result['wave_offset_O33'].append(value)
                result['wave_offset_O33_e'].append(error)
            
            elif line.startswith('snr_O32:'):
                result['snr_O32'].append(read_text(line))
            
            elif line.startswith('snr_O33:'):
                result['snr_O33'].append(read_text(line))
        
        
        # # logg results:
        # logg_path = f'{user_path}/ONC/data/20{year}{month}{day}/reduced/mcmc_median/{name}_O{orders}_logg/mcmc_params.txt'.format(
        #     year=str(year).zfill(2), month=month_list[month-1], day=str(day).zfill(2), name=name, orders=[34])
        # with open(logg_path, 'r') as file:
        #     lines = file.readlines()
        
        # for line in lines:
        #     if line.startswith('logg:'):
        #         value, error = read_text(line)
        #         result['logg'].append(value)
        #         result['logg_e'].append(error)
            
        #     elif line.startswith('veiling_param_O34:'):
        #         result['veiling_param_O34'].append(read_text(line))
            
        #     elif line.startswith('model_dip_O34:'):
        #         result['model_dip_O34'].append(read_text(line))
            
        #     elif line.startswith('model_std_O34:'):
        #         result['model_std_O34'].append(read_text(line))
            
        #     elif line.startswith('wave_offset_O34:'):
        #         value, error = read_text(line)
        #         result['wave_offset_O34'].append(value)
        #         result['wave_offset_O34_e'].append(error)
            
        #     elif line.startswith('flux_offset_O34:'):
        #         value, error = read_text(line)
        #         result['flux_offset_O34'].append(value)
        #         result['flux_offset_O34_e'].append(error)
            
        #     elif line.startswith('snr_O34:'):
        #         result['snr_O34'].append(read_text(line))
        
        
        # exceptions
        if (date in exceptions['dates']) and (name in exceptions['names']):
            if exceptions['dates'].index(date) == exceptions['names'].index(name):
                data_path = f"{user_path}/ONC/data/nirspao/{year}{month_list[month-1]}{str(day).zfill(2)}/reduced/mcmc_median/{name}_O{exceptions['orders']}_params/mcmc_params.txt"
                with open(data_path, 'r') as file:
                    lines = file.readlines()
                
                for line in lines:
                    if line.startswith('teff:'):
                        value, error = read_text(line)
                        result['teff'][-1] = value
                        result['teff_e'][-1] = error
                    
                    elif line.startswith('vsini:'):
                        value, error = read_text(line)
                        result['vsini'][-1] = value
                        result['vsini_e'][-1] = error
                    
                    elif line.startswith('rv:'):
                        value, error = read_text(line)
                        result['rv'][-1] = value
                        result['rv_e'][-1] = error
                    
                    elif line.startswith('rv_helio:'):
                        result['rv_helio'][-1] = read_text(line)
                    
                    elif line.startswith('airmass:'):
                        value, error = read_text(line)
                        result['airmass'][-1] = value
                        result['airmass_e'][-1] = error
                    
                    elif line.startswith('pwv:'):
                        value, error = read_text(line)
                        result['pwv'][-1] = value
                        result['pwv_e'][-1] = error
                    
                    elif line.startswith('veiling:'):
                        value, error = read_text(line)
                        result['veiling'][-1] = value
                        result['veiling_e'][-1] = error
                    
                    elif line.startswith('veiling_param_O35:'):
                        result['veiling_param_O35'].append(read_text(line))
                    
                    elif line.startswith('lsf:'):
                        value, error = read_text(line)
                        result['lsf'][-1] = value
                        result['lsf_e'][-1] = error
                    
                    elif line.startswith('noise:'):
                        value, error = read_text(line)
                        result['noise'][-1] = value
                        result['noise_e'][-1] = error
                    
                    elif line.startswith('model_dip_O35:'):
                        result['model_dip_O35'].append(read_text(line))
                    
                    elif line.startswith('model_std_O35:'):
                        result['model_std_O35'].append(read_text(line))
                    
                    elif line.startswith('wave_offset_O35:'):
                        value, error = read_text(line)
                        result['wave_offset_O35'].append(value)
                        result['wave_offset_O35_e'].append(error)
                    
                    elif line.startswith('snr_O35:'):
                        result['snr_O35'].append(read_text(line))
        
        # if not exception:
        else:
            result['veiling_param_O35'].append(None)
            result['model_dip_O35'].append(None)
            result['model_std_O35'].append(None)
            result['wave_offset_O35'].append(None)
            result['wave_offset_O35_e'].append(None)
            result['snr_O35'].append(None)
    
    result = pd.DataFrame.from_dict(result)
    result = QTable.from_pandas(result, units={
        'RAJ2000': u.deg,
        'DEJ2000': u.deg,
        'itime': u.s,
        'teff': u.K, 'teff_e': u.K,
        'vsini': u.km/u.s, 'vsini_e': u.km/u.s,
        'rv': u.km/u.s, 'rv_helio': u.km/u.s, 'rv_e': u.km/u.s, 
        'Kmag': u.mag, 'Kmag_e': u.mag, 'Hmag': u.mag, 'Hmag_e': u.mag
    })
    
    # write result
    if save_path is not None:
        result.write(save_path, overwrite=overwrite)
        # pd.DataFrame.from_dict(result).to_csv(save_path, index=False)
    
    return result




#################################################
################# Main Function #################
#################################################

if __name__ == '__main__':

    dates = [
        *list(repeat(Time('2015-12-23'), 4)),
        *list(repeat(Time('2015-12-24'), 8)),
        *list(repeat(Time('2016-12-14'), 4)),
        *list(repeat(Time('2018-2-11'), 7)),
        *list(repeat(Time('2018-2-12'), 5)),
        *list(repeat(Time('2018-2-13'), 6)),
        *list(repeat(Time('2019-1-12'), 5)),
        *list(repeat(Time('2019-1-13'), 6)),
        *list(repeat(Time('2019-1-16'), 6)),
        *list(repeat(Time('2019-1-17'), 5)),
        *list(repeat(Time('2020-1-18'), 2)),
        *list(repeat(Time('2020-1-19'), 3)),
        *list(repeat(Time('2020-1-20'), 6)),
        *list(repeat(Time('2020-1-21'), 7)),
        *list(repeat(Time('2021-2-1'), 2)),
        *list(repeat(Time('2021-10-20'), 4)),
        *list(repeat(Time('2022-1-18'), 6)),
        *list(repeat(Time('2022-1-19'), 5)),
        *list(repeat(Time('2022-1-20'), 7))
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
    
    exceptions = {
        'dates':[
            Time('2015-12-24'),
            Time('2018-2-11')
        ],
        'names':[
            '291_A',
            337
        ],
        'orders':[35]
    }
    
    result = nirspao_sources(dates=dates, names=names, exceptions=exceptions, save_path=f'{user_path}/ONC/starrynight/catalogs/nirspao sources.csv', overwrite=True)
    result = nirspao_sources(dates=dates, names=names, exceptions=exceptions, save_path=f'{user_path}/ONC/starrynight/catalogs/nirspao sources.ecsv', overwrite=True)
