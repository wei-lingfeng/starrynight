# Grab median combined nirspec params & proper motion table & fitted mass
import sys
import pandas as pd
from itertools import repeat
from collections.abc import Iterable
from astropy import units as u
from astropy.coordinates import SkyCoord

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

def nirspec_sources(dates, names, exceptions, save_path):
    '''Generate dataframe for NIRSPEC sources.
    - Parameters:
        dates: list of iterables of the form (yy, mm, dd).
        names: list of HC2000 IDs, integer or string. e.g. [322, '522A']
        exceptions: dictionary of exceptions using O35.
    - Returns:
        result: pandas dataframe of NIRSPEC sources.
    '''
    
    dim_check = [len(_) for _ in [dates, names]]
    
    if dim_check[0] != dim_check[1]:
        sys.exit('Dimensions not agree: dates {}, names {}'.format(*dim_check))
    
    result = {
        'HC2000':               [],
        '_RAJ2000':             [],
        '_DEJ2000':             [],
        'RAJ2000':              [],
        'DEJ2000':              [],
        'year':                 [],
        'month':                [],
        'day':                  [],
        'itime':                [],
        'sci_frames':           [],
        'tel_frames':           [],
        'teff':                 [],
        'teff_e':               [],
        'logg':                 [],
        'logg_e':               [],
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
        'veiling_param_O34':    [],
        'veiling_param_O35':    [],
        'lsf':                  [],
        'lsf_e':                [],
        'noise':                [],
        'noise_e':              [],
        'model_dip_O32':        [],
        'model_std_O32':        [],
        'model_dip_O33':        [],
        'model_std_O33':        [],
        'model_dip_O34':        [],
        'model_std_O34':        [],
        'model_dip_O35':        [],
        'model_std_O35':        [],
        'wave_offset_O32':      [],
        'wave_offset_O32_e':    [],
        'flux_offset_O32':      [],
        'flux_offset_O32_e':    [],
        'wave_offset_O33':      [],
        'wave_offset_O33_e':    [],
        'flux_offset_O33':      [],
        'flux_offset_O33_e':    [],
        'wave_offset_O34':      [],
        'wave_offset_O34_e':    [],
        'flux_offset_O34':      [],
        'flux_offset_O34_e':    [],
        'wave_offset_O35':      [],
        'wave_offset_O35_e':    [],
        'flux_offset_O35':      [],
        'flux_offset_O35_e':    [],
        'snr_O32':              [],
        'snr_O33':              [],
        'snr_O34':              [],
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
    # global hc_x_pm
    # hc_x_pm = pd.read_csv('/home/l3wei/ONC/Catalogs/Hillenbrand x Kim.csv',
    #                       dtype={'[HC2000]': str, 'ID': str}).to_dict(orient='list')
    hc2000 = pd.read_csv('/home/l3wei/ONC/Catalogs/HC2000.csv', dtype={'[HC2000]': str})
    
    #################################################
    ############### Construct Catalog ###############
    #################################################
    
    for date, name in zip(dates, names):
        year, month, day = date
        
        data_path = '/home/l3wei/ONC/Data/20{year}{month}{day}/reduced/mcmc_median/{name}_O{orders}_params/MCMC_Params.txt'.format(
            year=str(year).zfill(2), month=month_list[month-1], day=str(day).zfill(2), name=name, orders=[32, 33])
        with open(data_path, 'r') as file:
            lines = file.readlines()
        
        index = list(hc2000['[HC2000]']).index(name.split('_')[0])
        coord = SkyCoord(' '.join((hc2000['RAJ2000'][index], hc2000['DEJ2000'][index])), unit=(u.hourangle, u.deg))
        
        result['HC2000'].append(name)
        result['_RAJ2000'].append(coord.ra.degree)
        result['_DEJ2000'].append(coord.dec.degree)
        result['RAJ2000'].append(hc2000['RAJ2000'][index])
        result['DEJ2000'].append(hc2000['DEJ2000'][index])
        result['Kmag'].append(hc2000['Kmag'][index].strip())
        result['Hmag'].append(hc2000['Hmag'][index].strip())
        if hc2000['Kmag'][index].strip():
            result['Kmag_e'].append(hc2000['e_Kmag'][index])
        else:
            result['Kmag_e'].append('')
        
        if hc2000['Hmag'][index].strip():
            result['Hmag_e'].append(hc2000['e_Hmag'][index])
        else:
            result['Hmag_e'].append('')
        result['year'].append(date[0])
        result['month'].append(date[1])
        result['day'].append(date[2])
        
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
            
            elif line.startswith('flux_offset_O32:'):
                value, error = read_text(line)
                result['flux_offset_O32'].append(value)
                result['flux_offset_O32_e'].append(error)
            
            elif line.startswith('wave_offset_O33:'):
                value, error = read_text(line)
                result['wave_offset_O33'].append(value)
                result['wave_offset_O33_e'].append(error)
            
            elif line.startswith('flux_offset_O33:'):
                value, error = read_text(line)
                result['flux_offset_O33'].append(value)
                result['flux_offset_O33_e'].append(error)
            
            elif line.startswith('snr_O32:'):
                result['snr_O32'].append(read_text(line))
            
            elif line.startswith('snr_O33:'):
                result['snr_O33'].append(read_text(line))
        
        
        # logg results:
        logg_path = '/home/l3wei/ONC/Data/20{year}{month}{day}/reduced/mcmc_median/{name}_O{orders}_logg/MCMC_Params.txt'.format(
            year=str(year).zfill(2), month=month_list[month-1], day=str(day).zfill(2), name=name, orders=[34])
        with open(logg_path, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            if line.startswith('logg:'):
                value, error = read_text(line)
                result['logg'].append(value)
                result['logg_e'].append(error)
            
            elif line.startswith('veiling_param_O34:'):
                result['veiling_param_O34'].append(read_text(line))
            
            elif line.startswith('model_dip_O34:'):
                result['model_dip_O34'].append(read_text(line))
            
            elif line.startswith('model_std_O34:'):
                result['model_std_O34'].append(read_text(line))
            
            elif line.startswith('wave_offset_O34:'):
                value, error = read_text(line)
                result['wave_offset_O34'].append(value)
                result['wave_offset_O34_e'].append(error)
            
            elif line.startswith('flux_offset_O34:'):
                value, error = read_text(line)
                result['flux_offset_O34'].append(value)
                result['flux_offset_O34_e'].append(error)
            
            elif line.startswith('snr_O34:'):
                result['snr_O34'].append(read_text(line))
        
        
        # exceptions
        if (date in exceptions['dates']) and (name in exceptions['names']):
            if exceptions['dates'].index(date) == exceptions['names'].index(name):
                data_path = '/home/l3wei/ONC/Data/20{year}{month}{day}/reduced/mcmc_median/{name}_O{orders}_params/MCMC_Params.txt'.format(
                    year=str(year).zfill(2), month=month_list[month-1], day=str(day).zfill(2), name=name, orders=exceptions['orders'])
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
                    
                    elif line.startswith('flux_offset_O35:'):
                        value, error = read_text(line)
                        result['flux_offset_O35'].append(value)
                        result['flux_offset_O35_e'].append(error)
                    
                    elif line.startswith('snr_O35:'):
                        result['snr_O35'].append(read_text(line))
        
        # if not exception:
        else:
            result['veiling_param_O35'].append('')
            result['model_dip_O35'].append('')
            result['model_std_O35'].append('')
            result['wave_offset_O35'].append('')
            result['wave_offset_O35_e'].append('')
            result['flux_offset_O35'].append('')
            result['flux_offset_O35_e'].append('')
            result['snr_O35'].append('')
            
                    
                
    # result.update(cross_kim(result))
    result = pd.DataFrame.from_dict(result)
    
    # write into csv
    pd.DataFrame.from_dict(result).to_csv(save_path, index=False)
    
    return result




#################################################
################# Main Function #################
#################################################

if __name__ == '__main__':

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
    
    exceptions = {
        'dates':[
            (15, 12, 24),
            (18, 2, 11)
        ],
        'names':[
            '291_A',
            337
        ],
        'orders':[35]
    }
    
    result = nirspec_sources(dates=dates, names=names, exceptions=exceptions, save_path='/home/l3wei/ONC/Catalogs/nirspec sources.csv')
