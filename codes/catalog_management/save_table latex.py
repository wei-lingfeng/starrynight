import os, sys
import numpy as np
import pandas as pd
from numpy import ma
from astropy.io import ascii
from astropy import units as u
from astropy.table import Table, QTable, Column, MaskedColumn

user_path = os.path.expanduser('~')

month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sources = QTable.read(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog.csv')
hc2000 = sources[~sources['HC2000'].mask & sources['theta_orionis'].mask]

result = QTable.read(f'{user_path}/ONC/starrynight/catalogs/sources post-processing.ecsv')
result_df = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/sources post-processing.csv')
trapezium_only = result['obs_date'].mask & result['APOGEE'].mask
result = result[~trapezium_only]

obs_df = pd.DataFrame({
    'HC2000 ID': [f"HC2000 {ID:.0f}{m_ID}" for ID, m_ID in zip(hc2000['HC2000'], hc2000['m_HC2000'].filled(''))],
    'Obs. Date': [f"{date.split('-')[0]} {month_list[int(date.split('-')[1]) - 1]} {date.split('-')[2]}" for date in hc2000['obs_date']],
    'No. of Frames': [len(eval(_)) for _ in hc2000['sci_frames']],
    'Int. Time': [int(_) for _ in hc2000['itime']]
})


result_qtable = QTable({
    'HC2000':           result['HC2000'],
    'K19 ID':           result['ID_kim'],
    'APOGEE':           result['APOGEE'],
    'Gaia DR3':         result['Gaia DR3'], 
    'RAJ2000':          result['RAJ2000'],
    'DEJ2000':          result['DEJ2000'],
    'Obs. Date':        result['obs_date'],
    'Teff':             result['teff'],
    'e_Teff':           result['e_teff'],
    'RV':               result['rv'],
    'e_RV':             result['e_rv'],
    'pmRA':             result['pmRA'],
    'e_pmRA':           result['e_pmRA'],
    'pmDE' :            result['pmDE'],
    'e_pmDE':           result['e_pmDE'],
    
    'Teff_NIRSPAO':     result['teff_nirspao'],
    'e_Teff_NIRSPAO':   result['e_teff_nirspao'],
    'RV_NIRSPAO':       result['rv_helio'],
    'e_RV_NIRSPAO':     result['e_rv_nirspao'],
    'vsini_NIRSPAO':    result['vsini_nirspao'],
    'e_vsini_NIRSPAO':  result['e_vsini_nirspao'],
    'SNR O32':          result['snr_O32'],
    'SNR O33':          result['snr_O33'],
    'SNR O35':          result['snr_O35'],
    'Veil Param O32':   result['veiling_param_O32'],
    'Veil Param O33':   result['veiling_param_O33_nirspao'],
    'Veil Param O35':   result['veiling_param_O35'],
    
    'Teff_T22':         result['teff_apogee'],
    'e_Teff_T22':       result['e_teff_apogee'],
    'RV_T22':           result['rv_apogee'],
    'e_RV_T22':         result['e_rv_apogee'],
    'vsini_T22':        result['vsini_apogee'],
    'e_vsini_T22':      result['e_vsini_apogee'],
    
    'pmRA_K19':         result['pmRA_kim'],
    'e_pmRA_K19':       result['e_pmRA_kim'],
    'pmDE_K19':         result['pmDE_kim'],
    'e_pmDE_K19':       result['e_pmDE_kim'],
    
    'pmRA_DR3':         result['pmRA_gaia'],
    'e_pmRA_DR3':       result['e_pmRA_gaia'],
    'pmDE_DR3':         result['pmDE_gaia'],
    'e_pmDE_DR3':       result['e_pmDE_gaia'],
    
    'M_MIST':           result['mass_MIST'],
    'e_M_MIST':         result['e_mass_MIST'],
    'M_BHAC15':         result['mass_BHAC15'],
    'e_M_BHAC15':       result['e_mass_BHAC15'],
    'M_Feiden':         result['mass_Feiden'],
    'e_M_Feiden':       result['e_mass_Feiden'],
    'M_Palla':          result['mass_Palla'],
    'e_M_Palla':        result['e_mass_Palla'],
})
result_df = result_qtable.to_pandas()

rounding = {
    'RAJ2000':6, 'DEJ2000':6, 
    'Teff':1, 'e_Teff':1, 
    'RV': 2, 'e_RV':2, 
    'pmRA':2, 'e_pmRA':2, 'pmDE':2, 'e_pmDE':2, 
    'Teff_NIRSPAO':1, 'e_Teff_NIRSPAO':1, 
    'RV_NIRSPAO': 2, 'e_RV_NIRSPAO': 2, 
    'vsini_NIRSPAO':2, 'e_vsini_NIRSPAO':2, 
    'SNR O32':2, 'SNR O33':2, 'SNR O35':2, 
    'Veil Param O32':2, 'Veil Param O33':2, 'Veil Param O35':2, 
    'Teff_T22':1, 'e_Teff_T22':1, 
    'RV_T22':2, 'e_RV_T22':2,
    'vsini_T22':2, 'e_vsini_T22':2, 
    'pmRA_K19':2, 'e_pmRA_K19':2, 'pmDE_K19':2, 'e_pmDE_K19':2, 
    'pmRA_DR3':3, 'e_pmRA_DR3':3, 'pmDE_DR3':3, 'e_pmDE_DR3':3,
    'M_MIST':3, 'e_M_MIST':3, 'M_BHAC15':3, 'e_M_BHAC15':3, 
    'M_Feiden':3, 'e_M_Feiden':3, 'M_Palla':3, 'e_M_Palla':3
}

result_qtable.round(rounding)
result_df.round(rounding)

for key, precision in rounding.items():
    result_qtable[key].info.format = f'.{precision}f'

new_HC2000 = []
for i in range(len(result_qtable)):
    if result_qtable['HC2000'].mask[i]:
        new_HC2000.append('')
    elif result['m_HC2000'].mask[i]:
        new_HC2000.append(f"HC2000 {result_qtable['HC2000'][i]}")
    else:
        new_HC2000.append(f"HC2000 {result_qtable['HC2000'][i]}{result['m_HC2000'][i]}")

result_qtable['HC2000'] = MaskedColumn(new_HC2000, mask=result_qtable['HC2000'].mask)
result_qtable['APOGEE'] = MaskedColumn([result_qtable['APOGEE'][i].replace('-', '--') if ~result_qtable['APOGEE'].mask[i] else '' for _ in range(len(result_qtable))], mask=result_qtable['APOGEE'].mask)

# save latex
# observation latex table
print('Observation Table:')
obs_latex = obs_df.to_latex(index=False, header=obs_df.keys().to_list(), na_rep='snodata')
obs_latex.replace('snodata', r'\nodata')
print(obs_latex)

# result latex table
selected_columns = list(result_qtable.keys())[:15] + ['M_MIST', 'e_M_MIST']
selected_columns.remove('Obs. Date')
print('Result Table:')
result_qtable[:20][selected_columns].write('temp.txt', format='latex', fill_values=(ascii.masked, r'\nodata'))
with open('temp.txt', 'r') as file:
    result_latex = file.read()
os.remove('temp.txt')
result_latex = result_latex.replace('nan', r'\nodata')

# merge value and error columns
is_err = [True if 'e_' in key else False for key in selected_columns]
result_latex = result_latex.split('\n')[4:-3]
for i in range(len(result_latex)):
    line = result_latex[i].strip(' \\').split('&')
    for err_idx in np.where(is_err)[0]:
        if line[err_idx - 1].strip() == r'\nodata':
            pass
        else:
            line[err_idx - 1] = r'\pm '.join((line[err_idx - 1], line[err_idx].strip() + ' '))
    line = [element for idx, element in enumerate(line) if not is_err[idx]] # remove error columns
    line = ['$' + element.strip() + '$' if (idx >=4) & (element.strip()!='\\nodata') else element.strip() for idx, element in enumerate(line)]
    line.insert(-1, r'\nodata')
    result_latex[i] = ' & '.join(line)

result_latex = ' \\\\\n'.join(result_latex) + ' \\\\'
headers = [key for i, key in enumerate(selected_columns) if not is_err[i]]
headers.insert(-1, r'\nodata')
print(headers)
print(result_latex)

# Not Working!!
# # mrt table

# result_qtable['APOGEE'] = MaskedColumn([result_qtable['APOGEE'][i].replace('--', '-') if ~result_qtable['APOGEE'].mask[i] else '' for i in range(len(result_qtable))], mask=result_qtable['APOGEE'].mask)

# result_mrt = Table()
# result_mrt['HC2000']        = MaskedColumn(result_df['HC2000'], mask=result_df['HC2000'].isna(), format=str, description='Identifier in [HC2000], Hillenbrand & Carpenter (2000) [2000ApJ...540..236H]')
# result_mrt['K19 ID']        = MaskedColumn(result_df['K19 ID'], mask=result_df['K19 ID'].isna(), format=str, description='Identifier in Kim et al. (2019) [2019AJ....157..109K]')
# result_mrt['APOGEE']        = MaskedColumn(result_df['APOGEE'], mask=result_df['APOGEE'].isna(), format=str, description='Identifier in APOGEE-2 DR16 [2020AJ....160..120J]')
# result_mrt['Gaia DR3']      = MaskedColumn(result_df['Gaia DR3'], mask=result_df['Gaia DR3'].isna(), format=str, description='Identifier in Gaia DR3 [2023A&A...674A...1G]')
# result_mrt['RAJ2000']       = MaskedColumn(result_df['RAJ2000'], mask=result_df['RAJ2000'].isna(), unit=u.deg, description='Right ascension in decimal degrees (J2000)')
# result_mrt['DEJ2000']       = MaskedColumn(result_df['DEJ2000'], mask=result_df['DEJ2000'].isna(), unit=u.deg, description='Declination in decimal degrees (J2000)')
# result_mrt['Teff']          = MaskedColumn(result_df['Teff'], mask=result_df['Teff'].isna(), unit=u.K, description='Effective temperature adopted for analysis in this study. Teff_NIRSPAO supplemented by Teff_APOGEE where the former is not available')
# result_mrt['e_Teff']        = MaskedColumn(result_df['e_Teff'], mask=result_df['e_Teff'].isna(), unit=u.K, description='Uncertainty of effective temperature adopted for analysis in this study. e_Teff_NIRSPAO supplemented by e_Teff_APOGEE where the former is not available')
# result_mrt['RV']            = MaskedColumn(result_df['RV'], mask=result_df['RV'].isna(), unit=u.km/u.s, description='Radial velocity adopted for analysis in this study. RV_NIRSPAO supplemented by RV_APOGEE where the former is not available')
# result_mrt['e_RV']          = MaskedColumn(result_df['e_RV'], mask=result_df['e_RV'].isna(), unit=u.km/u.s, description='Uncertainty of radial velocity adopted for analysis in this study. e_RV_NIRSPAO supplemented by e_RV_APOGEE where the former is not available')
# result_mrt['pmRA']          = MaskedColumn(result_df['pmRA'], mask=result_df['pmRA'].isna(), unit=u.mas/u.yr, description='Proper motion in right ascension adopted for analysis in this study. pmRA_K19 supplemented by pmRA_DR3 where the former is not available')
# result_mrt['e_pmRA']        = MaskedColumn(result_df['e_pmRA'], mask=result_df['e_pmRA'].isna(), unit=u.mas/u.yr, description='Uncertainty of proper motion in right ascension adopted for analysis in this study. e_pmRA_K19 supplemented by e_pmRA_DR3 where the former is not available')
# result_mrt['pmDE']          = MaskedColumn(result_df['pmDE'], mask=result_df['pmDE'].isna(), format='.5f', unit=u.mas/u.yr, description='Proper motion in declination adopted for analysis in this study. pmDE_K19 supplemented by pmDE_DR3 where the former is not available')
# result_mrt['e_pmDE']        = MaskedColumn(result_df['e_pmDE'], mask=result_df['e_pmDE'].isna(), unit=u.mas/u.yr, description='Uncertainty of proper motion in declination adopted for analysis in this study. e_pmDE_K19 supplemented by e_pmDE_DR3 where the former is not available')

# result_mrt['Teff_NIRSPAO']    = MaskedColumn(result_df['Teff_NIRSPAO'], mask=result_df['Teff_NIRSPAO'].isna(), unit=u.K, description='Effective temperature derived from NIRSPAO observation in this study')
# result_mrt['e_Teff_NIRSPAO']  = MaskedColumn(result_df['e_Teff_NIRSPAO'], mask=result_df['e_Teff_NIRSPAO'].isna(), unit=u.K, description='Uncertainty of effective temperature from NIRSPAO observation in this study')
# result_mrt['RV_NIRSPAO']    = MaskedColumn(result_df['RV_NIRSPAO'], mask=result_df['RV_NIRSPAO'].isna(), unit=u.km/u.s, description='Radial velocity derived from NIRSPAO observation in this study')
# result_mrt['e_RV_NIRSPAO']  = MaskedColumn(result_df['e_RV_NIRSPAO'], mask=result_df['e_RV_NIRSPAO'].isna(), unit=u.km/u.s, description='Uncertainty of radial velocity derived from NIRSPAO in this study')
# result_mrt['vsini_NIRSPAO']   = MaskedColumn(result_df['vsini_NIRSPAO'], mask=result_df['vsini_NIRSPAO'].isna(), unit=u.km/u.s, description='Projected rotational velocity derived from NIRSPAO observation in this study')
# result_mrt['e_vsini_NIRSPAO'] = MaskedColumn(result_df['e_vsini_NIRSPAO'], mask=result_df['e_vsini_NIRSPAO'].isna(), unit=u.km/u.s, description='Uncertainty of projected rotational velocity derived from NIRSPAO observation in this study')
# result_mrt['SNR O32']         = MaskedColumn(result_df['SNR O32'], mask=result_df['SNR O32'].isna(), unit=u.dimensionless_unscaled, description='Signal-to-noise ratio in order 32')
# result_mrt['SNR O33']         = MaskedColumn(result_df['SNR O33'], mask=result_df['SNR O33'].isna(), unit=u.dimensionless_unscaled, description='Signal-to-noise ratio in order 33')
# result_mrt['SNR O35']         = MaskedColumn(result_df['SNR O35'], mask=result_df['SNR O35'].isna(), unit=u.dimensionless_unscaled, description='Signal-to-noise ratio in order 35')
# result_mrt['Veil Param O32']  = MaskedColumn(result_df['Veil Param O32'], mask=result_df['Veil Param O32'].isna(), unit=u.dimensionless_unscaled, description='Veiling parameter in order 32. Defined as in Theissen et al. (2022) [2022ApJ...926..141T]')
# result_mrt['Veil Param O33']  = MaskedColumn(result_df['Veil Param O33'], mask=result_df['Veil Param O33'].isna(), unit=u.dimensionless_unscaled, description='Veiling parameter in order 33. Defined as in Theissen et al. (2022) [2022ApJ...926..141T]')
# result_mrt['Veil Param O35']  = MaskedColumn(result_df['Veil Param O35'], mask=result_df['Veil Param O35'].isna(), unit=u.dimensionless_unscaled, description='Veiling parameter in order 35. Defined as in Theissen et al. (2022) [2022ApJ...926..141T]')

# result_mrt['Teff_T22']        = MaskedColumn(result_df['Teff_T22'], mask=result_df['Teff_T22'].isna(), unit=u.K, description='Effective temperature from Theissen et al. (2022) [2022ApJ...926..141T]')
# result_mrt['e_Teff_T22']      = MaskedColumn(result_df['e_Teff_T22'], mask=result_df['e_Teff_T22'].isna(), unit=u.K, description='Uncertainty of effective temperature from Theissen et al. (2022) [2022ApJ...926..141T]')
# result_mrt['RV_T22']        = MaskedColumn(result_df['RV_T22'], mask=result_df['RV_T22'].isna(), unit=u.km/u.s, description='Radial velocity from Theissen et al. (2022) [2022ApJ...926..141T]')
# result_mrt['e_RV_T22']      = MaskedColumn(result_df['e_RV_T22'], mask=result_df['e_RV_T22'].isna(), unit=u.km/u.s, description='Uncertainty of radial velocity from Theissen et al. (2022) [2022ApJ...926..141T]')
# result_mrt['vsini_T22']       = MaskedColumn(result_df['vsini_T22'], mask=result_df['vsini_T22'].isna(), unit=u.km/u.s, description='Projected rotational velocity from Theissen et al. (2022) [2022ApJ...926..141T]')
# result_mrt['e_vsini_T22']     = MaskedColumn(result_df['e_vsini_T22'], mask=result_df['e_vsini_T22'].isna(), unit=u.km/u.s, description='Uncertainty of projected rotational velocity from Theissen et al. (2022) [2022ApJ...926..141T]')

# result_mrt['pmRA_K19']        = MaskedColumn(result_df['pmRA_K19'], mask=result_df['pmRA_K19'].isna(), unit=u.mas/u.yr, description='Proper motion in right ascension from Kim et al. (2019) [2019AJ....157..109K]')
# result_mrt['e_pmRA_K19']      = MaskedColumn(result_df['e_pmRA_K19'], mask=result_df['e_pmRA_K19'].isna(), unit=u.mas/u.yr, description='Uncertainty of proper motion in right ascension from Kim et al. (2019) [2019AJ....157..109K]')
# result_mrt['pmDE_K19']        = MaskedColumn(result_df['pmDE_K19'], mask=result_df['pmDE_K19'].isna(), unit=u.mas/u.yr, description='Proper motion in declination from Kim et al. (2019) [2019AJ....157..109K]')
# result_mrt['e_pmDE_K19']      = MaskedColumn(result_df['e_pmDE_K19'], mask=result_df['e_pmDE_K19'].isna(), unit=u.mas/u.yr, description='Uncertainty of proper motion in declination from Kim et al. (2019) [2019AJ....157..109K]')

# result_mrt['pmRA_DR3']        = MaskedColumn(result_df['pmRA_DR3'], mask=result_df['pmRA_DR3'].isna(), unit=u.mas/u.yr, description='Proper motion in right ascension from Gaia DR3 [2023A&A...674A...1G]')
# result_mrt['e_pmRA_DR3']      = MaskedColumn(result_df['e_pmRA_DR3'], mask=result_df['e_pmRA_DR3'].isna(), unit=u.mas/u.yr, description='Uncertainty of proper motion in right ascension from Gaia DR3 [2023A&A...674A...1G]')
# result_mrt['pmDE_DR3']        = MaskedColumn(result_df['pmDE_DR3'], mask=result_df['pmDE_DR3'].isna(), unit=u.mas/u.yr, description='Proper motion in declination from Gaia DR3 [2023A&A...674A...1G]')
# result_mrt['e_pmDE_DR3']      = MaskedColumn(result_df['e_pmDE_DR3'], mask=result_df['e_pmDE_DR3'].isna(), unit=u.mas/u.yr, description='Uncertainty of proper motion in declination from Gaia DR3 [2023A&A...674A...1G]')

# bibcodes = ['2016ApJS..222....8D, 2016ApJ...823..102C', '2015A&A...577A..42B', '2016A&A...593A..99F', '1999ApJ...525..772P']
# for model_name, bibcode in zip(['MIST','BHAC15', 'Feiden', 'Palla'], bibcodes):
#     result_qtable[f'M_{model_name}'].description    = f'Stellar mass based on {model_name} model [{bibcode}]'
#     result_qtable[f'e_M_{model_name}'].description  = f'Stellar mass uncertainty based on {model_name} model [{bibcode}]'

# result_qtable.write(f'{user_path}/ONC/starrynight/catalogs/result_mrt.ecsv', overwrite=True)
# # result_qtable.write(f'{user_path}/ONC/starrynight/catalogs/result_mrt.dat', format='ascii.mrt', overwrite=True)
# result_qtable.write(sys.stdout, format='ascii.mrt')