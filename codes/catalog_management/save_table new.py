import os, sys
import numpy as np
import pandas as pd
from numpy import ma
from astropy import units as u
from astropy.table import Table, QTable, Column, MaskedColumn

user_path = os.path.expanduser('~')

month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sources = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog - new.csv')
hc2000 = sources.loc[~sources.HC2000.isna() & sources.theta_orionis.isna()]

result = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/sources post-processing.csv', dtype={'Gaia DR3': str, 'ID_kim': str})
trapezium_only = ~result.theta_orionis.isna() & result.obs_date.isna() & result.APOGEE.isna()
result = result.loc[~trapezium_only].reset_index(drop=True)

obs_df = pd.DataFrame({
    'HC2000 ID': [f"HC2000 {ID:.0f}{m_ID}" for ID, m_ID in zip(hc2000['HC2000'], hc2000['m_HC2000'].fillna(''))],
    'Obs. Date': [f"{date.split('-')[0]} {month_list[int(date.split('-')[1]) - 1]} {date.split('-')[2]}" for date in hc2000['obs_date']],
    'No. of Frames': [len(eval(_)) for _ in hc2000.sci_frames],
    'Int. Time': [int(_) for _ in hc2000.itime]
})

result_df = pd.DataFrame({
    'HC2000':           result.HC2000,
    'K19 ID':           result.ID_kim,
    'APOGEE':           result.APOGEE,
    'Gaia DR3':         result['Gaia DR3'], 
    'RAJ2000':          result.RAJ2000,
    'DEJ2000':          result.DEJ2000,
    'Obs. Date':        result.obs_date,
    'Teff':             result.teff,
    'e_Teff':           result.e_teff,
    'RVel':             result.rv,
    'e_RVel':           result.e_rv,
    'pmRA':             result.pmRA,
    'e_pmRA':           result.e_pmRA,
    'pmDE' :            result.pmDE,
    'e_pmDE':           result.e_pmDE,
    
    'Teff_NIRSPAO':     result.teff_nirspao,
    'e_Teff_NIRSPAO':   result.e_teff_nirspao,
    'RVel_NIRSPAO':     result.rv_helio,
    'e_RVel_NIRSPAO':   result.e_rv_nirspao,
    'vsini_NIRSPAO':    result.vsini_nirspao,
    'e_vsini_NIRSPAO':  result.e_vsini_nirspao,
    'SNR O32':          result.snr_O32,
    'SNR O33':          result.snr_O33,
    'SNR O35':          result.snr_O35,
    'Veil Param O32':   result.veiling_param_O32,
    'Veil Param O33':   result.veiling_param_O33_nirspao,
    'Veil Param O35':   result.veiling_param_O35,
    
    'Teff_T22':         result.teff_apogee,
    'e_Teff_T22':       result.e_teff_apogee,
    'RVel_T22':         result.rv_apogee,
    'e_RVel_T22':       result.e_rv_apogee,
    'vsini_T22':        result.vsini_apogee,
    'e_vsini_T22':      result.e_vsini_apogee,
    
    'pmRA_K19':         result.pmRA_kim,
    'e_pmRA_K19':       result.e_pmRA_kim,
    'pmDE_K19':         result.pmDE_kim,
    'e_pmDE_K19':       result.e_pmDE_kim,
    
    'pmRA_DR3':         result.pmRA_gaia,
    'e_pmRA_DR3':       result.e_pmRA_gaia,
    'pmDE_DR3':         result.pmDE_gaia,
    'e_pmDE_DR3':       result.e_pmDE_gaia,
    
    'M_MIST':           result.mass_MIST,
    'e_M_MIST':         result.e_mass_MIST,
    'M_BHAC15':         result.mass_BHAC15,
    'e_M_BHAC15':       result.e_mass_BHAC15,
    'M_Feiden':         result.mass_Feiden,
    'e_M_Feiden':       result.e_mass_Feiden,
    'M_Palla':          result.mass_Palla,
    'e_M_Palla':        result.e_mass_Palla,
})

rounding = {}
for key in result_df.keys():
    if key not in ['HC2000', 'K19 ID', 'APOGEE', 'Gaia DR3', 'RAJ2000', 'DEJ2000', 'Obs. Date']:
        rounding[key] = 2
    
result_rounded = result_df.round(rounding)
result_rounded['K19 ID'] = result_rounded['K19 ID'].fillna('snodata')
result_rounded['HC2000'] = ma.array([f"HC2000 {result.HC2000[i]:.0f}{result.m_HC2000[i]}" if ~result.m_HC2000.isna()[i] else f"HC2000 {result.HC2000[i]:.0f}" for i in range(len(result))], mask=result.HC2000.isna()).filled('snodata')

# save latex
print('Observation Table:')
obs_latex = obs_df.to_latex(index=False, header=obs_df.keys().to_list(), na_rep='snodata')
obs_latex.replace('snodata', r'\nodata')
print(obs_latex)

selected_columns = list(result_df.keys())[:15] + ['M_MIST', 'e_M_MIST']
selected_columns.remove('Obs. Date')

print('Result Table:')
result_latex = result_rounded.loc[
    :20, selected_columns
].to_latex(index=False, header=selected_columns, na_rep='snodata')
result_latex = result_latex.replace('snodata', r'\nodata')

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
    line = ['$' + element.strip() + '$' if (idx >=2) & (element.strip()!='\\nodata') else element.strip() for idx, element in enumerate(line)]
    line.insert(-1, r'\nodata')
    result_latex[i] = ' & '.join(line)

result_latex = ' \\\\\n'.join(result_latex) + ' \\\\'
headers = [key for i, key in enumerate(selected_columns) if not is_err[i]]
headers.insert(-1, r'\nodata')
print(headers)
print(result_latex)


# save mrt
# obs_table = QTable()
# obs_table['HC2000 ID'] = Column(obs_df['HC2000 ID'], format=str, description='Identifier in [HC2000], Hillenbrand & Carpenter (2000) [2000ApJ...540..236H]')
# obs_table['Obs Time'] = Column(obs_df['Obs. Date'], format=str, description='Observation Date')
# obs_table['No. of Frames'] = Column(obs_df['No. of Frames'], description='Number of Frames')
# obs_table['Int. Time'] = Column(obs_df['Int. Time'], unit=u.s, description='Integration Time of each frame')
# # obs_table.write(f'{user_path}/ONC/starrynight/catalogs/obs_table.dat', format='ascii.mrt', overwrite=True)
# obs_table.write(sys.stdout, format='ascii.mrt')


result_table = Table()
result_table['HC2000']          = Column(ma.array([f"HC2000 {result.HC2000[i]:.0f}{result.m_HC2000[i]}" if ~result.m_HC2000.isna()[i] else f"HC2000 {result.HC2000[i]:.0f}" for i in range(len(result))], mask=result.HC2000.isna()).filled(''), format=str, description='Identifier in [HC2000], Hillenbrand & Carpenter (2000) [2000ApJ...540..236H]')
result_table['K19 ID']          = Column(result_df['K19 ID'].fillna(''), format=str, description='Identifier in Kim et al. (2019) [2019AJ....157..109K]')
result_table['APOGEE']          = Column(result_df['APOGEE'].fillna(''), format=str, description='Identifier in APOGEE-2 DR16 [2020AJ....160..120J]')
result_table['Gaia DR3']        = Column(result_df['Gaia DR3'].fillna(''), format=str, description='Identifier in Gaia DR3 [2023A&A...674A...1G]')
result_table['RAJ2000']         = Column(result_df['RAJ2000'], unit=u.deg, description='Right ascension in decimal degrees (J2000)')
result_table['DEJ2000']         = Column(result_df['DEJ2000'], unit=u.deg, description='Declination in decimal degrees (J2000)')
result_table['Teff']            = Column(result_df['Teff'], unit=u.K, description='Effective temperature adopted for analysis in this study. Teff_NIRSPAO supplemented by Teff_APOGEE where the former is not available')
result_table['e_Teff']          = Column(result_df['e_Teff'], unit=u.K, description='Effective temperature uncertainty adopted for analysis in this study. e_Teff_NIRSPAO supplemented by e_Teff_APOGEE where the former is not available')
result_table['RVel']            = Column(result_df['RVel'], unit=u.km/u.s, description='Radial velocity adopted for analysis in this study. RVel_NIRSPAO supplemented by RVel_APOGEE where the former is not available')
result_table['e_RVel']          = Column(result_df['e_RVel'], unit=u.km/u.s, description='Radial velocity uncertainty adopted for analysis in this study. e_RVel_NIRSPAO supplemented by e_RVel_APOGEE where the former is not available')
result_table['pmRA']            = Column(result_df['pmRA'], unit=u.mas/u.yr, description='Proper motion in right ascension adopted for analysis in this study. pmRA_K19 supplemented by pmRA_DR3 where the former is not available')
result_table['e_pmRA']          = Column(result_df['e_pmRA'], unit=u.mas/u.yr, description='Proper motion uncertainty in right ascension adopted for analysis in this study. e_pmRA_K19 supplemented by e_pmRA_DR3 where the former is not available')
result_table['pmDE']            = Column(result_df['pmDE'], unit=u.mas/u.yr, description='Proper motion in declination adopted for analysis in this study. pmDE_K19 supplemented by pmDE_DR3 where the former is not available')
result_table['e_pmDE']          = Column(result_df['e_pmDE'], unit=u.mas/u.yr, description='Proper motion uncertainty in declination adopted for analysis in this study. e_pmDE_K19 supplemented by e_pmDE_DR3 where the former is not available')

result_table['Teff_NIRSPAO']    = Column(result_df['Teff_NIRSPAO'], unit=u.K, description='Effective temperature derived from NIRSPAO observation in this study')
result_table['e_Teff_NIRSPAO']  = Column(result_df['e_Teff_NIRSPAO'], unit=u.K, description='Effective temperature uncertainty from NIRSPAO observation in this study')
result_table['RVel_NIRSPAO']    = Column(result_df['RVel_NIRSPAO'], unit=u.km/u.s, description='Radial velocity derived from NIRSPAO observation in this study')
result_table['e_RVel_NIRSPAO']  = Column(result_df['e_RVel_NIRSPAO'], unit=u.km/u.s, description='Radial velocity uncertainty derived from NIRSPAO in this study')
result_table['vsini_NIRSPAO']   = Column(result_df['vsini_NIRSPAO'], unit=u.km/u.s, description='Projected rotational velocity derived from NIRSPAO observation in this study')
result_table['e_vsini_NIRSPAO'] = Column(result_df['e_vsini_NIRSPAO'], unit=u.km/u.s, description='Projected rotational velocity uncertainty derived from NIRSPAO observation in this study')
result_table['SNR O32']         = Column(result_df['SNR O32'], unit=u.dimensionless_unscaled, description='Signal-to-noise ratio in order 32')
result_table['SNR O33']         = Column(result_df['SNR O33'], unit=u.dimensionless_unscaled, description='Signal-to-noise ratio in order 33')
# result_table['SNR O35']         = Column(result_df['SNR O35'], unit=u.dimensionless_unscaled, description='Signal-to-noise ratio in order 35')
result_table['Veil Param O32']  = Column(result_df['Veil Param O32'], unit=u.dimensionless_unscaled, description='Veiling parameter in order 32. Defined as in Theissen et al. (2022) [2022ApJ...926..141T]')
result_table['Veil Param O33']  = Column(result_df['Veil Param O33'], unit=u.dimensionless_unscaled, description='Veiling parameter in order 32. Defined as in Theissen et al. (2022) [2022ApJ...926..141T]')
# result_table['Veil Param O35']  = Column(result_df['Veil Param O35'], unit=u.dimensionless_unscaled, description='Veiling parameter in order 32. Defined as in Theissen et al. (2022) [2022ApJ...926..141T]')

result_table['Teff_T22']        = Column(result_df['Teff_T22'], unit=u.K, description='Effective temperature from Theissen et al. (2022) [2022ApJ...926..141T]')
result_table['e_Teff_T22']      = Column(result_df['e_Teff_T22'], unit=u.K, description='Effective temperature uncertainty from Theissen et al. (2022) [2022ApJ...926..141T]')
result_table['RVel_T22']        = Column(result_df['RVel_T22'], unit=u.km/u.s, description='Radial velocity from Theissen et al. (2022) [2022ApJ...926..141T]')
result_table['e_RVel_T22']      = Column(result_df['e_RVel_T22'], unit=u.km/u.s, description='Radial velocity uncertainty from Theissen et al. (2022) [2022ApJ...926..141T]')
result_table['vsini_T22']       = Column(result_df['vsini_T22'], unit=u.km/u.s, description='Projected rotational velocity from Theissen et al. (2022) [2022ApJ...926..141T]')
result_table['e_vsini_T22']     = Column(result_df['e_vsini_T22'], unit=u.km/u.s, description='Projected rotational velocity uncertainty from Theissen et al. (2022) [2022ApJ...926..141T]')

result_table['pmRA_K19']        = Column(result_df['pmRA_K19'], unit=u.mas/u.yr, description='Proper motion in right ascension from Kim et al. (2019) [2019AJ....157..109K]')
result_table['e_pmRA_K19']      = Column(result_df['e_pmRA_K19'], unit=u.mas/u.yr, description='Proper motion uncertainty in right ascension from Kim et al. (2019) [2019AJ....157..109K]')
result_table['pmDE_K19']        = Column(result_df['pmDE_K19'], unit=u.mas/u.yr, description='Proper motion in declination from Kim et al. (2019) [2019AJ....157..109K]')
result_table['e_pmDE_K19']      = Column(result_df['e_pmDE_K19'], unit=u.mas/u.yr, description='Proper motion uncertainty in declination from Kim et al. (2019) [2019AJ....157..109K]')

result_table['pmRA_DR3']        = Column(result_df['pmRA_DR3'], unit=u.mas/u.yr, description='Proper motion in right ascension from Gaia DR3 [2023A&A...674A...1G]')
result_table['e_pmRA_DR3']      = Column(result_df['e_pmRA_DR3'], unit=u.mas/u.yr, description='Proper motion uncertainty in right ascension from Gaia DR3 [2023A&A...674A...1G]')
result_table['pmDE_DR3']        = Column(result_df['pmDE_DR3'], unit=u.mas/u.yr, description='Proper motion in declination from Gaia DR3 [2023A&A...674A...1G]')
result_table['e_pmDE_DR3']      = Column(result_df['e_pmDE_DR3'], unit=u.mas/u.yr, description='Proper motion uncertainty in declination from Gaia DR3 [2023A&A...674A...1G]')

bibcodes = ['2016ApJS..222....8D, 2016ApJ...823..102C', '2015A&A...577A..42B', '2016A&A...593A..99F', '1999ApJ...525..772P']
for model_name, bibcode in zip(['MIST','BHAC15', 'Feiden', 'Palla'], bibcodes):
    result_table[f'M_{model_name}'] = Column(result_df[f'M_{model_name}'], unit=u.solMass, description=f'Stellar mass based on {model_name} model [{bibcode}]')
    result_table[f'e_M_{model_name}'] = Column(result_df[f'e_M_{model_name}'], unit=u.solMass, description=f'Stellar mass uncertainty based on {model_name} model [{bibcode}]')


# result_table.write(f'{user_path}/ONC/starrynight/catalogs/result_table.dat', format='ascii.mrt', overwrite=True)
# result_table.write(sys.stdout, format='ascii.mrt')
