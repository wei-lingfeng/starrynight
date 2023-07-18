import os, sys
import numpy as np
import pandas as pd
from numpy import ma
from astropy import units as u
from astropy.table import QTable, Column, MaskedColumn

user_path = os.path.expanduser('~')

month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sources = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog - new.csv')
hc2000 = sources.loc[~sources.HC2000.isna() & sources.theta_orionis.isna()]

result = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/sources post-processing.csv', dtype={'ID_gaia': str, 'ID_kim': str})
trapezium_only = ~result.theta_orionis.isna() & result.obs_date.isna() & result.APOGEE.isna()
result = result.loc[~trapezium_only].reset_index(drop=True)
result.ID_kim = result.ID_kim.fillna('snodata')

obs_df = pd.DataFrame({
    'HC2000 ID': [f"HC2000 {ID:.0f}{m_ID}" for ID, m_ID in zip(hc2000['HC2000'], hc2000['m_HC2000'].fillna(''))],
    'Obs. Date': [f"{date.split('-')[0]} {month_list[int(date.split('-')[1]) - 1]} {date.split('-')[2]}" for date in hc2000['obs_date']],
    'No. of Frames': [len(eval(_)) for _ in hc2000.sci_frames],
    'Int. Time': [int(_) for _ in hc2000.itime]
})

result_df = pd.DataFrame({
    'HC2000 ID':        ma.array([f"HC2000 {result.HC2000[i]:.0f}{result.m_HC2000[i]}" if ~result.m_HC2000.isna()[i] else f"HC2000 {result.HC2000[i]:.0f}" for i in range(len(result))], mask=result.HC2000.isna()).filled('snodata'),
    'K19 ID':           result.ID_kim,
    'RAJ2000':          result.RAJ2000,
    'DEJ2000':          result.DEJ2000,
    'Rvel_NIRSPAO':     result.rv_,
    'e_rv':             result.e_rv,
    'pmRA':             result.pmRA,
    'e_pmRA':           result.e_pmRA,
    'pmDE' :            result.pmDE,
    'e_pmDE':           result.e_pmDE,
    'teff':             result.teff,
    'e_teff':           result.e_teff,
    'vsini':            result.vsini_nirspao,
    'e_vsini':          result.e_vsini_nirspao,
    'mass_MIST':        result.mass_MIST,
    'e_mass_MIST':      result.e_mass_MIST,
    'mass_BHAC15':      result.mass_BHAC15,
    'e_mass_BHAC15':    result.e_mass_BHAC15,
    'mass_Feiden':      result.mass_Feiden,
    'e_mass_Feiden':    result.e_mass_Feiden,
    'mass_Palla':       result.mass_Palla,
    'e_mass_Palla':     result.e_mass_Palla,
    'veiling param O32':    result.veiling_param_O32,
    'veiling param O33':    result.veiling_param_O33_nirspao,
    'veiling param O35':    result.veiling_param_O35,
    'SNR O32':          result.snr_O32,
    'SNR O33':          result.snr_O33,
    'SNR O35':          result.snr_O35
})

result_rounded = result_df.round({
    'rv':2, 'e_rv':2, 'pmRA':2, 'e_pmRA':2, 'pmDE':2, 'e_pmDE':2, 
    'teff':2, 'e_teff':2, 'vsini':2, 'e_vsini':2,
    'mass_MIST':2, 'e_mass_MIST':2, 'mass_BHAC15':2, 'e_mass_BHAC15':2, 
    'mass_Feiden':2, 'e_mass_Feiden':2, 'mass_Palla':2, 'e_mass_Palla':2,
    'veiling param O32':2, 'veiling param O33':2, 'veiling param O35':2, 
    'SNR O32':2, 'SNR O33':2, 'SNR O35':2
})


# save latex
print('Observation Table:')
obs_latex = obs_df.to_latex(index=False, header=obs_df.keys().to_list(), na_rep='snodata')
obs_latex.replace('snodata', r'\nodata')
print(obs_latex)

selected_columns = ['HC2000 ID', 'K19 ID', 'RAJ2000', 'DEJ2000', 'rv', 'e_rv', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE', 'teff', 'e_teff', 'vsini', 'e_vsini', 'mass_MIST', 'e_mass_MIST', 'SNR O32', 'SNR O33', 'SNR O35']
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
    line.insert(-3, r'\nodata')
    result_latex[i] = ' & '.join(line)

result_latex = ' \\\\\n'.join(result_latex) + ' \\\\'
headers = [key for i, key in enumerate(selected_columns) if not is_err[i]]
headers.insert(-3, r'\nodata')
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

result_table = QTable.read('/home/weilingfeng/ONC/starrynight/catalogs/sources post-processing.ecsv')
trapezium_only = (result_table['sci_frames'].mask) & (result_table['APOGEE'].mask)
result_table = result_table[~trapezium_only]
for key in result_table.keys():
    if result_table[key].dtype.name.startswith('str') or result_table[key].dtype.name.startswith('int') or result_table[key].dtype.name.startswith('object'):
        continue
    try:
        result_table[key] = MaskedColumn(result_table[key], mask=np.isnan(result_table[key]))
    except:
        pass
result_table.rename_columns(
    ['obs_date', 'ID_kim', 'rv_helio', 'e_rv_nirspao', 'rv_chris', 'e_rv_chris', 'teff_nirspao', 'e_teff_nirspao', 'teff_apogee', 'e_teff_apogee', 'snr_O32', 'snr_O33', 'snr_O35',  'veiling_param_O32', 'veiling_param_O33_nirspao', 'veiling_param_O35'], 
    ['Obs. Date', 'K19 ID', 'RVel', 'e_RVel', 'RVel_T22', 'e_RVel_T22', 'Teff_NIRSPAO', 'e_Teff_NIRSPAO', 'Teff_T22', 'e_Teff_T22', 'SNR O32', 'SNR O33', 'SNR O35', 'Veil Param O32', 'Veil Param O33', 'Veil Param O35']
)
for model_name in ['MIST','BHAC15', 'Feiden', 'Palla']:
    result_table.rename_columns([f'mass_{model_name}', f'e_mass_{model_name}'], [f'M_{model_name}', f'e_M_{model_name}'])

result_table['HC2000'].description='Identifier in [HC2000], Hillenbrand & Carpenter (2000) [2000ApJ...540..236H]'
result_table['K19 ID'].description='Identifier in Kim et al. (2019) [2019AJ....157..109K]'
result_table['RAJ2000'].description='Right ascension in decimal degrees (J2000)'
result_table['DEJ2000'].description='Declination in decimal degrees (J2000)'
result_table['RVel'].description='Radial Velocity'
# result_table['e_RVel']      = Column(result_df.e_rv,    unit=u.km/u.s, description='Radial Velocity uncertainty')
# result_table['  pmRA']      = Column(result_df.pmRA,    unit=u.mas/u.yr, description='Proper motion in Right ascension')
# result_table['e_pmRA']      = Column(result_df.e_pmRA,  unit=u.mas/u.yr, description='Proper motion uncertainty in right ascension')
# result_table['  pmDE']      = Column(result_df.pmDE,    unit=u.mas/u.yr, description='Proper motion in declination')
# result_table['e_pmDE']      = Column(result_df.e_pmDE,  unit=u.mas/u.yr, description='Proper motion uncertainty in declination')
# result_table['  teff']      = Column(result_df.teff,    unit=u.K, description='Effective temperature')
# result_table['e_teff']      = Column(result_df.e_teff,  unit=u.K, description='Effective temperature uncertainty')
# result_table['  vsini']     = Column(result_df.vsini,   unit=u.km/u.s, description='Rotational velocity')
# result_table['e_vsini']     = Column(result_df.e_vsini, unit=u.km/u.s, description='Rotational velocity uncertainty')
# for model_name in ['MIST','BHAC15', 'Feiden', 'Palla']:
#     result_table[f'M_{model_name}']   = Column(result_df[f'mass_{model_name}'], unit=u.Msun, description=f'Stellar mass based on {model_name} model')
#     result_table[f'e_M_{model_name}'] = Column(result_df[f'e_mass_{model_name}'], unit=u.Msun, description=f'Stellar mass uncertainty based on {model_name} model')


# result_table.write(f'{user_path}/ONC/starrynight/catalogs/result_table.dat', format='ascii.mrt', overwrite=True)
result_table.write(sys.stdout, format='ascii.mrt')
