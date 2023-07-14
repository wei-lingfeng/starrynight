import os, sys
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table, Column

user_path = os.path.expanduser('~')

month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sources = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog - new.csv')
hc2000 = sources.loc[~sources.HC2000.isna() & sources.theta_orionis.isna()]

result = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/sources post-processing.csv', dtype={'ID_gaia': str, 'ID_kim': str})
result = result.loc[result.theta_orionis.isna()]
result.ID_kim = result.ID_kim.fillna('')

obs_df = pd.DataFrame({
    'HC2000 ID': [f"HC2000 {ID:.0f}{m_ID}" for ID, m_ID in zip(hc2000['HC2000'], hc2000['m_HC2000'].fillna(''))],
    'Obs. Date': [f"{date.split('-')[0]} {month_list[int(date.split('-')[1]) - 1]} {date.split('-')[2]}" for date in hc2000['obs_date']],
    'No. of Frames': [len(eval(_)) for _ in hc2000.sci_frames],
    'Int. Time': [int(_) for _ in hc2000.itime]
})

result_df = pd.DataFrame({
    'HC2000 ID': [f"HC2000 {_.replace('_', '')}" if isinstance(_, str) else '' for _ in list(result.HC2000)],
    'K19 ID':           result.ID_kim,
    'RAJ2000':          result.RAJ2000,
    'DEJ2000':          result.DEJ2000,
    'rv':               result.rv,
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
    'veiling param':    result.veiling_param_O33_nirspao,
    'SNR O33':          result.snr_O33
})

result_rounded = result_df.round({
    'rv':2, 'rv_e':2, 'pmRA':2, 'pmRA_e':2, 'pmDE':2, 'pmDE_e':2, 
    'teff':2, 'teff_e':2, 'vsini':2, 'vsini_e':2,
    'mass_MIST':2, 'mass_e_MIST':2, 'mass_BHAC15':2, 'mass_e_BHAC15':2, 
    'mass_Feiden':2, 'mass_e_Feiden':2, 'mass_Palla':2, 'mass_e_Palla':2,
    'veiling param':2, 'SNR O33':2
})


# save latex
print('Observation Table:')
obs_latex = obs_df.to_latex(index=False, header=obs_df.keys().to_list(), na_rep='snodata')
obs_latex.replace('snodata', r'\nodata')
print(obs_latex)

print('Result Table:')
result_latex = result_rounded.head(20).to_latex(index=False, header=result_rounded.keys().to_list(), na_rep='snodata')
result_latex = result_latex.replace('snodata', r'\nodata')

is_err = [True if '_e' in key else False for key in result_rounded.keys()]
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
    result_latex[i] = ' & '.join(line)

result_latex = ' \\\\\n'.join(result_latex) + ' \\\\'
print([key for i, key in enumerate(result_rounded.keys()) if not is_err[i]])
print(result_latex)


# save mrt
obs_table = Table()
obs_table['HC2000 ID'] = Column(obs_df['HC2000 ID'], format=str, description='Identifier in [HC2000], Hillenbrand & Carpenter (2000) [2000ApJ...540..236H]')
obs_table['Obs Time'] = Column(obs_df['Obs. Date'], format=str, description='Observation Date')
obs_table['No. of Frames'] = Column(obs_df['No. of Frames'], description='Number of Frames')
obs_table['Int. Time'] = Column(obs_df['Int. Time'], unit=u.s, description='Integration Time of each frame')
# obs_table.write(f'{user_path}/ONC/starrynight/catalogs/obs_table.dat', format='ascii.mrt', overwrite=True)
obs_table.write(sys.stdout, format='ascii.mrt')

result_table = Table()
result_table['  HC2000 ID'] = Column(result_df['HC2000 ID'], format=str, description='Identifier in [HC2000], Hillenbrand & Carpenter (2000) [2000ApJ...540..236H]')
result_table['  K19 ID']    = Column(result_df['K19 ID'], format=str, description='Identifier in Kim et al. (2019) [2019AJ....157..109K]')
result_table['  RAJ2000']   = Column(result_df.RAJ2000, unit=u.deg, description='Right ascension in decimal degrees (J2000)')
result_table['  DEJ2000']   = Column(result_df.DEJ2000, unit=u.deg, description='Declination in decimal degrees (J2000)')
result_table['  RVel']      = Column(result_df.rv,      unit=u.km/u.s, description='Radial Velocity')
result_table['e_RVel']      = Column(result_df.e_rv,    unit=u.km/u.s, description='Radial Velocity uncertainty')
result_table['  pmRA']      = Column(result_df.pmRA,    unit=u.mas/u.yr, description='Proper motion in Right ascension')
result_table['e_pmRA']      = Column(result_df.e_pmRA,  unit=u.mas/u.yr, description='Proper motion uncertainty in right ascension')
result_table['  pmDE']      = Column(result_df.pmDE,    unit=u.mas/u.yr, description='Proper motion in declination')
result_table['e_pmDE']      = Column(result_df.e_pmDE,  unit=u.mas/u.yr, description='Proper motion uncertainty in declination')
result_table['  teff']      = Column(result_df.teff,    unit=u.K, description='Effective temperature')
result_table['e_teff']      = Column(result_df.e_teff,  unit=u.K, description='Effective temperature uncertainty')
result_table['  vsini']     = Column(result_df.vsini,   unit=u.km/u.s, description='Rotational velocity')
result_table['e_vsini']     = Column(result_df.e_vsini, unit=u.km/u.s, description='Rotational velocity uncertainty')
for model_name in ['MIST','BHAC15', 'Feiden', 'Palla']:
    result_table[f'M_{model_name}']   = Column(result_df[f'mass_{model_name}'], unit=u.Msun, description=f'Stellar mass based on {model_name} model')
    result_table[f'e_M_{model_name}'] = Column(result_df[f'mass_e_{model_name}'], unit=u.Msun, description=f'Stellar mass uncertainty based on {model_name} model')


# result_table.write(f'{user_path}/ONC/starrynight/catalogs/result_table.dat', format='ascii.mrt', overwrite=True)
result_table.write(sys.stdout, format='ascii.mrt')
