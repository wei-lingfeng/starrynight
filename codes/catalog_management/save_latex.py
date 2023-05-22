import os
import numpy as np
import pandas as pd
from functools import reduce

user_path = os.path.expanduser('~')

month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sources = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog.csv')
hc2000 = sources.loc[~sources.HC2000.isna() & sources.theta_orionis.isna()]

sources_2d = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/sources 2d.csv', dtype={'ID_gaia': str, 'ID_kim': str})
sources_2d = sources_2d.loc[sources_2d.theta_orionis.isna()]

obs_table = pd.DataFrame({
    'HC2000 ID': ['HC2000 {}'.format(_.replace('_', '')) for _ in list(hc2000.HC2000)],
    'Obs. Date': ['20{} {} {}'.format(str(int(year)).zfill(2), month_list[int(month)-1], int(day)) for year, month, day in zip(hc2000.year, hc2000.month, hc2000.day)],
    'No. of Frames': [len(eval(_)) for _ in hc2000.sci_frames],
    'Int. Time': [int(_) for _ in hc2000.itime]
})

result_table = pd.DataFrame({
    'HC2000 ID': [f"HC2000 {_.replace('_', '')}" if isinstance(_, str) else '' for _ in list(sources_2d.HC2000)],
    'K19 ID':   sources_2d.ID_kim,
    'RAJ2000':  sources_2d.RAJ2000,
    'DEJ2000':  sources_2d.DEJ2000,
    'rv':       sources_2d.rv,
    'rv_e':     sources_2d.rv_e,
    'pmRA':     sources_2d.pmRA,
    'pmRA_e':   sources_2d.pmRA_e,
    'pmDE' :    sources_2d.pmDE,
    'pmDE_e':   sources_2d.pmDE_e,
    'teff':     sources_2d.teff,
    'teff_e':   sources_2d.teff_e,
    'vsini':    sources_2d.vsini,
    'vsini_e':  sources_2d.vsini_e,
    'mass_MIST': sources_2d.mass_MIST,
    'mass_e_MIST': sources_2d.mass_e_MIST,
    # 'mass_BHAC15': sources_2d.mass_BHAC15,
    # 'mass_e_BHAC15': sources_2d.mass_e_BHAC15,
    # 'mass_Feiden': sources_2d.mass_Feiden,
    # 'mass_e_Feiden': sources_2d.mass_e_Feiden,
    # 'mass_Palla': sources_2d.mass_Palla,
    # 'mass_e_Palla': sources_2d.mass_e_Palla,
    'veiling param': sources_2d.veiling_param_O33,
    'SNR O33':          sources_2d.snr_O33
}).round({
    'rv':2, 'rv_e':2, 'pmRA':2, 'pmRA_e':2, 'pmDE':2, 'pmDE_e':2, 
    'teff':2, 'teff_e':2, 'vsini':2, 'vsini_e':2,
    'mass_MIST':2, 'mass_e_MIST':2, 
    # 'mass_BHAC15':2, 'mass_e_BHAC15':2, 
    # 'mass_Feiden':2, 'mass_e_Feiden':2, 'mass_Palla':2, 'mass_e_Palla':2,
    'veiling param':2, 'SNR O33':2
})

print('Observation Table:')
obs_latex = obs_table.head(20).to_latex(index=False, header=obs_table.keys().to_list(), na_rep='snodata')
# obs_latex.replace('snodata', r'\nodata')
# print(obs_latex)

print('Result Table:')
result_latex = result_table.head(20).to_latex(index=False, header=result_table.keys().to_list(), na_rep='snodata')
result_latex = result_latex.replace('snodata', r'\nodata')

is_err = [True if '_e' in key else False for key in result_table.keys()]
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
print([key for i, key in enumerate(result_table.keys()) if not is_err[i]])
print(result_latex)