import os
import pandas as pd
from astropy.io import ascii
from astropy.table import Table

user_path = os.path.expanduser('~')

month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sources = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog.csv')
hc2000 = sources.loc[~sources.HC2000.isna() & sources.theta_orionis.isna()]

sources_2d = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/sources 2d.csv')
sources_2d = sources_2d.loc[sources_2d.theta_orionis.isna()]

observation_table = pd.DataFrame({
    'HC2000 ID': ['HC2000 {}'.format(_.replace('_', ' ')) for _ in list(hc2000.HC2000)],
    'Obs. Date': ['20{} {} {}'.format(str(int(year)).zfill(2), month_list[int(month)-1], int(day)) for year, month, day in zip(hc2000.year, hc2000.month, hc2000.day)],
    'No. of Frames': [len(eval(_)) for _ in hc2000.sci_frames],
    'Int. Time': [int(_) for _ in hc2000.itime]
})
observation_table = Table.from_pandas(observation_table)

result_table = pd.DataFrame({
    'HC2000 ID': [f"HC2000 {_.replace('_', ' ')}" if isinstance(_, str) else '' for _ in list(sources_2d.HC2000)],
    'K19 ID': sources_2d.ID_kim,
    'APOGEE ID': sources_2d.ID_kim,
    'RAJ2000': sources_2d._RAJ2000,
    'DEJ2000': sources_2d._DEJ2000,
    'rv': sources_2d.rv,
    'rv_e': sources_2d.rv_e,
    'pmRA': sources_2d.pmRA,
    'pmRA_e': sources_2d.pmRA_e,
    'pmDE_e': sources_2d.pmDE_e,
    'teff': sources_2d.teff,
    'teff_e': sources_2d.teff_e,
    'vsini': sources_2d.vsini,
    'veiling param': sources_2d.veiling_param_O33
})
result_table = Table.from_pandas(result_table)

ascii.write(observation_table[:5], format='latex')
ascii.write(result_table[:5], format='latex')