import os
import astropy
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

results_table = pd.DataFrame({
    'HC2000 ID': ['HC2000 {}'.format(_.replace('_', ' ')) for _ in list(sources_2d.HC2000)],
    'K19 ID': sources_2d.ID_kim,
    'APOGEE ID': sources_2d.ID_kim,
    'RAJ2000': sources_2d._RAJ2000
})
ascii.write(observation_table[:5], format='latex')