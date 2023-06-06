import pandas as pd
sources = pd.read_csv('/home/l3wei/ONC/Catalogs/synthetic catalog.csv', dtype={'ID_gaia': str})
sort_key = 'model_dip_O33'
sources.sort_values(sort_key).head(20).loc[:, ['HC2000', 'year', 'month', 'day', 'sci_frames', 'tel_frames', 'teff', 'teff_e', 'teff_apogee', 'teff_e_apogee', 'teff_kounkel', 'teff_e_kounkel', sort_key]]