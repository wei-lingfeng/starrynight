"""_summary_"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.optimize import curve_fit

from codes import nirspao

pm_table  = Table.read('../catalogs/pm_hillenbrand.csv')
obs = Table.read('../../catalogs/synthetic catalog.ecsv')
observing_plan = Table.read('../catalogs/observing_plan_all.csv')

pm_ID = np.ma.array([0] * len(obs), mask=False, dtype=int)

for i in range(sum(~obs['HC2000'].mask)):
    try:
        pm_ID[i] = observing_plan['#'][pm_table['H#'] == obs['HC2000'][i]].value[0]
    except:
        pass

pm_ID = np.ma.masked_equal(pm_ID, 0)
obs['pm_ID'] = pm_ID

for i in range(len(obs)):
    try:
        obs['Kmag'][i] = observing_plan['K'][observing_plan['H#'] == obs['HC2000'][i]].value[0]
        obs['itime'][i] = observing_plan['itime'][observing_plan['H#'] == obs['HC2000'][i]].value[0]
    except:
        pass


rv_nirspao = obs['rv_helio'].filled(np.nan)
rv_apogee = obs['rv_apogee'].filled(np.nan)

rv_lim_nirspao = [22, 33]
rv_lim_apogee = [22, 32]

for rv, rv_lim in zip([rv_nirspao, rv_apogee], [rv_lim_nirspao, rv_lim_apogee]):
    plt.hist(rv, range=(18, 38), bins=20)
    rv_mean = np.mean(rv[(rv >= rv_lim[0]) & (rv <= rv_lim[1])])
    plt.axvline(rv_mean, color='C3')
    plt.axvline(rv_lim[0], color='C3', ls='--')
    plt.axvline(rv_lim[1], color='C3', ls='--')
    plt.show()

columns = ['HC2000', 'pm_ID', 'Kmag', 'Hmag', 'theta_orionis', 'rv_helio', 'rv_apogee', 'rv_tobin', 'itime']
nirspao_anomalies = obs[~np.isnan(rv_nirspao) & ((rv_nirspao <= rv_lim_nirspao[0]) | (rv_nirspao >= rv_lim_nirspao[1]))][columns]
apogee_anomalies = obs[~np.isnan(rv_apogee) & ((rv_apogee <= rv_lim_apogee[0]) | (rv_apogee >= rv_lim_apogee[1]))][columns]

mask = np.array([_ in [214, 170, 172, 546] for _ in nirspao_anomalies['HC2000']])
nirspao_anomalies = nirspao_anomalies[~mask]