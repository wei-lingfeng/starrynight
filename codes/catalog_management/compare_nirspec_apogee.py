import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord

# nirspec = pd.read_csv('/home/l3wei/ONC/Catalogs/nirspec sources.csv')
# apogee = pd.read_csv('/home/l3wei/ONC/Catalogs/apogee x 2mass.csv')

# nirspec_coord = SkyCoord(ra=nirspec._RAJ2000*u.degree, dec=nirspec._DEJ2000*u.degree)
# apogee_coord = SkyCoord(ra=apogee._RAJ2000*u.degree, dec=apogee._DEJ2000*u.degree)

# max_sep = 1.*u.arcsec
# idx, d2d, d3d = nirspec_coord.match_to_catalog_sky(apogee_coord)
# sep_constraint = d2d < max_sep
# teff_constraint = sep_constraint & (nirspec.teff_e < 100)

# nirspec_teff = nirspec.loc[teff_constraint, 'teff'].to_numpy()
# nirspec_teff_e = nirspec.loc[teff_constraint, 'teff_e'].to_numpy()
# apogee_teff = apogee.loc[idx[teff_constraint], 'teff'].to_numpy()
# apogee_teff_e = apogee.loc[idx[teff_constraint], 'teff_e'].to_numpy()

# vsini_constraint = sep_constraint
# nirspec_vsini = nirspec.loc[vsini_constraint, 'vsini'].to_numpy()
# nirspec_vsini_e = nirspec.loc[vsini_constraint, 'vsini_e'].to_numpy()
# apogee_vsini = apogee.loc[idx[vsini_constraint], 'vsini'].to_numpy()
# apogee_vsini_e = apogee.loc[idx[vsini_constraint], 'vsini_e'].to_numpy()


# hmag = apogee.loc[idx[teff_constraint], 'Hmag'].to_numpy()
# hmag_e = apogee.loc[idx[teff_constraint], 'Hmag_e'].to_numpy()

# fig, ax = plt.subplots(figsize=(7, 5))
# ax.errorbar(hmag, apogee_teff - nirspec_teff, xerr=hmag_e, yerr=(nirspec_teff_e**2 + apogee_teff_e**2)**0.5, fmt='.')
# ax.set_xlabel('Hmag')
# ax.set_ylabel('Teff Difference (APOGEE - NIRSPEC) (K)')
# plt.show()

# fig, ax = plt.subplots(figsize=(7, 5))
# ax.errorbar(nirspec_vsini, apogee_vsini, xerr=nirspec_vsini_e, yerr=apogee_vsini_e, fmt='.')
# ax.plot([10, 60], [10, 60], linestyle='dashed', color='C3')
# ax.set_xlabel('NIRSPEC vsini (km/s)')
# ax.set_ylabel('APOGEE vsini (km/s)')
# plt.show()

sources = pd.read_csv('/home/l3wei/ONC/Catalogs/synthetic catalog - epoch combined.csv')

teff_constraint = (sources.teff_e_apogee < 100)
teff_constraint1 = (sources.teff_e < 100) & (sources.teff_e_apogee < 100)
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(sources.loc[teff_constraint, 'teff_apogee'], 10**sources.loc[teff_constraint, 'logT'], xerr=sources.loc[teff_constraint, 'teff_e_apogee'], fmt='.', label='Hillenbrand vs APOGEE')
ax.errorbar(sources.loc[teff_constraint1, 'teff_apogee'], sources.loc[teff_constraint1, 'teff'], xerr=sources.loc[teff_constraint1, 'teff_e_apogee'], yerr=sources.loc[teff_constraint1, 'teff_e'], fmt='.', label='NIRSPEC vs APOGEE')
ax.plot([3000, 7000], [3000, 7000], linestyle='dashed', color='C7', label='Equal Temperature')
ax.set_xlabel('APOGEE Teff (K)')
ax.set_ylabel('Hillenbrand / NIRSPEC Teff (K)')
ax.legend()
plt.show()