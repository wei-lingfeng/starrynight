# Match Laser Guide Star:
# Input: ONC Sources Table, pm_hillenbrand, trap_stars.list, target list
# Output: Match Laser Guide Star.csv

import os
import numpy as np
import pandas as pd
from astropy.table import Table, join
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt


###########################################################
####################### Target List #######################
###########################################################
year = '2023'
month = 'jan'

data_path = year + month + '/' + year + ' ' + month + ' observing plan.csv'
save_path = year + month

targets = pd.read_csv(data_path)

sources = pd.read_csv('/home/l3wei/ONC/Catalogs/synthetic catalog - epoch combined.csv')
sources = sources[(~sources.ID_apogee.isna()) & (sources.HC2000.isna())]
apogee_coord = SkyCoord(
    ra=sources._RAJ2000.to_numpy()*u.degree, 
    dec=sources._DEJ2000.to_numpy()*u.degree
)

###########################################################
################# Build Target List Table #################
###########################################################

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Get the targets
t000 = Table.read('pm_hillenbrand.txt', format='ascii').filled(-9999)

t000['[HC2000]'] = t000['col2']
t00   = Table.read('ONC Sources - Combined_all_Redux.csv', format='csv').filled(-9999)

c0 = SkyCoord(t00['RAJ2000_2'], t00['DEJ2000'], unit=(u.hourangle, u.deg))
t00['ra_use']  = c0.ra.deg
t00['dec_use'] = c0.dec.deg


t00['[HC2000]'][np.where(t00['[HC2000]'] == '306_A')] = 306
t00['[HC2000]'][np.where(t00['[HC2000]'] == '306_B')] = 9990
t00['[HC2000]'][np.where(t00['[HC2000]'] == '522_A')] = 522
t00['[HC2000]'][np.where(t00['[HC2000]'] == '522_B')] = 9991
t00['[HC2000]'] = np.array(t00['[HC2000]'], dtype=int)


t0 = join(t00, t000, keys='[HC2000]')
# T  = t0[np.where( (t0['RV_us']<-100) )]#& (t0['Jmag_R']-t0['Hmag_R']<=0.9) )]

idx = []
for target in targets['H#']:
    idx.append(list(t0['[HC2000]']).index(target))
T = t0[idx]


############################################################
################### Build LGS Star Table ###################
############################################################

with open('trap_stars.list', 'r') as file:
    raw = file.readlines()

# Build Laser Guide Star Table
lgs_name = []
lgs_coords_original = [] # in "XX XX XX.XX +/-XX XX XX.X"
lgs_coords = []  # in "XXhXXmXX.XXs +/-XXdXXmXXs"
lgs_rmag = []

for line in raw:
    lgs_name.append(line[0:16].strip()) # name
    lgs_coords.append(
        line[16:18] + 'h' + line[19:21] + 'm' + line[22:27] + 's '
        + line[28:31] + 'd' + line[32:34] + 'm' + line[35:39] + 's'
    )
    lgs_coords_original.append(line[16:39])
    lgs_rmag.append(
        line[45:].strip()   # rmag, b-v, b-r
    )


############################################################
############## Match All LGS Stars For Targets #############
############################################################

# create starlist file
with open(save_path + '/Starlist_' + save_path + '.list', 'w') as file:
    pass


star_names = ['[HC2000]' + str(_) for _ in T['[HC2000]']] + list(sources.ID_apogee)
star_ras = list(T['ra_use']) + list(sources._RAJ2000)
star_decs = list(T['dec_use']) + list(sources._DEJ2000)
star_kmags = list(T['Kmag']) + list(sources.Kmag)

for i, (star_name, star_ra, star_dec, star_kmag) in enumerate(zip(star_names, star_ras, star_decs, star_kmags)):
    star_coord = SkyCoord(ra=star_ra * u.degree, dec=star_dec * u.degree, frame='fk5')
    
    # Calculate Separation
    sep = []
    for coord in lgs_coords:
        lgs_coord = SkyCoord(coord, frame='fk5')
        sep.append(star_coord.separation(lgs_coord).arcsecond)
    
    sep = np.array(sep)
    
    if min(sep) >= 60:  # no lgs star within 1 arcminute: list the 2 closest lgs.
        idxs = np.argsort(sep)[:2]
        print('Warning: Minimum LGS star is greater than 60 arcsec away for {}!'.format(star_name))
    elif sum(sep <= 30) < 2:   # less than 2 lgs stars within 30 arcsec: select the closest 2 (maximum).
        idxs = np.where(sep <= 60)[0]
        idxs = idxs[np.argsort(sep[idxs])[:2]]
    else:   # all lgs stars within 30 arcsec.
        idxs = np.where(sep <= 30)[0]
        idxs = idxs[np.argsort(sep[idxs])]
    
    
    # write target
    # star_coord = '%02d %02d %05.2f %03d %02d %04.1f' \
    #     %(T['col3'][i], T['col4'][i], T['col5'][i],
    #     T['col6'][i], T['col7'][i], T['col8'][i])
    
    star_coord_str = '{:02.0f} {:02.0f} {:05.2f} {:03.0f} {:02.0f} {:04.1f}'.format(
        *star_coord.ra.hms, 
        star_coord.dec.signed_dms.sign*star_coord.dec.signed_dms.d, 
        star_coord.dec.signed_dms.m, 
        star_coord.dec.signed_dms.s
    )
    
    with open(save_path + '/Starlist_' + save_path + '.list', 'a') as file:
        file.write('%-23s%s 2000 lgs=1\n' %(str(star_name), star_coord_str))
    
    # write LGS stars
    for idx in idxs:
        with open(save_path + '/Starlist_' + save_path + '.list', 'a') as file:
            file.write('    %-19s%s 2000 sep=%05.2f %s\n' \
                %(lgs_name[idx], lgs_coords_original[idx],sep[idx], lgs_rmag[idx]))


