#!/usr/bin/env python
import numpy as np
import csv
from scipy.stats import norm
import sys, os, os.path, time, gc
from astropy.table import Table, join, hstack, vstack
from astropy.io import ascii
import astropy.io.fits as fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy.lib.recfunctions
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LogNorm
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import MultipleLocator
from astropy.wcs import WCS
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
#plt.rc('text', usetex=True)
plt.rc('legend', fontsize=7)
plt.rc('axes', labelsize=10)

# Set Units
d2a  = 3600.
d2ma = 3600000.


#########################################################################################################


# filename = get_pkg_data_filename('hlsp_orion_hst_acs_colorimage_r_v1_drz.fits')

# hdu = fits.open(filename)[0]
# wcs = WCS(hdu.header)

# plt.figure(figsize=(7,7))
# #ax = plt.subplot(2,1,1,projection=wcs)
# #ax2 = plt.subplot(2,1,2,projection=wcs)
# ax2 = plt.subplot(1,1,1,projection=wcs)
# #ax.imshow(hdu.data, origin='lower', cmap=plt.get_cmap('gray'))
# ax2.imshow(hdu.data, origin='lower', vmin=200, vmax=255, cmap=plt.get_cmap('gray'))
# #ax.grid(color='white', ls='dotted', zorder=10)
# ax2.grid(color='white', ls='dotted', zorder=10)
# #ax.set_xlabel(r'$\alpha_\mathrm{J2000}$')
# #ax.set_ylabel(r'$\delta_\mathrm{J2000}$')
# ax2.set_xlabel(r'$\alpha_\mathrm{J2000}$')
# ax2.set_ylabel(r'$\delta_\mathrm{J2000}$')
# #ax.minorticks_on()
# ax2.minorticks_on()
# #print(ax.get_xlim())
# #print(ax.get_ylim())


# Get the targets
t000 = Table.read('pm_hillenbrand.txt', format='ascii').filled(-9999)

t000['[HC2000]'] = t000['col2']
#sys.exit()
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

tp = t0[np.where( (t0['mualpha']!=-9999) & (t0['mudelta']!=-9999) & (t0['[HC2000]']<100) )]
t1 = t0[np.where( (t0['RV_us']>-100) )]#& (t0['Mass(daRio12)']!=-9999)) ]
#t11= t0[np.where( (t0['RV_us']!=-9999) & (t0['RVe_us']<1) & (t0['RV_us']!=-10) & (t0['J-K']!=-9999) )]#& (t0['Mass(daRio12)']!=-9999)) ]
#t22= t0[np.where( (t0['RV_us']!=-9999) & (t0['RVe_us']>=1) & (t0['RV_us']!=-10) & (t0['J-K']!=-9999) )]#& (t0['Mass(daRio12)']!=-9999)) ]
#t2 = t0[np.where( (t0['RV_us']==-10) & (t0['J-K']!=-9999) )]#& (t0['Mass(daRio12)']!=-9999)) ]
#t3 = t0[np.where( (t0['RV_us']==-9999) & (t0['[HC2000]']<100) & (t0['J-K']!=-9999) )]#& (t0['Mass(daRio12)']!=-9999)) ]
tT = t0[np.where( (t0['RV_tobin']!=-9999) & (t0['[HC2000]']<100) )]#& (t0['Mass(daRio12)']!=-9999)) ]
tA = t0[np.where( (t0['RV_apogee']!=-9999) & (t0['[HC2000]']<100) )]#& (t0['Mass(daRio12)']!=-9999)) ]

N1 = t0[np.where( (t0['[HC2000]']==99) | (t0['[HC2000]']==94) | (t0['[HC2000]']==86) | (t0['[HC2000]']==90) | (t0['[HC2000]']==79) | (t0['[HC2000]']==98) )]
N2 = t0[np.where( (t0['[HC2000]']==80) | (t0['[HC2000]']==74) | (t0['[HC2000]']==69) | (t0['[HC2000]']==61) | (t0['[HC2000]']==71) )]
N3 = t0[np.where( (t0['[HC2000]']==22) | (t0['[HC2000]']==18) | (t0['[HC2000]']==10) )]
N4 = t0[np.where( (t0['[HC2000]']==5) | (t0['[HC2000]']==4) | (t0['[HC2000]']==38) | (t0['[HC2000]']==32) | (t0['[HC2000]']==70) | (t0['[HC2000]']==43) | (t0['[HC2000]']==17) | (t0['[HC2000]']==66))]

T  = t0[np.where( (t0['RV_us']<-100) )]#& (t0['Jmag_R']-t0['Hmag_R']<=0.9) )]
TP = t0[np.where( (t0['RV_us']==-9999) & (t0['[HC2000]']<=39) & (t0['[HC2000]']>=35) )]#& (t0['Jmag_R']-t0['Hmag_R']<=0.9) )]

# ax2.scatter(T['ra_use'], T['dec_use'], marker='.', edgecolors='lime', facecolors='none', alpha=0.9, transform=ax2.get_transform('world'), label='Targets', zorder=14)
# ax2.scatter(t1['ra_use'], t1['dec_use'], marker='.', edgecolors='coral', facecolors='none', alpha=0.9, transform=ax2.get_transform('world'), label='Observed', zorder=14)
# ax2.scatter(tT['ra_use'], tT['dec_use'], marker='^', edgecolors='c', facecolors='none', alpha=0.9, transform=ax2.get_transform('world'), label='Tobin RV', zorder=14)
# ax2.scatter(tA['ra_use'], tA['dec_use'], marker='s', edgecolors='m', facecolors='none', alpha=0.9, transform=ax2.get_transform('world'), label='APOGEE RV', zorder=14)
# ax2.scatter(N1['ra_use'], N1['dec_use'], marker='x', edgecolors='fuchsia', facecolors='fuchsia', alpha=0.9, transform=ax2.get_transform('world'), label='Jan 17', zorder=14)
# ax2.scatter(N2['ra_use'], N2['dec_use'], marker='x', edgecolors='cornflowerblue', facecolors='cornflowerblue', alpha=0.9, transform=ax2.get_transform('world'), label='Jan 18', zorder=14)
# ax2.scatter(N3['ra_use'], N3['dec_use'], marker='x', edgecolors='orange', facecolors='orange', alpha=0.9, transform=ax2.get_transform('world'), label='Jan 19', zorder=14)
# ax2.scatter(N4['ra_use'], N4['dec_use'], marker='x', edgecolors='tomato', facecolors='tomato', alpha=0.9, transform=ax2.get_transform('world'), label='Jan 20', zorder=14)

with open('trap_stars.list', 'r') as file:
    raw = file.readlines()

# Laser Guide Star
lgs_name = []
lgs_coord = []

for line in raw:
    lgs_name.append(line[0:16].strip())
    lgs_coord.append(
        line[16:18] + 'h' + line[19:21] + 'm' + line[22:27] + 's '
        + line[28:31] + 'd' + line[32:34] + 'm' + line[35:37] + 's')

# CA       = SkyCoord('5h35m15.81s' '-5d23m14.3s', frame='fk5') # done
# CB       = SkyCoord('5h35m16.11s' '-5d23m06.8s', frame='fk5') # done
# CC       = SkyCoord('5h35m16.46s' '-5d23m23.0s', frame='fk5') # done
# CD       = SkyCoord('5h35m17.24s' '-5d23m16.6s', frame='fk5') # done
# CE       = SkyCoord('5h35m15.77s' '-5d23m09.9s', frame='fk5') # done
# V1333ori = SkyCoord('5h35m17.01s' '-5d22m33.0s', frame='fk5') # done
print('HC2000\tTT-Star\tDist(arcsec)\tKmag')
with open('Laser Guide Star.csv', 'w') as file:
	writer = csv.writer(file)
	writer.writerow(['HC2000', 'TT-Star', 'Dist(arcsec)', 'Kmag'])

for star_name, star_ra, star_dec, star_kmag in zip(T['[HC2000]'], T['ra_use'], T['dec_use'], T['Kmag']):
	# ax2.text(x=b, y=c, s=a, color='lime', transform=ax2.get_transform('world'), fontsize=8, zorder=100)
	c2 = SkyCoord(ra=star_ra * u.degree, dec=star_dec * u.degree, frame='fk5')
	sep = []
	for coord in lgs_coord:
		CA = SkyCoord(coord, frame='fk5')
		sep.append(c2.separation(CA).arcsecond)
	sep = np.array(sep)
	idx = np.argmin(sep)
	print('%s\t%s\t%0.2f\t%0.3f' %(star_name, lgs_name[idx], sep[idx], star_kmag))
	with open('Laser Guide Star.csv', 'a') as file:
		writer = csv.writer(file)
		writer.writerow([star_name, lgs_name[idx], sep[idx], star_kmag])
		
	# sepA = c2.separation(CA)
	# sepB = c2.separation(CB)
	# sepC = c2.separation(CC)
	# sepD = c2.separation(CD)
	# sepE = c2.separation(CE)
	# sepV = c2.separation(V1333ori)
	# print('%s\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.3f'%(a, sepA.arcsecond, sepB.arcsecond, sepC.arcsecond, sepD.arcsecond, sepE.arcsecond, sepV.arcsecond, d))

# for a,b,c in zip(t1['ID'], t1['ra_use'], t1['dec_use']):
# 	ax2.text(x=b, y=c, s=a, color='coral', transform=ax2.get_transform('world'), fontsize=8, zorder=100)

# for a,b,c in zip(N1['ID'], N1['ra_use'], N1['dec_use']):
# 	ax2.text(x=b, y=c, s=a, color='fuchsia', transform=ax2.get_transform('world'), fontsize=8, zorder=100)
# for a,b,c in zip(N2['ID'], N2['ra_use'], N2['dec_use']):
# 	ax2.text(x=b, y=c, s=a, color='cornflowerblue', transform=ax2.get_transform('world'), fontsize=8, zorder=100)
# for a,b,c in zip(N3['ID'], N3['ra_use'], N3['dec_use']):
# 	ax2.text(x=b, y=c, s=a, color='orange', transform=ax2.get_transform('world'), fontsize=8, zorder=100)
# for a,b,c in zip(N4['ID'], N4['ra_use'], N4['dec_use']):
# 	ax2.text(x=b, y=c, s=a, color='tomato', transform=ax2.get_transform('world'), fontsize=8, zorder=100)


#ax2.scatter(TP['ra2'], TP['dec2'], marker='x', edgecolors='r', facecolors='r', alpha=0.9, transform=ax2.get_transform('world'), label='Targets to observe', zorder=14)

# ax2.set_xlim(7000, 9000)
# ax2.set_ylim(9000, 11000)
# #ax.set_xlim(-0.5, 17999.5)
# #ax.set_ylim(-0.5, 17999.5)
# #ax.legend().set_zorder(16)
# ax2.legend().set_zorder(16)
# #plt.subplots_adjust(wspace=-2, hspace=-2)
# #plt.tight_layout()
# plt.savefig('../Plots/ONC_image_potential_targets_jan2021.png', dpi=600, bbox_inches="tight")
# #plt.savefig('Plots/ONC_All_Targets.pdf', bbox_inches="tight")
# plt.show()