import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle, Circle
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization.wcsaxes import SphericalCircle

user_path = os.path.expanduser('~')

ra_offset = 8
de_offset = 12

trapezium = SkyCoord("05h35m16.26s", "-05d23m16.4s")
sources = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog - epoch combined.csv')
apogee = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/broader_apogee.csv')
tobin = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/tobin 2009.csv')

sources_coord = SkyCoord(ra=sources._RAJ2000.to_numpy()*u.degree, dec=sources._DEJ2000.to_numpy()*u.degree)
apogee_coord = SkyCoord(ra=apogee._RAJ2000.to_numpy()*u.degree, dec=apogee._DEJ2000.to_numpy()*u.degree)
tobin_coord = SkyCoord(ra=tobin._RAJ2000.to_numpy()*u.degree, dec=tobin._DEJ2000.to_numpy()*u.degree)

sources['sep_to_trapezium'] = sources_coord.separation(trapezium).arcmin
apogee['sep_to_trapezium'] = apogee_coord.separation(trapezium).arcmin
tobin['sep_to_trapezium'] = tobin_coord.separation(trapezium).arcmin

# Load fits file. See https://docs.astropy.org/en/stable/visualization/wcsaxes/index.html.
image_path = f'{user_path}/ONC/figures/skymap/hlsp_orion_hst_acs_colorimage_r_v1_drz.fits'
hdu = fits.open(image_path)[0]
wcs = WCS(image_path)


##################################################
################# Large Plot Data ################
##################################################

image_data_large = hdu.data
image_wcs_large = wcs

image_size=18000
margin=0.05

# filter
ra_upper_large, dec_lower_large = image_wcs_large.wcs_pix2world(0 - image_size*margin, 0 - image_size*margin, 0)
ra_lower_large, dec_upper_large = image_wcs_large.wcs_pix2world(image_size*(margin + 1) - 1, image_size*(margin + 1) - 1, 0)
apogee_large = apogee.loc[
    (apogee._RAJ2000 >= ra_lower_large) & (apogee._RAJ2000 <= ra_upper_large) & (apogee._DEJ2000 >= dec_lower_large) & (apogee._DEJ2000 <= dec_upper_large)
]
tobin_large = tobin.loc[
    (tobin._RAJ2000 >= ra_lower_large) & (tobin._RAJ2000 <= ra_upper_large) & (tobin._DEJ2000 >= dec_lower_large) & (tobin._DEJ2000 <= dec_upper_large)
].reset_index(drop=True)

binary_coord = SkyCoord('05h35m14.9890227096s -05d21m59.923205424s')

# Sources index
idx_hc2000 = ~sources.HC2000.isna()

# Convert sources to pixels: https://python4astronomers.github.io/astropy/wcs.html.
hc2000_ra_large, hc2000_de_large = image_wcs_large.wcs_world2pix(sources._RAJ2000[idx_hc2000], sources._DEJ2000[idx_hc2000], 0)
apogee_ra_large, apogee_de_large = image_wcs_large.wcs_world2pix(apogee_large._RAJ2000, apogee_large._DEJ2000, 0)
tobin_ra_large, tobin_de_large = image_wcs_large.wcs_world2pix(tobin_large._RAJ2000, tobin_large._DEJ2000, 0)
# binary_ra, binary_de = image_wcs.wcs_world2pix(binary_coord.ra, binary_coord.dec, 0)
hc2000_ra_large -= ra_offset
hc2000_de_large -= de_offset
apogee_ra_large -= ra_offset
apogee_de_large -= de_offset
tobin_ra_large  -= ra_offset
tobin_de_large  -= de_offset


##################################################
############### Zoomed In Plot Data ##############
##################################################

box_size = 5000
# Cutout. See https://docs.astropy.org/en/stable/nddata/utils.html.
cutout = Cutout2D(hdu.data, position=trapezium, size=(box_size, box_size), wcs=wcs)

# Power Scale. See Page 3 of http://aspbooks.org/publications/442/633.pdf.
a = 100
image_data_zoom = ((np.power(a, cutout.data/255) - 1)/a)*255
# image_data_zoom = (cutout.data/255)**2*255
image_wcs_zoom = cutout.wcs

# # filter
# margin = 0.05
# # 左下
# ra_upper_zoom, dec_lower_zoom = image_wcs_zoom.wcs_pix2world(0 - cut_size*margin, 0 - cut_size*margin, 0)
# # 右上
# ra_lower_zoom, dec_upper_zoom = image_wcs_zoom.wcs_pix2world(cut_size*(margin + 1) - 1, cut_size*(margin + 1) - 1, 0)
# apogee_zoom = apogee.loc[
#     (apogee._RAJ2000 >= ra_lower_zoom) & (apogee._RAJ2000 <= ra_upper_zoom) & (apogee._DEJ2000 >= dec_lower_zoom) & (apogee._DEJ2000 <= dec_upper_zoom)
# ]
# tobin_zoom = tobin.loc[
#     (tobin._RAJ2000 >= ra_lower_zoom) & (tobin._RAJ2000 <= ra_upper_zoom) & (tobin._DEJ2000 >= dec_lower_zoom) & (tobin._DEJ2000 <= dec_upper_zoom)
# ].reset_index(drop=True)

apogee_zoom = apogee.loc[apogee.sep_to_trapezium <= 4]
tobin_zoom = tobin.loc[tobin.sep_to_trapezium <= 4]

# Convert sources to pixels: https://python4astronomers.github.io/astropy/wcs.html.
hc2000_ra_zoom, hc2000_de_zoom = image_wcs_zoom.wcs_world2pix(sources._RAJ2000[idx_hc2000], sources._DEJ2000[idx_hc2000], 0)
apogee_ra_zoom, apogee_de_zoom = image_wcs_zoom.wcs_world2pix(apogee_zoom._RAJ2000, apogee_zoom._DEJ2000, 0)
tobin_ra_zoom, tobin_de_zoom = image_wcs_zoom.wcs_world2pix(tobin_zoom._RAJ2000, tobin_zoom._DEJ2000, 0)
# binary_ra, binary_de = image_wcs.wcs_world2pix(binary_coord.ra, binary_coord.dec, 0)
hc2000_ra_zoom -= ra_offset
hc2000_de_zoom -= de_offset
apogee_ra_zoom -= ra_offset
apogee_de_zoom -= de_offset
tobin_ra_zoom  -= ra_offset
tobin_de_zoom  -= de_offset


##################################################
################ Side-by-Side Plot ###############
##################################################

fig = plt.figure(figsize=(12, 5), dpi=300)
ax1 = fig.add_subplot(1, 2, 1, projection=image_wcs_large)
ax1.imshow(image_data_large, cmap='gray')
r = SphericalCircle((trapezium.ra, trapezium.dec), 4. * u.arcmin,
                    linestyle='dashed', linewidth=1.5, 
                    edgecolor='w', facecolor='none', alpha=0.8, zorder=4, 
                    transform=ax1.get_transform('icrs'))
ax1.add_patch(r)
ax1.scatter(hc2000_ra_large, hc2000_de_large, s=10, edgecolor='C6', linewidths=1, facecolor='none', label='NIRSPEC (This Study)', zorder=3)
ax1.scatter(apogee_ra_large, apogee_de_large, s=10, marker='s', edgecolor='C9', linewidths=1, facecolor='none', label='APOGEE', zorder=2)
ax1.scatter(tobin_ra_large, tobin_de_large, s=15, marker='^', edgecolor='C1', linewidths=1, facecolor='none', label='Tobin et al. 2009', zorder=1)
ax1.set_xlim([0, image_size - 1])
ax1.set_ylim([0, image_size - 1])
ax1.set_xlabel('Right Ascension', fontsize=12)
ax1.set_ylabel('Declination', fontsize=12)
ax1.legend(loc='upper right')

ax2 = fig.add_subplot(1, 2, 2, projection=image_wcs_zoom)
ax2.imshow(image_data_zoom, cmap='gray')
ax2.scatter(hc2000_ra_zoom, hc2000_de_zoom, s=15, edgecolor='C6', linewidths=1.25, facecolor='none', label='NIRSPEC (This Study)', zorder=3)
ax2.scatter(apogee_ra_zoom, apogee_de_zoom, s=15, marker='s', edgecolor='C9', linewidths=1.25, facecolor='none', label='APOGEE', zorder=2)
# ax2.scatter(tobin_zoom_ra, tobin_zoom_de, s=40, marker='^', edgecolor='C1', linewidths=1.5, facecolor='none', label='Tobin et al. 2009', zorder=1)
ax2.set_xlim([0, box_size - 1])
ax2.set_ylim([0, box_size - 1])
ax2.set_xlabel('Right Ascension', fontsize=12)
ax2.set_ylabel('Declination', fontsize=12)
ax2.legend(loc='upper right')
plt.tight_layout()
# plt.savefig(f'{user_path}/ONC/Figures/skymap/Skymap Side-by-Side.pdf', bbox_inches='tight')
plt.savefig(f'{user_path}/ONC/figures/skymap/Skymap Side-by-Side.png', bbox_inches='tight', transparent=True)
plt.show()