import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

ra_offset = 8
de_offset = 12

sources = pd.read_csv('/home/l3wei/ONC/Catalogs/synthetic catalog - epoch combined.csv')
apogee = pd.read_csv('/home/l3wei/ONC/Catalogs/broader_apogee.csv')
tobin = pd.read_csv('/home/l3wei/ONC/Catalogs/tobin 2009.csv')

# Load fits file. See https://docs.astropy.org/en/stable/visualization/wcsaxes/index.html.
image_path = '/home/l3wei/ONC/Figures/Skymap/hlsp_orion_hst_acs_colorimage_r_v1_drz.fits'
hdu = fits.open(image_path)[0]
wcs = WCS(image_path)


##################################################
################# Large Plot Data ################
##################################################

image_data_large = hdu.data
image_wcs_large = wcs

# filter
ra_upper_large, dec_lower_large = image_wcs_large.wcs_pix2world(0 - 18000*0.05, 0 - 18000*0.05, 0)
ra_lower_large, dec_upper_large = image_wcs_large.wcs_pix2world(17999 + 18000*0.05, 17999 + 18000*0.05, 0)
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

center = SkyCoord("05h35m16.26s", "-05d23m16.4s")

# Cutout. See https://docs.astropy.org/en/stable/nddata/utils.html.
cutout = Cutout2D(hdu.data, center, (2500, 2500), wcs=wcs)

# Power Scale. See Page 3 of http://aspbooks.org/publications/442/633.pdf.
a = 1000
image_data_zoom = ((np.power(a, cutout.data/255) - 1)/a)*255
image_wcs_zoom = cutout.wcs

# filter
ra_upper_zoom, dec_lower_zoom = image_wcs_zoom.wcs_pix2world(0 - 2500*0.05, 0 - 2500*0.05, 0)
ra_lower_zoom, dec_upper_zoom = image_wcs_zoom.wcs_pix2world(2499 + 2500*0.05, 2499 + 2500*0.05, 0)
apogee_zoom = apogee.loc[
    (apogee._RAJ2000 >= ra_lower_zoom) & (apogee._RAJ2000 <= ra_upper_zoom) & (apogee._DEJ2000 >= dec_lower_zoom) & (apogee._DEJ2000 <= dec_upper_zoom)
]
tobin_zoom = tobin.loc[
    (tobin._RAJ2000 >= ra_lower_zoom) & (tobin._RAJ2000 <= ra_upper_zoom) & (tobin._DEJ2000 >= dec_lower_zoom) & (tobin._DEJ2000 <= dec_upper_zoom)
].reset_index(drop=True)

# Convert sources to pixels: https://python4astronomers.github.io/astropy/wcs.html.
hc2000_zoom_ra, hc2000_zoom_de = image_wcs_zoom.wcs_world2pix(sources._RAJ2000[idx_hc2000], sources._DEJ2000[idx_hc2000], 0)
apogee_zoom_ra, apogee_zoom_de = image_wcs_zoom.wcs_world2pix(apogee_zoom._RAJ2000, apogee_zoom._DEJ2000, 0)
tobin_zoom_ra, tobin_zoom_de = image_wcs_zoom.wcs_world2pix(tobin_zoom._RAJ2000, tobin_zoom._DEJ2000, 0)
# binary_ra, binary_de = image_wcs.wcs_world2pix(binary_coord.ra, binary_coord.dec, 0)
hc2000_zoom_ra -= ra_offset
hc2000_zoom_de -= de_offset
apogee_zoom_ra -= ra_offset
apogee_zoom_de -= de_offset
tobin_zoom_ra  -= ra_offset
tobin_zoom_de  -= de_offset


##################################################
###################### Plot ######################
##################################################
box_x, box_y = image_wcs_large.wcs_world2pix(center.ra.degree, center.dec.degree, 0)
box_x -= 2500/2
box_y -= 2500/2

# # Large Plot
# fig = plt.figure(figsize=(7, 7), dpi=300)
# ax1 = fig.add_subplot(projection=image_wcs_large)
# ax1.imshow(image_data_large, cmap='gray')
# ax1.add_patch(Rectangle((box_x, box_y), 2500, 2500, linestyle='dashed', linewidth=1.5, edgecolor='k', facecolor='none', zorder=4))
# ax1.scatter(hc2000_ra_large, hc2000_de_large, s=10, edgecolor='C6', linewidths=1, facecolor='none', label='NIRSPEC (This Study)', zorder=3)
# ax1.scatter(apogee_ra_large, apogee_de_large, s=10, marker='s', edgecolor='C9', linewidths=1, facecolor='none', label='APOGEE', zorder=2)
# ax1.scatter(tobin_ra_large, tobin_de_large, s=15, marker='^', edgecolor='C1', linewidths=1, facecolor='none', label='Tobin et al. 2009', zorder=1)
# ax1.set_xlim([0, 17999])
# ax1.set_ylim([0, 17999])
# ax1.set_xlabel('Right Ascension', fontsize=12)
# ax1.set_ylabel('Declination', fontsize=12)
# ax1.legend(loc='upper right')
# plt.savefig('/home/l3wei/ONC/Figures/Skymap/Skymap Large.pdf', bbox_inches='tight')
# plt.savefig('/home/l3wei/ONC/Figures/Skymap/Skymap Large.png', bbox_inches='tight')
# plt.show()

# # Zoom Plot
# fig = plt.figure(figsize=(7, 7), dpi=300)
# ax2 = fig.add_subplot(projection=image_wcs_zoom)
# ax2.imshow(image_data_zoom, cmap='gray')
# ax2.scatter(hc2000_zoom_ra, hc2000_zoom_de, s=35, edgecolor='C6', linewidths=1.5, facecolor='none', label='NIRSPEC (This Study)', zorder=3)
# ax2.scatter(apogee_zoom_ra, apogee_zoom_de, s=35, marker='s', edgecolor='C9', linewidths=1.5, facecolor='none', label='APOGEE', zorder=2)
# ax2.scatter(tobin_zoom_ra, tobin_zoom_de, s=40, marker='^', edgecolor='C1', linewidths=1.5, facecolor='none', label='Tobin et al. 2009', zorder=1)
# ax2.set_xlim([0, 2499])
# ax2.set_ylim([0, 2499])
# ax2.set_xlabel('Right Ascension', fontsize=12)
# ax2.set_ylabel('Declination', fontsize=12)
# ax2.legend(loc='upper right')
# plt.savefig('/home/l3wei/ONC/Figures/Skymap/Skymap Zoomed.pdf', bbox_inches='tight')
# plt.savefig('/home/l3wei/ONC/Figures/Skymap/Skymap Zoomed.png', bbox_inches='tight')
# plt.show()


##################################################
################ Side-by-Side Plot ###############
##################################################

# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"

fig = plt.figure(figsize=(12, 5), dpi=300)
ax1 = fig.add_subplot(1, 2, 1, projection=image_wcs_large)
ax1.imshow(image_data_large, cmap='gray')
ax1.add_patch(Rectangle((box_x, box_y), 2500, 2500, linestyle='dashed', linewidth=1.5, edgecolor='k', facecolor='none', zorder=4))
ax1.scatter(hc2000_ra_large, hc2000_de_large, s=10, edgecolor='C6', linewidths=1, facecolor='none', label='NIRSPEC (This Study)', zorder=3)
ax1.scatter(apogee_ra_large, apogee_de_large, s=10, marker='s', edgecolor='C9', linewidths=1, facecolor='none', label='APOGEE', zorder=2)
ax1.scatter(tobin_ra_large, tobin_de_large, s=15, marker='^', edgecolor='C1', linewidths=1, facecolor='none', label='Tobin et al. 2009', zorder=1)
ax1.set_xlim([0, 17999])
ax1.set_ylim([0, 17999])
ax1.set_xlabel('Right Ascension', fontsize=12)
ax1.set_ylabel('Declination', fontsize=12)
ax1.legend(loc='upper right')

ax2 = fig.add_subplot(1, 2, 2, projection=image_wcs_zoom)
ax2.imshow(image_data_zoom, cmap='gray')
ax2.scatter(hc2000_zoom_ra, hc2000_zoom_de, s=35, edgecolor='C6', linewidths=1.5, facecolor='none', label='NIRSPEC (This Study)', zorder=3)
ax2.scatter(apogee_zoom_ra, apogee_zoom_de, s=35, marker='s', edgecolor='C9', linewidths=1.5, facecolor='none', label='APOGEE', zorder=2)
ax2.scatter(tobin_zoom_ra, tobin_zoom_de, s=40, marker='^', edgecolor='C1', linewidths=1.5, facecolor='none', label='Tobin et al. 2009', zorder=1)
ax2.set_xlim([0, 2500])
ax2.set_ylim([0, 2500])
ax2.set_xlabel('Right Ascension', fontsize=12)
ax2.set_ylabel('Declination', fontsize=12)
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('/home/l3wei/ONC/Figures/Skymap/Skymap Side-by-Side.pdf', bbox_inches='tight')
plt.savefig('/home/l3wei/ONC/Figures/Skymap/Skymap Side-by-Side.png', bbox_inches='tight')
plt.show()