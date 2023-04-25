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

ra_offset = 6
de_offset = 12

trapezium = SkyCoord("05h35m16.26s", "-05d23m16.4s")
sources = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/target list.csv')
Parenago_idx = sources.loc[sources.HC2000 == 546].index[0]

sources_coord = SkyCoord(ra=sources._RAJ2000.to_numpy()*u.degree, dec=sources._DEJ2000.to_numpy()*u.degree)
scale_pc = 0.1 # pc
scale_sec = scale_pc * 43200/np.pi / 389 # second
scale_arcmin = 0.1/389 * 180/np.pi * 60
scale_coord = SkyCoord(['5h35m{:.2f}s -5d24m45s'.format(10.5), '5h35m{:.2f}s -5d24m45s'.format(10.5 + scale_sec)])

sources['sep_to_trapezium'] = sources_coord.separation(trapezium).arcmin

# Load fits file. See https://docs.astropy.org/en/stable/visualization/wcsaxes/index.html.
image_path = f'{user_path}/ONC/figures/Skymap/hlsp_orion_hst_acs_colorimage_r_v1_drz.fits'
hdu = fits.open(image_path)[0]
wcs = WCS(image_path)


##################################################
############### Zoomed In Plot Data ##############
##################################################

box_size = 2000
# Cutout. See https://docs.astropy.org/en/stable/nddata/utils.html.
cutout = Cutout2D(hdu.data, position=trapezium, size=(box_size, box_size), wcs=wcs)

# Power Scale. See Page 3 of http://aspbooks.org/publications/442/633.pdf.
a = 10000
image_data_zoom = ((np.power(a, cutout.data/255) - 1)/a)*255
# image_data_zoom = (cutout.data/255)**2*255
image_wcs_zoom = cutout.wcs


# Convert sources to pixels: https://python4astronomers.github.io/astropy/wcs.html.
# hc2000_zoom_ra, hc2000_zoom_de = image_wcs_zoom.wcs_world2pix(sources.loc[(~sources.HC2000.isna()) & (sources.theta_orionis.isna()), '_RAJ2000'], sources.loc[(~sources.HC2000.isna()) & (sources.theta_orionis.isna()), '_DEJ2000'], 0)
hc2000_zoom_ra, hc2000_zoom_de = image_wcs_zoom.wcs_world2pix(sources._RAJ2000, sources._DEJ2000, 0)
scale_ra, scale_de = image_wcs_zoom.wcs_world2pix(scale_coord.ra.deg, scale_coord.dec.deg, 0)
# binary_ra, binary_de = image_wcs.wcs_world2pix(binary_coord.ra, binary_coord.dec, 0)

hc2000_zoom_ra -= ra_offset
hc2000_zoom_de -= de_offset


##################################################
################ Side-by-Side Plot ###############
##################################################

# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"

fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=image_wcs_zoom)
ax.imshow(image_data_zoom, cmap='gray')
ax.scatter(hc2000_zoom_ra, hc2000_zoom_de, s=25, edgecolor='C6', linewidths=1.25, facecolor='none', label='NIRSPEC Sources', zorder=1)
ax.scatter(hc2000_zoom_ra[Parenago_idx], hc2000_zoom_de[Parenago_idx], s=100, marker='*', edgecolor='C9', linewidth=1, facecolor='none', label='Parenago', zorder=2)
ax.plot(scale_ra, scale_de, color='w', linewidth=1.5)
ax.text(np.mean(scale_ra), scale_de[0] + 45, '${:.1f}$ pc'.format(scale_pc), color='w', horizontalalignment='center', verticalalignment='center')
ax.text(np.mean(scale_ra), scale_de[0] - 50, "${:.2f}'$".format(scale_arcmin), color='w', horizontalalignment='center', verticalalignment='center')
ax.set_xlim([0, box_size - 1])
ax.set_ylim([0, box_size - 1])
ax.set_xlabel('Right Ascension', fontsize=12)
ax.set_ylabel('Declination', fontsize=12)
ax.legend(loc='upper left')
plt.tight_layout()
# plt.savefig(f'{user_path}/ONC/Figures/Skymap/Skymap Proposal.pdf', bbox_inches='tight')
# plt.savefig(f'{user_path}/ONC/Figures/Skymap/Skymap Side-by-Side.png', bbox_inches='tight')
plt.show()