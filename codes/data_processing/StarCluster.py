import os
import copy
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
from itertools import compress

user_path = os.path.expanduser('~')
trapezium = SkyCoord("05h35m16.26s", "-05d23m16.4s", distance=1000/2.59226*u.pc)

class StarCluster:
    def __init__(self, pandas_df) -> None:
        self.data = pandas_df.copy()
        self.validate_columns(self)
        self.ra         = self.data.RAJ2000.values  * u.degree
        self.dec        = self.data.DEJ2000.values  * u.degree
        self.teff       = self.data.teff.values     * u.K
        self.teff_e     = self.data.teff_e.values   * u.K
        self.rv         = self.data.rv.values       * u.km/u.s
        self.rv_e       = self.data.rv_e.values     * u.km/u.s
        self.vsini      = self.data.vsini.values    * u.km/u.s
        self.vsini_e    = self.data.vsini_e.values  * u.km/u.s
        self.pmRA       = self.data.pmRA.values     * u.mas/u.yr
        self.pmRA_e     = self.data.pmRA_e.values   * u.mas/u.yr
        self.pmDE       = self.data.pmDE.values     * u.mas/u.yr
        self.pmDE_e     = self.data.pmDE_e.values   * u.mas/u.yr
    
    @staticmethod
    def validate_columns(self, columns=['RAJ2000', 'DEJ2000', 'teff', 'teff_e', 'rv', 'rv_e', 'pmRA', 'pmRA_e', 'pmDE', 'pmDE_e']):
        if 'rv' in columns:
            if 'rv' not in self.data.keys():
                print('Replacing rv with rv_helio')
                columns[columns.index('rv')] = 'rv_helio'
        
        iskey = [_ in self.data.keys() for _ in columns]
        if all(iskey):
            pass
        else:
            raise KeyError(f'Column not found: {", ".join(list(compress(columns, ~np.array(iskey))))}')
    
    @property
    def len(self):
        return len(self.data)
    
    def apply_constraint(self, constraint, return_copy=False):
        """Apply constraint to the star cluster.

        Parameters
        ----------
        constraint : array-like
            boolean or index of the constraint.
        return_copy : bool, optional
            whether to return a copied version of self, by default False

        Returns
        -------
        instance of class StarCluster
        """
        if return_copy:
            return StarCluster(self.data.loc[constraint].reset_index(drop=True))
        else:
            self.__init__(self.data.loc[constraint].reset_index(drop=True))
    
    
    def assign_coord(self, distance):
        """Assign coordinates in 3D given distance

        Parameters
        ----------
        distance : astropy quantity
            distance from Earth
        """
        self.distance=distance
        self.coord = SkyCoord(
            ra=self.ra,
            dec=self.dec,
            pm_ra_cosdec=self.pmRA,
            pm_dec=self.pmDE,
            radial_velocity=self.rv,
            distance=self.distance
        )
        self.velocity = self.coord.velocity.d_xyz
    

    def assign_mass(self, model_name):
        """Assign mass to self

        Parameters
        ----------
        model_name : str
            name of stellar evolutionary model from which the masses are derived
        """
        self.mass = self.data[f'mass_{model_name}'].values * u.solMass
        self.mass_e = self.data[f'mass_e_{model_name}'].values * u.solMass
    
    
    def plot_skymap(self, background_path=None, show_figure=True, save_path=None, **kwargs):
        color = kwargs.get('color', 'C6')
        linewidth = kwargs.get('linewidth', 1)
        ra_offset = kwargs.get('ra_offset', 0)
        dec_offset = kwargs.get('dec_offset', 0)
        
        if background_path is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(self.ra.value, self.dec.value, s=10, edgecolor=color, linewidths=linewidth, facecolor='none')
            ax.set_xlabel('Right Ascension (degree)', fontsize=12)
            ax.set_ylabel('Declination (degree)', fontsize=12)
            
        else:
            hdu = fits.open(background_path)[0]
            wcs = WCS(background_path)
            image_size = np.shape(hdu.data)[0]
            
            ra, dec = wcs.wcs_world2pix(self.ra.value, self.dec.value, 0)
            ra  += ra_offset # offset in pixels
            dec += dec_offset # offset in pixels
            fig = plt.figure(figsize=(6, 6))
            ax  = fig.add_subplot(1, 1, 1, projection=wcs)
            print(f'Type when initialized: {type(ax)}')
            ax.imshow(hdu.data, cmap='gray')
            ax.scatter(ra, dec, s=10, edgecolor=color, linewidths=linewidth, facecolor='none', zorder=1)
            ax.set_xlim([0, image_size - 1])
            ax.set_ylim([0, image_size - 1])
            ax.set_xlabel('Right Ascension', fontsize=12)
            ax.set_ylabel('Declination', fontsize=12)
            ax.legend(loc='upper right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show_figure:
            plt.show()
        print(f'Type before returned: {type(ax)}')
        return fig, ax



class ONC(StarCluster):
    def __init__(self, pandas_df) -> None:
        super().__init__(pandas_df)
    
    def plot_skymap(self, circle=4.*u.arcmin, zoom=False, background_path=None, show_figure=True, **kwargs):
        color=kwargs.get('color', 'C6')
        if zoom:
            linewidth=kwargs.get('linewidth', 1.25)
            hdu = fits.open(background_path)[0]
            wcs = WCS(background_path)
            box_size=5000
            cutout = Cutout2D(hdu.data, position=trapezium, size=(box_size, box_size), wcs=wcs)
            # Power Scale. See Page 3 of http://aspbooks.org/publications/442/633.pdf.
            a = 100
            image_data = ((np.power(a, cutout.data/255) - 1)/a)*255
            image_wcs = cutout.wcs
            ra, dec = image_wcs.wcs_world2pix(self.ra.value, self.dec.value, 0)
            ra -= 8
            dec -= 12
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(1, 1, 1, projection=image_wcs)
            ax.imshow(image_data, cmap='gray')
            ax.scatter(ra, dec, s=15, edgecolor=color, linewidths=linewidth, facecolor='none')
            ax.set_xlabel('Right Ascension', fontsize=12)
            ax.set_ylabel('Declination', fontsize=12)
        
        else:
            fig, ax = super().plot_skymap(background_path, show_figure=False, ra_offset=-8, dec_offset=-12, **kwargs)
            if (circle is not None) and (background_path is not None):
                r = SphericalCircle((trapezium.ra, trapezium.dec), circle,
                    linestyle='dashed', linewidth=1.5, 
                    edgecolor='w', facecolor='none', alpha=0.8, zorder=4, 
                    transform=ax.get_transform('icrs'))
                ax.add_patch(r)

        if show_figure:
            plt.show()
        return fig, ax


# Main function
orion = ONC(pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/sources 2d.csv', dtype={'ID_gaia': str, 'ID_kim': str}))
orion.assign_mass('MIST')
orion.assign_coord(distance=389*u.pc)

max_mass_error=0.5 * u.solMass
max_rv=np.inf * u.km/u.s
max_v_error=5. * u.km/u.s
constraint = \
    (~np.isnan(orion.pmRA)) & (~np.isnan(orion.pmDE)) & \
    (~np.isnan(orion.mass)) & \
    (orion.mass_e < max_mass_error) & \
    (orion.rv < max_rv)

orion.apply_constraint(constraint=constraint)
orion.plot_skymap()
fig, ax = orion.plot_skymap(background_path=f'{user_path}/ONC/figures/skymap/hlsp_orion_hst_acs_colorimage_r_v1_drz.fits', show_figure=False)
plt.show()