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
        """Initialize StarCluster with attribute: data

        Parameters
        ----------
        pandas_df : pandas dataframe
            Dataframe containing information about the star cluster
        """
        self.data = pandas_df.copy()

    @property
    def len(self):
        return len(self.data)

    def _string_or_quantity(self, value, unit=1):
        """Access data for setting attributes.

        Parameters
        ----------
        value : str | astropy quantity
            The value to be determined
        unit : int | astropy unit, optional
            Unit of value if value is str, by default 1

        Returns
        -------
        data
            Astropy quanitity from the data column or user input

        Raises
        ------
        KeyError
            If value is a str but not a key in self.data
        ValueError
            If value is neither of str, ndarray, int, or float
        """
        if isinstance(value, str):
            try:
                return self.data[value].values * unit
            except KeyError:
                raise KeyError(f'{value} is not a key in self.data!') from None
        elif isinstance(value, np.ndarray) or isinstance(value, int) or isinstance(value, float):
            return value
        else:
            raise ValueError(f'The parameter must be either a string of column name or a quantity, not {type(value)}.')
    
    
    def set_ra_dec(self, ra, dec):
        """Set ra and dec with attributes: ra, dec
        
        Parameters
        ----------
        ra : str | astropy quantity
            Column name or astropy quantity of ra
        dec : str | astropy quantity
            Column name or astropy quantity of dec
        """
        self.ra  = self._string_or_quantity(ra, unit=u.degree)
        self.dec = self._string_or_quantity(dec, unit=u.degree)
    
    
    def set_pm(self, pmRA, pmRA_e, pmDE, pmDE_e):
        """Set proper motion with attributes: pmRA, pmRA_e, pmDE, pmDE_e

        Parameters
        ----------
        pmRA : str | astropy quantity
            Column name or astropy quantity of proper motion in RA
        pmRA_e : str | astropy quantity
            Column name or astropy quantity of proper motion in RA uncertainty
        pmDE : str | astropy quantity
            Column name or astropy quantity of proper motion in DE
        pmDE_e : str | astropy quantity
            Column name or astropy quantity of proper motion in DE uncertainty
        """
        self.pmRA   = self._string_or_quantity(pmRA,   unit=u.mas/u.yr)
        self.pmRA_e = self._string_or_quantity(pmRA_e, unit=u.mas/u.yr)
        self.pmDE   = self._string_or_quantity(pmDE,   unit=u.mas/u.yr)
        self.pmDE_e = self._string_or_quantity(pmDE_e, unit=u.mas/u.yr)
    
    
    def set_rv(self, rv, rv_e):
        """Set radial velocity with attributes: rv, rv_e

        Parameters
        ----------
        rv : str | astropy quantity
            Column name or astropy quantity of radial velocity
        rv_e : str | astropy quantity
            Column name or astropy quantity of radial velocity uncertainty
        """
        self.rv   = self._string_or_quantity(rv,   unit=u.km/u.s)
        self.rv_e = self._string_or_quantity(rv_e, unit=u.km/u.s)
    
    
    def set_coord(self, distance=None, distance_e=None):
        """Set astropy SkyCoord with attributes: coord (and velocity, if distance is not None)

        Parameters
        ----------
        distance : str | astropy quantity, optional
            Column name or astropy quantity of distance, by default None
        distance_e : str | astropy quantity, optional
            Column name or astropy quantity of distance uncertainty, by default None
        """
        if distance is not None:
            self.distance = self._string_or_quantity(distance, unit=u.pc)
            if distance_e is not None:
                self.distance_e = self._string_or_quantity(distance_e, unit=u.pc)
            self.coord = SkyCoord(
                ra=self.ra,
                dec=self.dec,
                pm_ra_cosdec=self.pmRA,
                pm_dec=self.pmDE,
                radial_velocity=self.rv,
                distance=self.distance
            )
            self.velocity = self.coord.velocity.d_xyz
        else:
            self.coord = SkyCoord(
                ra=self.ra,
                dec=self.dec,
                pm_ra_cosdec=self.pmRA,
                pm_dec=self.pmDE,
                radial_velocity=self.rv
            )
    
    
    def set_teff(self, teff, teff_e):
        """Set effective temperature with attributes: teff, teff_e

        Parameters
        ----------
        teff : str | astropy quantity
            Column name or astropy quantity of effective temperature
        teff_e : str | astropy quantity
            Column name or astropy quantity of effective temperature uncertainty
        """
        self.teff   = self._string_or_quantity(teff,   unit=u.K)
        self.teff_e = self._string_or_quantity(teff_e, unit=u.K)
    
        
    def set_mass(self, mass, mass_e):
        """Set mass with attributes: mass, mass_e

        Parameters
        ----------
        mass : str | astropy quantity
            Column name or astropy quantity of mass.
        mass_e : str | astropy quantity
            Column name or astropy quantity of mass uncertainty.
        """
        self.mass = self._string_or_quantity(mass, unit=u.solMass)
        self.mass_e = self._string_or_quantity(mass_e, unit=u.solMass)
    
        
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
        return fig, ax



class ONC(StarCluster):
    def __init__(self, pandas_df) -> None:
        super().__init__(pandas_df)
        super().set_ra_dec('RAJ2000', 'DEJ2000')
        super().set_pm('pmRA', 'pmRA_e', 'pmDE', 'pmDE_e')
        super().set_rv('rv', 'rv_e')
        super().set_coord(distance=389*u.pc, distance_e=3*u.pc)
        super().set_teff('teff', 'teff_e')
    
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
    
    def preprocessing(self):
        trapezium_names = ['A', 'B', 'C', 'D', 'E']
            # Replace Trapezium stars fitting results with literature values.
        for i in range(len(trapezium_names)):
            trapezium_index = self.data.loc[self.data.theta_orionis == trapezium_names[i]].index[-1]
            for model_name in ['BHAC15', 'MIST', 'Feiden', 'Palla']:
                self.data.loc[trapezium_index, [f'mass_{model_name}', f'mass_e_{model_name}']] = [self.data.loc[trapezium_index, 'mass_literature'], self.data.loc[trapezium_index, 'mass_e_literature']]

# Main function
sources = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/sources 2d.csv', dtype={'ID_gaia': str, 'ID_kim': str})
orion = ONC(sources)

orion.plot_skymap()
orion.plot_skymap(background_path=f'{user_path}/ONC/figures/skymap/hlsp_orion_hst_acs_colorimage_r_v1_drz.fits')
orion.plot_skymap(zoom=True, background_path=f'{user_path}/ONC/figures/skymap/hlsp_orion_hst_acs_colorimage_r_v1_drz.fits')