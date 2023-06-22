import os
import copy
import numpy as np
from numpy import ma
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import QTable, MaskedColumn
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from itertools import compress
Vizier.ROW_LIMIT = -1
Gaia.ROW_LIMIT = -1
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

user_path = os.path.expanduser('~')
trapezium = SkyCoord("05h35m16.26s", "-05d23m16.4s", distance=1000/2.59226*u.pc)

def merge(array1, array2):
    '''
    Union of two lists with nans.
    If value in array1 exists, take array1. Otherwise, take array2.
    '''
    merge = copy.copy(array1[:])
    
    if array1.dtype.name.startswith('float'):
        nan_idx = np.isnan(array1)
    merge[nan_idx] = array2[nan_idx]
    return merge


def distance_cut(dist, e_dist, min_dist=300*u.pc, max_dist=500*u.pc, min_plx_over_e=5):
    plx = 1000/dist.to(u.pc).value * u.mas
    plx_e = e_dist / dist * plx
    dist_constraint = (dist >= min_dist) & (dist <= max_dist) & (plx/plx_e >= min_plx_over_e) | np.isnan(dist)
    print(f'{min_dist}~{max_dist} distance range constraint: {sum(dist_constraint) - 5} sources out of {len(dist_constraint) - 5} remains.')
    return dist_constraint


class StarCluster:
    def __init__(self, path) -> None:
        """Initialize StarCluster with attribute: data

        Parameters
        ----------
        path : str
            Path of astropy table in ecsv format
        """
        self.data = QTable.read(path)

    @property
    def len(self):
        return len(self.data)

    def _string_or_quantity(self, value):
        """Access data for setting attributes.

        Parameters
        ----------
        value : str | astropy quantity
            The value to be accessed, either a str of column name of self.data or array

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
                return self.data[value]
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
        self.ra  = self._string_or_quantity(ra)
        self.dec = self._string_or_quantity(dec)
    
    
    def set_pm(self, pmRA, e_pmRA, pmDE, e_pmDE):
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
        self.pmRA   = self._string_or_quantity(pmRA)
        self.e_pmRA = self._string_or_quantity(e_pmRA)
        self.pmDE   = self._string_or_quantity(pmDE)
        self.e_pmDE = self._string_or_quantity(e_pmDE)
    
    
    def set_rv(self, rv, e_rv):
        """Set radial velocity with attributes: rv, rv_e

        Parameters
        ----------
        rv : str | astropy quantity
            Column name or astropy quantity of radial velocity
        e_rv : str | astropy quantity
            Column name or astropy quantity of radial velocity uncertainty
        """
        self.rv   = self._string_or_quantity(rv)
        self.rv_e = self._string_or_quantity(e_rv)
    
    
    def set_coord(self, ra=None, dec=None, distance=None, e_distance=None):
        """Set astropy SkyCoord with attributes: coord (and velocity, if distance is not None)

        Parameters
        ----------
        ra : str | astropy quantity, optional
            Column name or astropy quantity of ra, by default None
        dec : str | astropy quantity, optional
            Column name or astropy quantity of dec, by default None
        distance : str | astropy quantity, optional
            Column name or astropy quantity of distance, by default None
        e_distance : str | astropy quantity, optional
            Column name or astropy quantity of distance uncertainty, by default None
        """
        if hasattr(self, 'ra') and hasattr(self, 'dec'):
            pass
        else:
            self.ra  = self._string_or_quantity(ra)
            self.dec = self._string_or_quantity(dec)
        if distance is not None:
            self.distance = self._string_or_quantity(distance)
            if e_distance is not None:
                self.e_distance = self._string_or_quantity(e_distance)
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
    
    
    def set_teff(self, teff, e_teff):
        """Set effective temperature with attributes: teff, teff_e

        Parameters
        ----------
        teff : str | astropy quantity
            Column name or astropy quantity of effective temperature
        teff_e : str | astropy quantity
            Column name or astropy quantity of effective temperature uncertainty
        """
        self.teff   = self._string_or_quantity(teff)
        self.e_teff = self._string_or_quantity(e_teff)
    
        
    def set_mass(self, mass, e_mass):
        """Set mass with attributes: mass, e_mass

        Parameters
        ----------
        mass : str | astropy quantity
            Column name or astropy quantity of mass.
        e_mass : str | astropy quantity
            Column name or astropy quantity of mass uncertainty.
        """
        self.mass = self._string_or_quantity(mass)
        self.e_mass = self._string_or_quantity(e_mass)
    
        
    # def apply_constraint(self, constraint, return_copy=False):
    #     """Apply constraint to the star cluster.

    #     Parameters
    #     ----------
    #     constraint : array-like
    #         Boolean or index of the constraint.
    #     return_copy : bool, optional
    #         Whether to return a copied version of self, by default False

    #     Returns
    #     -------
    #     instance of class StarCluster
    #     """
    #     if return_copy:
    #         return StarCluster(self.data.loc[constraint].reset_index(drop=True))
    #     else:
    #         self.__init__(self.data.loc[constraint].reset_index(drop=True))
    
    
    def plot_skymap(self, background_path=None, show_figure=True, save_path=None, ra_offset=0, dec_offset=0, label='Sources', color='C6', lw=1, zorder=1, constraint=None):
        if not (hasattr(self, 'ra') and hasattr(self, 'dec')):
            raise KeyError("'ra' and 'dec' are required attributes for plotting skymap. Please run self.set_ra_dec() first.")
        
        if background_path is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(self.ra.value, self.dec.value, s=10, edgecolor=color, linewidths=lw, facecolor='none', zorder=zorder)
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
            if constraint is None:
                ax.scatter(ra, dec, s=10, label=label, edgecolor=color, linewidths=lw, facecolor='none', zorder=zorder)
            else:
                ax.scatter(ra[constraint], dec[constraint], s=10, label=label, edgecolor=color, linewidths=lw, facecolor='none', zorder=zorder)
            ax.set_xlim([0, image_size - 1])
            ax.set_ylim([0, image_size - 1])
            ax.set_xlabel('Right Ascension', fontsize=12)
            ax.set_ylabel('Declination', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show_figure:
            ax.legend(loc='upper right')
            plt.show()
        return fig, ax, wcs



class ONC(StarCluster):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.preprocessing()
        super().set_ra_dec('RAJ2000', 'DEJ2000')
        super().set_pm('pmRA', 'e_pmRA', 'pmDE', 'e_pmDE')
        super().set_rv('RV', 'e_RV')
        super().set_coord(distance=389*u.pc, e_distance=3*u.pc)
        super().set_teff('Teff', 'e_Teff')
        self.mass_MIST = self.data['mass_MIST']
        self.e_mass_MIST = self.data['e_mass_MIST']
        self.mass_BHAC15 = self.data['mass_MIST']
        self.e_mass_BHAC15 = self.data['e_mass_MIST']
        self.mass_Feiden = self.data['mass_Feiden']
        self.e_mass_Feiden = self.data['e_mass_Feiden']
        self.mass_Palla = self.data['mass_Palla']
        self.e_mass_Palla = self.data['e_mass_Palla']
    
    def preprocessing(self):
        trapezium_names = ['A', 'B', 'C', 'D', 'E']
        
        # Replace Trapezium stars fitting results with literature values.
        for i in range(len(trapezium_names)):
            trapezium_index = self.data['theta_orionis'] == trapezium_names[i]
            for model_name in ['BHAC15', 'MIST', 'Feiden', 'Palla']:
                self.data[f'mass_{model_name}'][trapezium_index]    = self.data['mass_literature'][trapezium_index]
                self.data[f'e_mass_{model_name}'][trapezium_index]  = self.data['e_mass_literature'][trapezium_index]
        
        max_rv_e = 5*u.km/u.s
        rv_constraint = ((
            (self.data['e_RV_nirspao']   <= max_rv_e) |
            (self.data['e_RV_apogee']    <= max_rv_e)
        ) | (
            ~self.data['theta_orionis'].mask
        ))
        print(f"Maximum RV error of {max_rv_e} constraint: {sum(rv_constraint) - sum(~self.data['theta_orionis'].mask)} out of {len(rv_constraint) - sum(~self.data['theta_orionis'].mask)} remaining.")
        self.data = self.data[rv_constraint]
        
        rv_use_apogee = (self.data['e_RV_nirspao'] > max_rv_e) & (self.data['e_RV_apogee'] <= max_rv_e)
        self.data['RV_nirspao', 'e_RV_nirspao'][rv_use_apogee]      = self.data['RV_apogee', 'e_RV_apogee'][rv_use_apogee]
        
        # Apply gaia constraint
        gaia_columns = [key for key in self.data.keys() if (key.endswith('gaia') | key.startswith('plx') | key.startswith('Gmag') | key.startswith('astrometric') | (key=='ruwe') | (key=='bp_rp'))]
        gaia_filter = (self.data['astrometric_gof_al'] < 16) & (self.data['Gmag'] < 16*u.mag)
        self.data['astrometric_n_good_obs_al'] = MaskedColumn([float(_) for _ in self.data['astrometric_n_good_obs_al']])
        self.data[*gaia_columns][~gaia_filter] = np.nan
        offset_RA = np.nanmedian(self.data['pmRA_gaia'] - self.data['pmRA_kim'])
        offset_DE = np.nanmedian(self.data['pmDE_gaia'] - self.data['pmDE_kim'])
        print(f'offset in RA and DEC is {offset_RA}, {offset_DE}.')
        with open(f'{user_path}/ONC/starrynight/codes/analysis/pm_offset.txt', 'w') as file:
            file.write(f'pmRA_gaia - pmRA_kim = {offset_RA}\npmDE_gaia - pmDE_kim = {offset_DE}')
        
        # Plot pm comparison
        # Remove binaries
        unique, count = np.unique(self.data['HC2000'], return_counts=True)
        dup_idx = [list(self.data['HC2000']).index(_) for _ in list(unique[(count > 1) & (count < 10)])]
        not_dup = np.ones(len(self.data), dtype=bool)
        not_dup[dup_idx] = False
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.errorbar(
            (self.data['pmRA_gaia'] - self.data['pmRA_kim'] - offset_RA).value[not_dup],
            (self.data['pmDE_gaia'] - self.data['pmDE_kim'] - offset_DE).value[not_dup],
            xerr = ((self.data['e_pmRA_gaia']**2 + self.data['e_pmDE_kim']**2)**0.5).value[not_dup],
            yerr = ((self.data['e_pmDE_gaia']**2 + self.data['e_pmDE_kim']**2)**0.5).value[not_dup],
            fmt='o', color=(.2, .2, .2, .8), markersize=3, ecolor='black', alpha=0.2
        )
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.hlines(0, xmin, xmax, colors='C3', ls='--')
        ax.vlines(0, ymin, ymax, colors='C3', ls='--')
        ax.set_xlabel(r'$\mu_{\alpha^*, DR3} - \mu_{\alpha^*, HK} - \widetilde{\Delta\mu_{\alpha^*}} \quad \left(\mathrm{mas}\cdot\mathrm{yr}^{-1}\right)$', fontsize=12)
        ax.set_ylabel(r'$\mu_{\delta, DR3} - \mu_{\delta, HK} - \widetilde{\Delta\mu_\delta} \quad \left(\mathrm{mas}\cdot\mathrm{yr}^{-1}\right)$', fontsize=12)
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        plt.savefig(f'{user_path}/ONC/figures/Proper Motion Comparison.pdf')
        plt.show()
        
        # Correct Gaia values
        self.data['pmRA_gaia'] -= offset_RA
        self.data['pmDE_gaia'] -= offset_DE
        
        # merge proper motion and rv
        # prioritize kim
        self.data['pmRA']   = merge(self.data['pmRA_kim'],      self.data['pmRA_gaia'])
        self.data['e_pmRA'] = merge(self.data['e_pmRA_kim'],    self.data['e_pmRA_gaia'])
        self.data['pmDE']   = merge(self.data['pmDE_kim'],      self.data['pmDE_gaia'])
        self.data['e_pmDE'] = merge(self.data['e_pmDE_kim'],    self.data['e_pmDE_gaia'])
        
        # choose one from the two options: weighted avrg or prioritize nirspec.
        # # weighted average
        # self.data['RV'], self.data['e_RV'] = weighted_avrg_and_merge(self.datarv_helio, self.datarv_apogee, error1=self.datarv_e_nirspec, error2=self.datarv_e_apogee)
        # prioritize nirspec values
        self.data['RV'] = merge(self.data['RVhelio'], self.data['RV_apogee'])
        self.data['e_RV'] = merge(self.data['e_RV_nirspao'], self.data['e_RV_apogee'])
        self.data['dist'] = (1000/self.data['plx'].to(u.mas).value) * u.pc
        self.data['e_dist'] = self.data['e_plx'] / self.data['plx'] * self.data['dist']
        
        dist_constraint = distance_cut(self.data['dist'], self.data['e_dist'])
        self.data.remove_rows(~dist_constraint)
        
        print('After all constraint:\nNIRSPEC:\t{}\nAPOGEE:\t{}\nMatched:\t{}\nTotal:\t{}'.format(
            sum((self.data['theta_orionis'].mask) & (~self.data['HC2000'].mask)),
            sum((self.data['theta_orionis'].mask) & (~self.data['APOGEE'].mask)),
            sum((self.data['theta_orionis'].mask) & (~self.data['HC2000'].mask) & (~self.data['APOGEE'].mask)),
            sum((self.data['theta_orionis'].mask))
        ))

        self.data.write(f'{user_path}/ONC/starrynight/catalogs/sources 2d.ecsv', overwrite=True)
    
    
    def plot_skymap(self, circle=4.*u.arcmin, zoom=False, background_path=None, show_figure=True, label='Sources', color='C6', lw=1.25, zorder=1, constraint=None):
        if zoom:
            hdu = fits.open(background_path)[0]
            wcs = WCS(background_path)
            box_size=5000
            cutout = Cutout2D(hdu.data, position=trapezium, size=(box_size, box_size), wcs=wcs)
            # Power Scale. See Page 3 of http://aspbooks.org/publications/442/633.pdf.
            a = 100
            image_data = ((np.power(a, cutout.data/255) - 1)/a)*255
            wcs = cutout.wcs
            ra, dec = wcs.wcs_world2pix(self.ra.value, self.dec.value, 0)
            ra -= 8
            dec -= 12
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(1, 1, 1, projection=wcs)
            ax.imshow(image_data, cmap='gray')
            if constraint is None:
                ax.scatter(ra, dec, s=15, edgecolor=color, linewidths=lw, facecolor='none', zorder=zorder, label=label)
            else:
                ax.scatter(ra[constraint], dec[constraint], s=15, edgecolor=color, linewidths=lw, facecolor='none', zorder=zorder, label=label)
            ax.set_xlim([0, box_size - 1])
            ax.set_ylim([0, box_size - 1])
            ax.set_xlabel('Right Ascension', fontsize=12)
            ax.set_ylabel('Declination', fontsize=12)
        
        else:
            fig, ax, wcs = super().plot_skymap(background_path, show_figure=False, ra_offset=-8, dec_offset=-12, label=label, zorder=zorder, constraint=constraint)
            if (circle is not None) and (background_path is not None):
                r = SphericalCircle((trapezium.ra, trapezium.dec), circle,
                    linestyle='dashed', linewidth=1.5, 
                    edgecolor='w', facecolor='none', alpha=0.8, zorder=4, 
                    transform=ax.get_transform('icrs'))
                ax.add_patch(r)

        if show_figure:
            ax.legend(loc='upper right')
            plt.show()
        return fig, ax, wcs



def plot_skymaps(orion, background_path=f'{user_path}/ONC/figures/skymap/hlsp_orion_hst_acs_colorimage_r_v1_drz.fits', ra_offset=-8, dec_offset=-12):
    tobin       = Vizier.get_catalogs('J/ApJ/697/1103/table3')[0]
    tobin_coord = SkyCoord([ra + dec for ra, dec in zip(tobin['RAJ2000'], tobin['DEJ2000'])], unit=(u.hourangle, u.deg))
    gaia = Gaia.cone_search_async(trapezium, radius=u.Quantity(4.2, u.arcmin)).get_results()
    apogee = Vizier.query_region(trapezium, radius=0.4*u.deg, catalog='III/284/allstars')[0]

    # Wide field view figure
    fig, ax, wcs = orion.plot_skymap(background_path=background_path, show_figure=False, label='NIRSPAO', constraint=~orion.data['HC2000'].mask, zorder=3)
    tobin_ra, tobin_dec = wcs.wcs_world2pix(tobin_coord.ra.value, tobin_coord.dec.value, 0)
    apogee_ra, apogee_dec = wcs.wcs_world2pix(apogee['RAJ2000'].value, apogee['DEJ2000'].value, 0)
    tobin_ra    += ra_offset
    tobin_dec   += dec_offset
    apogee_ra   += ra_offset
    apogee_dec  += dec_offset

    ax.scatter(apogee_ra, apogee_dec, s=10, marker='s', edgecolor='C9', linewidths=1, facecolor='none', label='APOGEE', zorder=2)
    ax.scatter(tobin_ra, tobin_dec, s=10, marker='^', edgecolor='C1', linewidths=1, facecolor='none', label='Tobin et al. 2009', zorder=1)
    ax.legend(loc='upper right')
    plt.show()

    # Zoom-in figure
    Parenago_idx = list(orion.data['HC2000']).index(546)
    fig, ax, wcs = orion.plot_skymap(background_path=background_path, show_figure=False, label='NIRSPAO', zoom=True, constraint=~orion.data['HC2000'].mask, zorder=3)
    apogee_ra, apogee_dec = wcs.wcs_world2pix(apogee['RAJ2000'].value, apogee['DEJ2000'].value, 0)
    Parenago_ra, Parenago_dec = wcs.wcs_world2pix(orion.ra[Parenago_idx].value, orion.dec[Parenago_idx].value, 0)
    apogee_ra       += ra_offset
    apogee_dec      += dec_offset
    Parenago_ra     += ra_offset
    Parenago_dec    += dec_offset
    ax.scatter(apogee_ra, apogee_dec, s=15, marker='s', edgecolor='C9', linewidths=1.25, facecolor='none', label='APOGEE', zorder=2)
    ax.scatter(Parenago_ra, Parenago_dec, s=100, marker='*', edgecolor='yellow', linewidth=1, facecolor='none', label='Parenago 1837', zorder=4)
    ax.legend(loc='upper right')
    plt.show()



# Main function
orion = ONC(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog - epoch combined.ecsv')
# plot_skymaps(orion)

