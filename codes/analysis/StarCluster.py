import os
import copy
import numpy as np
from numpy import ma
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
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


#################################################
##################### Class #####################
#################################################

class StarCluster:
    def __init__(self, table) -> None:
        """Initialize StarCluster with attribute: data

        Parameters
        ----------
        table : astropy QTable
            Astropy table of data
        """
        self.data = table
        # fill with nan
        for key in self.data.keys():
            if self.data[key].dtype.name.startswith('str') or self.data[key].dtype.name.startswith('int'):
                continue
            try:
                self.data[key] = self.data[key].filled(np.nan)
            except:
                pass

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
            raise AttributeError("Attributes 'ra' and 'dec' are required for plotting skymap. Please run self.set_ra_dec() first.")
        
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
    
    
    def plot_3d(self, scale=3, show_figure=True, label='Sources'):
        """Plot 3D html figures of position and velocity

        Parameters
        ----------
        scale : int, optional
            Scale of cone, by default 3
        show_figure : bool, optional
            Show figure or not, by default True
        label : str
            Label of cone plot, by default 'Sources'

        Returns
        -------
        fig : plotly figure handle
            Cone figure of position and velocity
        """
        if not hasattr(self, 'coord_3d'):
            raise AttributeError('Attribute coord_3d missing. Please set coord_3d prior to running plot_3d.')

        fig_data = [
            go.Cone(
                x=self.coord_3d.cartesian.x.value,
                y=self.coord_3d.cartesian.y.value,
                z=self.coord_3d.cartesian.z.value,
                u=self.coord_3d.velocity.d_x.value,
                v=self.coord_3d.velocity.d_y.value,
                w=self.coord_3d.velocity.d_z.value,
                sizeref=scale,
                colorscale='Blues',
                colorbar=dict(title=r'Velocity $\left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$'), #, thickness=20),
                colorbar_title_side='right',
                name=label
            )
        ]
        
        fig = go.Figure(fig_data)
        
        fig.update_layout(
            width=700,
            height=700,
            scene = dict(
                xaxis_title='X (pc)',
                yaxis_title='Y (pc)',
                zaxis_title='Z (pc)',
                aspectratio=dict(x=1, y=1, z=1)
            )
        )
        
        if show_figure:
            fig.show()

        return fig

    
    def plot_pm_rv(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.quiver(
            self.ra.value,
            self.dec.value,
            -self.pmRA.value,   # minius sign because ra increases from right to left
            self.pmDE.value,
            self.rv.value,
            cmap='coolwarm',
            width=0.006,
            scale=25
        )
        ax.quiverkey(im, X=0.15, Y=0.95, U=1, color='k',
            label=r'$1~\mathrm{mas}\cdot\mathrm{yr}^{-1}$', 
            labelcolor='w', labelpos='S', coordinates='axes')
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('DEC (deg)')
        plt.show()

    
    def pm_angle_distribution(self, save_path=None, constraint=None):
        def angle_between(v1, v2):
            """Angle between two vectors

            Parameters
            ----------
            v1 : array-like
                Vector 1
            v2 : array-like
                Vector 2

            Returns
            -------
            angle
                Angle between v1 and v2 in radians. Positive angle means clockwise rotation from v1 to v2.
            """
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            return np.arctan2(np.linalg.det(np.array([v1, v2])), np.dot(v1, v2))
        
        if constraint is None: constraint = np.ones(self.len, dtype=bool)
        position = np.array([
            -(self.ra - trapezium.ra)[constraint].value,   # minius sign because ra increases from right to left
            (self.dec - trapezium.dec)[constraint].value
        ])

        pm = np.array([
            -self.pmRA[constraint].value,
            self.pmDE[constraint].value
        ])
        
        # After adding minus sign: clock-wise is positive, counter clock-wise is negative
        angles = -np.array([angle_between(position[:, i], pm[:, i]) for i in range(np.shape(position)[1])])

        nbins = 12
        # angles[angles > np.pi - np.pi/nbins] = angles[angles > np.pi - np.pi/nbins] - 2*np.pi
        # hist, bin_edges = np.histogram(angles, nbins, range=(-np.pi - np.pi/nbins, np.pi - np.pi/nbins))
        hist, bin_edges = np.histogram(angles, nbins, range=(-np.pi, np.pi))
        theta = (bin_edges[:-1] + bin_edges[1:])/2
        colors = plt.cm.viridis(hist / max(hist))

        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot(projection='polar')
        ax.bar(theta, hist, width=2*np.pi/nbins, bottom=0.0, color=colors, alpha=0.5)
        ax.set_xticks(np.linspace(np.pi, -np.pi, 8, endpoint=False))
        ax.set_yticks(ax.get_yticks()[1:-2])
        ax.set_thetalim(-np.pi, np.pi)
        plt.draw()
        xticklabels = [label.get_text() for label in ax.get_xticklabels()]
        xticklabels[0] = '±180°   '
        ax.set_xticklabels(xticklabels)
        if save_path:
            if save_path.endswith('png'):
                plt.savefig(save_path, bbox_inches='tight', transparent=True)
            else:
                plt.savefig(save_path, bbox_inches='tight')
        plt.show()




class ONC(StarCluster):
    def __init__(self, table) -> None:
        super().__init__(table)
    
    def set_attr(self) -> None:
        """Set attributes of ra, dec, pmRA, pmDE rv, coord, teff, teff, mass of MIST, BHAC15, Feiden, Palla, and corresponding errors.
        """
        super().set_ra_dec('RAJ2000', 'DEJ2000')
        super().set_pm('pmRA', 'e_pmRA', 'pmDE', 'e_pmDE')
        super().set_rv('rv', 'e_rv')
        super().set_coord(distance=389*u.pc, e_distance=3*u.pc)
        super().set_teff('teff', 'e_teff')
        self.mass_MIST = self.data['mass_MIST']
        self.e_mass_MIST = self.data['e_mass_MIST']
        self.mass_BHAC15 = self.data['mass_MIST']
        self.e_mass_BHAC15 = self.data['e_mass_MIST']
        self.mass_Feiden = self.data['mass_Feiden']
        self.e_mass_Feiden = self.data['e_mass_Feiden']
        self.mass_Palla = self.data['mass_Palla']
        self.e_mass_Palla = self.data['e_mass_Palla']
        self.mass_Hillenbrand = self.data['mass_Hillenbrand']
    
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
            (self.data['e_rv_nirspao']   <= max_rv_e) |
            (self.data['e_rv_apogee']    <= max_rv_e)
        ) | (
            ~self.data['theta_orionis'].mask
        ))
        print(f"Maximum RV error of {max_rv_e} constraint: {sum(rv_constraint) - sum(~self.data['theta_orionis'].mask)} out of {len(rv_constraint) - sum(~self.data['theta_orionis'].mask)} remaining.")
        self.data = self.data[rv_constraint]
        
        rv_use_apogee = (self.data['e_rv_nirspao'] > max_rv_e) & (self.data['e_rv_apogee'] <= max_rv_e)
        self.data['rv_nirspao', 'e_rv_nirspao'][rv_use_apogee] = self.data['rv_apogee', 'e_rv_apogee'][rv_use_apogee]
        
        # Apply gaia constraint
        gaia_columns = [key for key in self.data.keys() if (key.endswith('gaia') | key.startswith('plx') | key.startswith('Gmag') | key.startswith('astrometric') | (key=='ruwe') | (key=='bp_rp'))]
        gaia_filter = (self.data['astrometric_gof_al'] < 16) & (self.data['Gmag'] < 16*u.mag)
        self.data['astrometric_n_good_obs_al'] = MaskedColumn([float(_) for _ in self.data['astrometric_n_good_obs_al']])
        for col in gaia_columns:
            self.data[col][~gaia_filter] = np.nan
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
            fmt='o', color=(.2, .2, .2, .8), markersize=3, ecolor='black', alpha=0.4
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
        
        # choose one from the two options: weighted avrg or prioritize NIRSPAO.
        # # weighted average
        # self.data['rv'], self.data['e_rv'] = weighted_avrg_and_merge(self.datarv_helio, self.datarv_apogee, error1=self.datarv_e_NIRSPAO, error2=self.datarv_e_apogee)
        # prioritize NIRSPAO values
        self.data['rv'] = merge(self.data['rv_helio'], self.data['rv_apogee'])
        self.data['e_rv'] = merge(self.data['e_rv_nirspao'], self.data['e_rv_apogee'])
        self.data['dist'] = (1000/self.data['plx'].to(u.mas).value) * u.pc
        self.data['e_dist'] = self.data['e_plx'] / self.data['plx'] * self.data['dist']
        
        dist_constraint = distance_cut(self.data['dist'], self.data['e_dist'])
        self.data.remove_rows(~dist_constraint)
        
        print('After all constraint:\nNIRSPAO:\t{}\nAPOGEE:\t{}\nMatched:\t{}\nTotal:\t{}'.format(
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
            if constraint is None: constraint = np.ones(self.len, dtype=bool)
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
    
    
    def plot_2d(self, scale=0.004):
        """Plot 2D html figure of position and proper motions

        Parameters
        ----------
        scale : float, optional
            Scale of quiver, by default 0.004

        Returns
        -------
        fig : plotly figure handle
            Quiver figure of position and proper motion
        """
        
        line_width=2
        opacity=0.8
        marker_size = 6

        nirspec_flag    = np.logical_and(~self.data['HC2000'].mask, self.data['APOGEE'].mask)
        apogee_flag     = np.logical_and(self.data['HC2000'].mask, ~self.data['APOGEE'].mask)
        matched_flag    = np.logical_and(~self.data['HC2000'].mask, ~self.data['APOGEE'].mask)

        fig_data = [
            # nirspec quiver
            ff.create_quiver(
                self.ra[nirspec_flag].value,
                self.dec[nirspec_flag].value,
                self.pmRA[nirspec_flag].value,
                self.pmDE[nirspec_flag].value,
                name='NIRSPEC Velocity',
                scale=scale,
                line=dict(
                    color='rgba' + str(hex_to_rgb(C0) + (opacity,)),
                    width=line_width
                ),
                showlegend=False
            ).data[0],
            
            # apogee quiver
            ff.create_quiver(
                self.ra[apogee_flag].value,
                self.dec[apogee_flag].value,
                self.pmRA[apogee_flag].value,
                self.pmDE[apogee_flag].value,
                name='APOGEE Velocity',
                scale=scale,
                line=dict(
                    color='rgba' + str(hex_to_rgb(C4) + (opacity,)),
                    width=line_width
                ),
                showlegend=False
            ).data[0],
            
            # matched quiver
            ff.create_quiver(
                self.ra[matched_flag].value,
                self.dec[matched_flag].value,
                self.pmRA[matched_flag].value,
                self.pmDE[matched_flag].value,
                name='Matched Velocity', 
                scale=scale, 
                line=dict(
                    color='rgba' + str(hex_to_rgb(C3) + (opacity,)),
                    width=line_width
                ),
                showlegend=False
            ).data[0],
            
            
            # nirspec scatter
            go.Scatter(
                name='NIRSPEC Sources',
                x=self.ra[nirspec_flag].value, 
                y=self.dec[nirspec_flag].value, 
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=C0
                )
            ),
            
            # apogee scatter
            go.Scatter(
                name='APOGEE Sources',
                x=self.ra[apogee_flag].value, 
                y=self.dec[apogee_flag].value, 
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=C4
                )
            ),
            
            # matched scatter
            go.Scatter(
                name='Matched Sources',
                x=self.ra[matched_flag].value, 
                y=self.dec[matched_flag].value, 
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=C3
                )
            )
        ]

        fig = go.Figure()
        fig.add_traces(fig_data)
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        )

        fig.update_layout(
            width=800,
            height=800,
            xaxis_title="Right Ascension (degree)",
            yaxis_title="Declination (degree)",
        )

        fig.show()
        
        return fig
    
    
    def plot_3d(self, scale=3, show_figure=True, label='Sources'):
        marker_size = 3
        opacity = 0.7
        fig = super().plot_3d(scale, show_figure, label)
        fig.add_trace(
            go.Scatter3d(
                mode='markers',
                name='Trapezium',
                x=[trapezium.cartesian.x.value],
                y=[trapezium.cartesian.y.value],
                z=[trapezium.cartesian.z.value],
                marker=dict(
                    size=marker_size*2,
                    color=C3
                )
            )
        )
        
        if show_figure:
            fig.show()
        return fig


    def plot_pm_rv(self, background_path=f'{user_path}/ONC/figures/skymap/hlsp_orion_hst_acs_colorimage_r_v1_drz.fits', save_path=None, constraint=None):
        """Plot proper motion and radial velocity

        Parameters
        ----------
        background_path : str, optional
            background path, by default f'{user_path}/ONC/figures/skymap/hlsp_orion_hst_acs_colorimage_r_v1_drz.fits'
        save_path : str, optional
            Save path, by default None
        constraint : array-like, optional
            Constraint array, by default None
        """
        hdu = fits.open(background_path)[0]
        wcs = WCS(background_path)
        box_size = 5200
        # Cutout. See https://docs.astropy.org/en/stable/nddata/utils.html.
        cutout = Cutout2D(hdu.data, position=trapezium, size=(box_size, box_size), wcs=wcs)
        a = 1e4
        image_data = ((np.power(a, cutout.data/255) - 1)/a)*255
        # image_data_zoom = (cutout.data/255)**2*255
        image_wcs = cutout.wcs
        ra_wcs, dec_wcs = image_wcs.wcs_world2pix(self.ra.value, self.dec.value, 0)
        
        fig = plt.figure(figsize=(7, 6), dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection=image_wcs)
        ax.imshow(image_data, cmap='gray')
        
        if constraint is None: constraint = np.ones(self.len, dtype=bool)
        im = ax.quiver(
            ra_wcs[constraint],
            dec_wcs[constraint],
            -self.pmRA[constraint].value,   # minius sign because ra increases from right to left
            self.pmDE[constraint].value,
            self.rv[constraint].value,
            cmap='coolwarm',
            width=0.006,
            scale=25
        )
        im.set_clim(vmax=36)
        
        ax.quiverkey(im, X=0.15, Y=0.95, U=1, color='w',
                    label=r'$1~\mathrm{mas}\cdot\mathrm{yr}^{-1}$'
                    '\n'
                    r'$\left(1.844~\mathrm{km}\cdot\mathrm{s}^{-1}\right)$', labelcolor='w', labelpos='S', coordinates='axes')
        
        ax.set_xlabel('Right Ascension', fontsize=12)
        ax.set_ylabel('Declination', fontsize=12)
        ax.set_xlim([0, box_size - 1])
        ax.set_ylim([0, box_size - 1])
        
        cax = fig.colorbar(im, fraction=0.1, shrink=1, pad=0.03)
        cax.set_label(label=r'RV $\left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$', size=12, labelpad=10)
        
        if save_path:
            if save_path.endswith('png'):
                plt.savefig(save_path, bbox_inches='tight', transparent=True)
            else:
                plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    
    
    
    def compare_mass(self, save_path=None):
        fltr = ~(
            np.isnan(self.mass_MIST) | \
            np.isnan(self.mass_BHAC15) | \
            np.isnan(self.mass_Feiden) | \
            np.isnan(self.mass_Palla) | \
            np.isnan(self.mass_Hillenbrand) | \
            ~self.data['theta_orionis'].mask
        )

        mass_corner = np.array([
            self.mass_MIST[fltr],
            self.mass_BHAC15[fltr],
            self.mass_Feiden[fltr],
            self.mass_Palla[fltr],
            self.mass_Hillenbrand[fltr]
        ])
        
        limit = (0.09, 1.35)
        
        fig = corner_plot(
            data=mass_corner.transpose(),
            labels=['MIST', 'BHAC15', 'Feiden', 'Palla', 'Hillenbrand'], 
            limit=limit,
            save_path=save_path
        )




#################################################
################### Functions ###################
#################################################

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


def hex_to_rgb(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in range(0, len(value), 2))


def plot_skymaps(orion, background_path=f'{user_path}/ONC/figures/skymap/hlsp_orion_hst_acs_colorimage_r_v1_drz.fits', ra_offset=-8, dec_offset=-12):
    tobin       = Vizier.get_catalogs('J/ApJ/697/1103/table3')[0]
    tobin.rename_columns(['RAJ2000', 'DEJ2000'], ['_RAJ2000', '_DEJ2000'])
    tobin_coord = SkyCoord([ra + dec for ra, dec in zip(tobin['_RAJ2000'], tobin['_DEJ2000'])], unit=(u.hourangle, u.deg))
    tobin['RAJ2000'] = tobin_coord.ra.value
    tobin['DEJ2000'] = tobin_coord.dec.value
    apogee = Vizier.query_region(trapezium, radius=0.4*u.deg, catalog='III/284/allstars')[0]
    apogee_coord = SkyCoord(ra=apogee['RAJ2000'], dec=apogee['DEJ2000'])
    apogee['sep_to_trapezium'] = apogee_coord.separation(trapezium)
    
    # Wide field view figure
    fig, ax, wcs = orion.plot_skymap(background_path=background_path, show_figure=False, label='NIRSPAO', constraint=~orion.data['HC2000'].mask, zorder=3)
    image_size=18000
    margin=0.05
    # filter
    ra_upper_large, dec_lower_large = wcs.wcs_pix2world(0 - image_size*margin, 0 - image_size*margin, 0)
    ra_lower_large, dec_upper_large = wcs.wcs_pix2world(image_size*(margin + 1) - 1, image_size*(margin + 1) - 1, 0)
    apogee = apogee[
        (apogee['RAJ2000'].value >= ra_lower_large) & (apogee['RAJ2000'].value <= ra_upper_large) & (apogee['DEJ2000'].value >= dec_lower_large) & (apogee['DEJ2000'].value <= dec_upper_large)
    ]
    tobin = tobin[
        (tobin['RAJ2000'].value >= ra_lower_large) & (tobin['RAJ2000'].value <= ra_upper_large) & (tobin['DEJ2000'].value >= dec_lower_large) & (tobin['DEJ2000'].value <= dec_upper_large)
    ]
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
    apogee = apogee[apogee['sep_to_trapezium'] <= 4*u.arcmin]
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


def compare_velocity(orion, save_path=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4))
    ax1.errorbar(orion.data['pmRA_kim'].value, orion.data['pmRA_gaia'].value, xerr=orion.data['e_pmRA_kim'].value, yerr=orion.data['e_pmRA_gaia'].value, fmt='o', color=(.2, .2, .2, .8), alpha=0.4, markersize=3)
    ax1.plot([-2, 3], [-2, 3], color='C3', linestyle='--', label='Equal Line')
    ax1.set_xlabel(r'$\mu_{\alpha^*, HK} \quad \left(\mathrm{mas}\cdot\mathrm{yr}^{-1}\right)$')
    ax1.set_ylabel(r'$\mu_{\alpha^*, DR3} - \widetilde{\Delta\mu_{\alpha^*}} \quad \left(\mathrm{mas}\cdot\mathrm{yr}^{-1}\right)$')
    ax1.legend()
    
    ax2.errorbar(orion.data['pmDE_kim'].value, orion.data['pmDE_gaia'].value, xerr=orion.data['e_pmDE_kim'].value, yerr=orion.data['e_pmDE_gaia'].value, fmt='o', color=(.2, .2, .2, .8), alpha=0.4, markersize=3)
    ax2.plot([-2, 3], [-2, 3], color='C3', linestyle='--', label='Equal Line')
    ax2.set_xlabel(r'$\mu_{\delta, HK} \quad \left(\mathrm{mas}\cdot\mathrm{yr}^{-1}\right)$')
    ax2.set_ylabel(r'$\mu_{\delta, DR3} - \widetilde{\Delta\mu_{\alpha^*}} \quad \left(\mathrm{mas}\cdot\mathrm{yr}^{-1}\right)$')
    ax2.legend()
    
    ax3.errorbar(orion.data['rv_helio'].value, orion.data['rv_apogee'].value, xerr=orion.data['e_rv_nirspao'].value, yerr=orion.data['e_rv_apogee'].value, fmt='o', color=(.2, .2, .2, .8), alpha=0.4, markersize=3)
    ax3.plot([25, 36], [25, 36], color='C3', linestyle='--', label='Equal Line')
    ax3.set_xlabel(r'$\mathrm{RV}_\mathrm{NIRSPAO} \quad \left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$')
    ax3.set_ylabel(r'$\mathrm{RV}_\mathrm{APOGEE} \quad \left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$')
    ax3.legend()
    
    fig.subplots_adjust(wspace=0.28)
    if save_path:
        if save_path.endswith('png'):
            plt.savefig(save_path, bbox_inches='tight', transparent=True)
        else:
            plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def corner_plot(data, labels, limit, save_path=None):
    ndim = np.shape(data)[1]
    fontsize=14
    fig, axs = plt.subplots(ndim, ndim, figsize=(2*ndim, 2*ndim))
    plt.subplots_adjust(hspace=0.06, wspace=0.06)
    plt.locator_params(nbins=3)
    for i in range(ndim):
        for j in range(ndim):
            if j > i:
                axs[i, j].axis('off')
            elif j==i:
                axs[i, j].hist(data[:, i], color='k', range=(0.09, 1.35), bins=20, histtype='step')
                axs[i, j].set_yticks([])
                axs[i, j].set_yticklabels([])
                axs[i, j].set_xlim(limit)
                if i==4:
                    axs[i, j].set_xlabel(labels[-1], fontsize=fontsize)
                    axs[i, j].set_xticklabels(axs[i, j].get_xticks(), rotation=45)
                    axs[i, j].tick_params(axis='both', which='major', labelsize=12)
            else:
                axs[i, j].plot(limit, limit, color='C3', linestyle='--', lw=1.5)
                axs[i, j].scatter(data[:, j], data[:, i], s=10, c='k', edgecolor='none', alpha=0.3)
                axs[i, j].set_xlim(limit)
                axs[i, j].set_ylim(limit)
                
                if i!=4:
                    axs[i, j].set_xticklabels([])
                else:
                    axs[i, j].set_xlabel(labels[j], fontsize=fontsize)
                    axs[i, j].set_xticklabels(axs[i, j].get_xticks(), rotation=45)
                    axs[i, j].tick_params(axis='both', which='major', labelsize=12)
                if j!=0:
                    axs[i, j].set_yticklabels([])
                else:
                    axs[i, j].set_ylabel(labels[i], fontsize=fontsize)
                    axs[i, j].set_yticklabels(axs[i, j].get_yticks(), rotation=45)
                    axs[i, j].tick_params(axis='both', which='major', labelsize=12)
    
    if save_path:
        if save_path.endswith('png'):
            plt.savefig(save_path, bbox_inches='tight', transparent=True)
        else:
            plt.savefig(save_path, bbox_inches='tight')
    return fig



#################################################
################# Main Function #################
#################################################

C0 = '#1f77b4'
C1 = '#ff7f0e'
C3 = '#d62728'
C4 = '#9467bd'
C6 = '#e377c2'
C7 = '#7f7f7f'
C9 = '#17becf'

orion = ONC(QTable.read(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog - epoch combined.ecsv'))
orion.preprocessing()
orion.set_attr()
unique, count = np.unique(orion.data['HC2000'], return_counts=True)
dup_idx = [list(orion.data['HC2000']).index(_) for _ in list(unique[(count > 1) & (count < 10)])]
not_dup = np.ones(orion.len, dtype=bool)
not_dup[dup_idx] = False

# plot_skymaps(orion)

# orion.coord_3d = SkyCoord(
#     ra=orion.ra,
#     dec=orion.dec,
#     pm_ra_cosdec=orion.pmRA,
#     pm_dec=orion.pmDE,
#     radial_velocity=orion.rv,
#     distance=1000/orion.data['plx'].value * u.pc
# )

# fig = orion.plot_2d()
# fig.write_html(f'{user_path}/ONC/figures/sky 2d.html')
# fig = orion.plot_3d()
# fig.write_html(f'{user_path}/ONC/figures/sky 3d.html')

# compare_velocity(orion)
# orion.plot_pm_rv(constraint=orion.data['theta_orionis'].mask)
# orion.pm_angle_distribution(constraint=(orion.data['theta_orionis'].mask) & not_dup)
orion.compare_mass()