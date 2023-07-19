import os
import copy
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.ticker as mticker
from typing import Tuple
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.table import QTable, MaskedColumn
from astropy.visualization.wcsaxes import SphericalCircle
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import AnchoredText
from fit_vdisp import fit_vdisp

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
            if self.data[key].dtype.name.startswith('str') or self.data[key].dtype.name.startswith('int') or self.data[key].dtype.name.startswith('object'):
                continue
            try:
                self.data[key] = self.data[key].filled(np.nan)
            except:
                pass

    @property
    def len(self):
        return len(self.data)    
    
    
    def set_coord(self, ra, dec, pmRA, pmDE, rv, distance=None):
        """Set astropy SkyCoord with attributes: coord (and velocity, if distance is not None)

        Parameters
        ----------
        ra : astropy quantity
            Right ascension
        dec : astropy quantity
            Declination
        pmRA : astropy quantity
            Proper motion in right ascension
        pmDE : astropy quantity
            Proper motion in declination
        rv : astropy quantity
            Radial velocity
        distance : astropy quantity, optional
            Distance to sources, by default None
        """
        if distance is not None:
            self.coord = SkyCoord(
                ra=ra,
                dec=dec,
                pm_ra_cosdec=pmRA,
                pm_dec=pmDE,
                radial_velocity=rv,
                distance=distance
            )
            self.v_xyz = self.coord.velocity.d_xyz
        else:
            self.coord = SkyCoord(
                ra=ra,
                dec=dec,
                pm_ra_cosdec=pmRA,
                pm_dec=pmDE,
                radial_velocity=rv
            )
    
    
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
    
    
    def calculate_velocity(self, pmRA, e_pmRA, pmDE, e_pmDE, rv, e_rv, dist, e_dist):
        """Calculate velocity from proper motions, radial velocity, and distance.
        StarCluster object must have attributes pmRA, e_pmRA, pmDE, e_pmDE.
        Add columns in data: vRA, e_vRA, vDE, e_vDE, vt, e_vt, v, e_v

        Parameters
        ----------
        pmRA : astropy quantity
            Proper motion in right ascension
        e_pmRA : astropy quantity
            Proper motion uncertainty in right ascension
        pmDE : astropy quantity
            Proper motion in declination
        e_pmDE : astropy quantity
            Proper motion uncertainty in declination
        rv : astropy quantity
            Radial velocity
        e_rv : astropy quantity
            Radial velocity uncertainty
        dist : astropy quantity
            Distance to sources
        e_dist : astropy quantity
            Distance uncertainty
        """
        
        vRA = (pmRA * dist).to(u.rad * u.km/u.s).value * u.km/u.s
        e_vRA = np.sqrt((e_pmRA * dist)**2 + (e_dist * pmRA)**2).to(u.rad * u.km/u.s).value * u.km/u.s
        
        vDE = (pmDE * dist).to(u.rad * u.km/u.s).value * u.km/u.s
        e_vDE = np.sqrt((e_pmDE * dist)**2 + (e_dist * pmDE)**2).to(u.rad * u.km/u.s).value * u.km/u.s
        
        pm = np.sqrt(pmRA**2 + pmDE**2)
        e_pm = 1/pm * np.sqrt((pmRA * e_pmRA)**2 + (pmDE * e_pmDE)**2)
        
        vt = (pm * dist).to(u.rad * u.km/u.s).value * u.km/u.s
        e_vt = vt * np.sqrt((e_pm / pm)**2 + (e_dist / dist)**2)
        
        vr = rv
        e_vr = e_rv
        
        v = np.sqrt(vt**2 + vr**2)
        e_v = 1/v * np.sqrt((vt * e_vt)**2 + (vr * e_vr)**2)
        
        self.data['vRA'] = vRA
        self.data['e_vRA'] = e_vRA
        self.data['vDE'] = vDE
        self.data['e_vDE'] = e_vDE
        self.data['vt'] = vt
        self.data['e_vt'] = e_vt
        self.data['v'] = v
        self.data['e_v'] = e_v


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

    
    def pm_angle_distribution(self, save_path=None):
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
        
        # if constraint is None: constraint = np.ones(self.len, dtype=bool)
        position = np.array([
            -(self.ra - trapezium.ra).value,   # minius sign because ra increases from right to left
            (self.dec - trapezium.dec).value
        ])

        pm = np.array([
            -self.pmRA.value,
            self.pmDE.value
        ])
        
        # After adding minus sign: clock-wise is positive, counter clock-wise is negative
        angles = -np.array([angle_between(position[:, i], pm[:, i]) for i in range(np.shape(position)[1])])

        nbins = 12
        angles[angles > np.pi - np.pi/nbins] = angles[angles > np.pi - np.pi/nbins] - 2*np.pi
        hist, bin_edges = np.histogram(angles, nbins, range=(-np.pi - np.pi/nbins, np.pi - np.pi/nbins))
        # hist, bin_edges = np.histogram(angles, nbins, range=(-np.pi, np.pi))
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
    
    
    def vrel_vs_mass(self, model_name, radius=0.1*u.pc, model_type='linear', resampling=100000, self_included=True, min_rv=-np.inf*u.km/u.s, max_rv=np.inf*u.km/u.s, max_v_error=5.*u.km/u.s, max_mass_error=0.5*u.solMass, kde_percentile=84, update_self=False, save_path=None, show_figure=True, bin_method='equally grouped', **kwargs):
        """Velocity relative to the neighbors of each source within a radius vs mass.

        Parameters
        ----------
        model_name : str
            One of ['MIST', 'BHAC15', 'Feiden', 'Palla']
        radius : astropy.Quantity, optional
            Radius within which count as neighbors, by default 0.1*u.pc
        model_func : str, optional
            Format of model function: 'linear' or 'power'. V=k*M + b or V=A*M**k, by default 'linear'.
        resampling : any, optional
            Whether resample or not, and number of resamples, 100000 by default. If False or None, will try to read from previous results
        self_included : bool, optional
            Include the source itself or not when calculating the center of mass velocity of its neighbors, by default True
        min_rv : astropy quantity, optional
            Maximum radial velocity, by default inf.
        max_rv : astropy quantity, optional
            Maximum radial velocity, by default inf.
        max_rv_error : astropy quantity, optional
            Maximum radial velocity error, by default 5
        max_mass_error : astropy quantity, optional
            Maximum mass error, by default 0.5
        update_self : bool, optional
            Update the original sources dataframe or not, by default False
        kde_percentile : int, optional
            Percentile of KDE contour, 84 by default
        show_figure : bool, optional
            Whether to show the figure, by default True
        save_path : str, optional
            Save path, by default None
        bin_method: str, optional
            Binning method when calculating running average, 'equally spaced' or 'equally grouped', by default 'equally grouped'
        kwargs:
            nbins: int, optional
                Number of bins, by default 7 for 'equally grouped' and 5 for 'equally spaced'.

        Returns
        -------
        mass, vrel, mass_e, vrel_e, fit_result
            mass, vrel, mass_e, vrel_e: 1-D array.
            fit_result: pd.DataFrame with keys 'k', 'b'.

        Raises
        ------
        ValueError
            bin_method must be one of 'equally spaced' or 'equally grouped'.
        """
        
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        
        mass = self.data[f'mass_{model_name}']
        e_mass = self.data[f'e_mass_{model_name}']
        
        constraint = \
            (~np.isnan(self.data['pmRA'])) & (~np.isnan(self.data['pmDE'])) & \
            (~np.isnan(mass)) & \
            (e_mass <= max_mass_error) & \
            (self.data['rv'] >= min_rv) & (self.data['rv'] <= max_rv) & \
            (self.data['e_v'] <= max_v_error)
        
        sources_coord = self.coord[constraint]
        
        mass = mass[constraint]
        e_mass = e_mass[constraint]
        e_v = self.data['e_v'][constraint]
        
        ############# calculate vcom within radius #############
        # v & vcom: n-by-3 velocity in cartesian coordinates
        v = self.v_xyz.T[constraint, :]
        vcom = np.empty((sum(constraint), 3))*u.km/u.s
        e_vcom = np.empty(sum(constraint))*u.km/u.s
        
        # is_neighbor: boolean symmetric neighborhood matrix
        is_neighbor = np.empty((len(sources_coord), len(sources_coord)), dtype=bool)
        
        for i, star in enumerate(sources_coord):
            sep = star.separation_3d(sources_coord)
            is_neighbor[i] = sep < radius
            if not self_included:
                is_neighbor[i, i] = False
            # vel_com[i]: 1-by-3 center of mass velocity
            vcom[i] = (mass[is_neighbor[i]] @ v[is_neighbor[i]]) / sum(mass[is_neighbor[i]])
        
        n_neighbors = np.sum(is_neighbor, axis=1)
        
        # delete those without any neighbors
        if self_included:
            has_neighbor = n_neighbors > 1
        else:
            has_neighbor = n_neighbors > 0
            
        v = v[has_neighbor, :]
        vcom = vcom[has_neighbor, :]
        e_vcom = e_vcom[has_neighbor]
        # is_neighbor = np.delete(is_neighbor, has_neighbor, axis=1)
        is_neighbor = is_neighbor[has_neighbor, :][:, has_neighbor]
        n_neighbors = n_neighbors[has_neighbor]
        valid_idx = np.where(constraint)[0][has_neighbor]
        
        sources_coord = sources_coord[has_neighbor]
        
        mass = mass[has_neighbor]
        e_mass = e_mass[has_neighbor]
        e_v = e_v[has_neighbor]
        
        print(f'Median neighbors in a group: {np.median(n_neighbors):.0f}')
        
        vrel_vector = v - vcom
        vrel = np.linalg.norm(vrel_vector, axis=1)
        
        ############# Calculate vrel error #############
        for i in range(len(sources_coord)):
            vcom_e_j = [sum((vrel_vector[is_neighbor[i], j]/sum(mass[is_neighbor[i]]) * e_mass[is_neighbor[i]])**2 + (mass[is_neighbor[i]] / sum(mass[is_neighbor[i]]) * e_v[is_neighbor[i]])**2)**0.5 for j in range(3)]
            e_vcom[i] = np.sqrt(sum([(vcom[i,j] / np.linalg.norm(vcom[i]) * vcom_e_j[j])**2 for j in range(3)]))
        
        e_vrel = np.sqrt(e_v**2 + e_vcom**2)
        
        
        ############# Resampling #############
        R = np.corrcoef(mass.to(u.solMass).value, vrel.to(u.km/u.s).value)[1, 0]   # Pearson's R
        
        if model_type=='linear':
            def model_func(x, k, b):
                return k*x + b
        elif model_type=='power':
            def model_func(x, k, b):
                return b*x**k
        else:
            raise ValueError(f"model_func must be one of 'linear' or 'power', not {model_func}.")

        # Resampling
        if resampling is True:
            resampling = 100000
        if resampling:
            ks = np.empty(resampling)
            bs = np.empty(resampling)
            Rs = np.empty(resampling)
            
            if model_type=='linear':
                for i in range(resampling):
                    mass_resample = np.random.normal(loc=mass.to(u.solMass).value, scale=e_mass.to(u.solMass).value)
                    vrel_resample = np.random.normal(loc=vrel.to(u.km/u.s).value, scale=e_vrel.to(u.km/u.s).value)
                    valid_resample_idx = (mass_resample > 0) & (vrel_resample > 0)
                    mass_resample = mass_resample[valid_resample_idx]
                    vrel_resample = vrel_resample[valid_resample_idx]
                    result = linregress(mass_resample, vrel_resample)
                    ks[i] = result.slope
                    bs[i] = result.intercept
                    # popt, pcov = curve_fit(model_func, mass_resample, vrel_resample)
                    # ks[i] = popt[0]
                    # bs[i] = popt[1]
                    Rs[i] = np.corrcoef(mass_resample, vrel_resample)[1, 0]
                
            elif model_type=='power':
                for i in range(resampling):
                    mass_resample = np.random.normal(loc=mass.to(u.solMass).value, scale=e_mass.to(u.solMass).value)
                    vrel_resample = np.random.normal(loc=vrel.to(u.km/u.s).value, scale=e_vrel.to(u.km/u.s).value)
                    valid_resample_idx = (mass_resample > 0) & (vrel_resample > 0)
                    mass_resample = mass_resample[valid_resample_idx]
                    vrel_resample = vrel_resample[valid_resample_idx]
                    result = linregress(np.log10(mass_resample), np.log10(vrel_resample))
                    ks[i] = result.slope
                    bs[i] = result.intercept
                    # popt, pcov = curve_fit(model_func, mass_resample, vrel_resample)
                    # ks[i] = popt[0]
                    # bs[i] = popt[1]
                    Rs[i] = np.corrcoef(mass_resample, vrel_resample)[1, 0]
            
            k_resample = np.median(ks)
            k_e = np.diff(np.percentile(ks, [16, 84]))[0]/2
            b_resample = np.median(bs)
            b_e = np.diff(np.percentile(bs, [16, 84]))[0]/2
            R_resample = np.median(Rs)
            R_e = np.diff(np.percentile(Rs, [16, 84]))[0]/2
        
        else:
            with open(f'{save_path}/{model_name}-{model_type}-{radius.value:.2f}pc params.txt', 'r') as file:
                raw = file.readlines()
            for line in raw:
                if line.startswith('k_resample:'):
                    k_resample, k_e = eval(', '.join(line.strip('k_resample:\t\n').split('± ')))
                elif line.startswith('b_resample:'):
                    b_resample, b_e = eval(', '.join(line.strip('b_resample:\t\n').split('± ')))
                elif line.startswith('R_resample:'):
                    R_resample, R_e = eval(', '.join(line.strip('R_resample:\t\n').split('± ')))
        
        # p-value
        result = linregress(mass, vrel)
        p = result.pvalue
        
        print(f'k_resample = {k_resample:.2f} ± {k_e:.2f}')
        print(f'p = {p:.2E}')
        print(f'R = {R:.2f}, R_resample = {R_resample:.2f}')
        
        # write params
        if save_path:
            with open(f'{save_path}/{model_name}-{model_type}-{radius.value:.2f}pc params.txt', 'w') as file:
                file.write(f'Median of neighbors in a group:\t{np.median(n_neighbors):.0f}\n')
                file.write(f'k_resample:\t{k_resample} ± {k_e}\n')
                file.write(f'b_resample:\t{b_resample} ± {b_e}\n')
                file.write(f'p:\t{p}\n')
                file.write(f'R_resample:\t{R_resample} ± {R_e}\n')
                file.write(f'R:\t{R}\n')
        
        ############# Running average #############
        # equally grouped
        if bin_method == 'equally grouped':
            nbins = kwargs.get('nbins', 7)
            sources_in_bins = [len(mass) // nbins + (1 if x < len(valid_idx) % nbins else 0) for x in range (nbins)]
            division_idx = np.cumsum(sources_in_bins)[:-1] - 1
            mass_sorted = np.sort(mass)
            mass_borders = np.array([np.nanmin(mass).to(u.solMass).value - 1e-3, *(mass_sorted[division_idx] + mass_sorted[division_idx + 1]).to(u.solMass).value/2, np.nanmax(mass).to(u.solMass).value])
        
        # equally spaced
        elif bin_method == 'equally spaced':
            nbins = kwargs.get('nbins', 5)
            mass_borders = np.linspace(np.nanmin(mass).to(u.solMass).value - 1e-3, np.nanmax(mass).to(u.solMass).value, nbins + 1)
        
        else:
            raise ValueError("bin_method must be one of the following: ['equally grouped', 'equally spaced']")
        
        mass_value      = mass.to(u.solMass).value
        e_mass_value    = e_mass.to(u.solMass).value
        vrel_value      = vrel.to(u.km/u.s).value
        e_vrel_value    = e_vrel.to(u.km/u.s).value
        mass_binned_avrg    = np.empty(nbins)
        e_mass_binned       = np.empty(nbins)
        mass_weight = 1 / e_mass_value**2
        vrel_binned_avrg    = np.empty(nbins)
        e_vrel_binned       = np.empty(nbins)
        vrel_weight = 1 / e_vrel_value**2
        
        for i, min_mass, max_mass in zip(range(nbins), mass_borders[:-1]*u.solMass, mass_borders[1:]*u.solMass):
            idx = (mass > min_mass) & (mass <= max_mass)
            mass_weight_sum = sum(mass_weight[idx])
            mass_binned_avrg[i] = np.average(mass_value[idx], weights=mass_weight[idx])
            e_mass_binned[i] = 1/mass_weight_sum * sum(mass_weight[idx] * e_mass_value[idx])
            
            vrel_weight_sum = sum(vrel_weight[idx])
            vrel_binned_avrg[i] = np.average(vrel_value[idx], weights=vrel_weight[idx])
            e_vrel_binned[i] = 1/vrel_weight_sum * sum(vrel_weight[idx] * e_vrel_value[idx])
        
        
        ########## Kernel Density Estimation in Linear Space ##########
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        resolution = 200
        X, Y = np.meshgrid(np.linspace(0, mass_value.max(), resolution), np.linspace(0, vrel_value.max(), resolution))
        positions = np.vstack([X.T.ravel(), Y.T.ravel()])
        values = np.vstack([mass_value, vrel_value])
        kernel = stats.gaussian_kde(values)
        Z = np.rot90(np.reshape(kernel(positions).T, X.shape))
        
        
        ########## Linear Fit Plot - Original Error ##########
        xs = np.linspace(mass_value.min(), mass_value.max(), 100)
        
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
        
        if model_type=='linear':
            # Errorbar with uniform transparency
            h1 = ax.errorbar(
                mass_value, vrel_value, xerr=e_mass_value, yerr=e_vrel_value,
                fmt='.', 
                markersize=6, markeredgecolor='none', markerfacecolor='C0', 
                elinewidth=1, ecolor='C0', alpha=0.5,
                zorder=2
            )
            
            # Running Average
            h2 = ax.errorbar(
                mass_binned_avrg, vrel_binned_avrg, 
                xerr=e_mass_binned, 
                yerr=e_vrel_binned, 
                fmt='.', 
                markersize=8, markeredgecolor='none', markerfacecolor='C3', 
                elinewidth=1.2, ecolor='C3', 
                alpha=0.8,
                zorder=4
            )
        
            # Running Average Fill
            f2 = ax.fill_between(mass_binned_avrg, vrel_binned_avrg - e_vrel_binned, vrel_binned_avrg + e_vrel_binned, color='C3', edgecolor='none', alpha=0.5)
            
            # Model
            h4, = ax.plot(xs, k_resample*xs + b_resample, color='k', label='Best Fit', zorder=3)
        
        
        elif model_type=='power':
            mass_log = np.log10(mass_value)
            vrel_log = np.log10(vrel_value)
            e_mass_log = 1/(np.log(10) * mass_value) * e_mass_value
            e_vrel_log = 1/(np.log(10) * vrel_value) * e_vrel_value
            h1 = ax.errorbar(
                mass_value, vrel_value, 
                xerr=np.array([mass_value - 10**(mass_log - e_mass_log), 10**(mass_log + e_mass_log) - mass_value]), 
                yerr=np.array([vrel_value - 10**(vrel_log - e_vrel_log), 10**(vrel_log + e_vrel_log) - vrel_value]), 
                fmt='.', 
                markersize=6, markeredgecolor='none', markerfacecolor='C0', 
                elinewidth=1, ecolor='C0', alpha=0.5,
                zorder=2
            )
            
            mass_avrg_log = np.log10(mass_binned_avrg)
            vrel_avrg_log = np.log10(vrel_binned_avrg)
            e_mass_avrg_log = 1/(np.log(10) * mass_binned_avrg) * e_mass_binned
            e_vrel_avrg_log = 1/(np.log(10) * vrel_binned_avrg) * e_vrel_binned
            # Running Average
            h2 = ax.errorbar(
                mass_binned_avrg, vrel_binned_avrg, 
                xerr=np.array([mass_binned_avrg - 10**(mass_avrg_log - e_mass_avrg_log), 10**(mass_avrg_log + e_mass_avrg_log) - mass_binned_avrg]), 
                yerr=np.array([vrel_binned_avrg - 10**(vrel_avrg_log - e_vrel_avrg_log), 10**(vrel_avrg_log + e_vrel_avrg_log) - vrel_binned_avrg]), 
                fmt='.', 
                markersize=8, markeredgecolor='none', markerfacecolor='C3', 
                elinewidth=1.2, ecolor='C3', 
                alpha=0.8,
                zorder=4
            )
            
            # Running Average Fill
            f2 = ax.fill_between(mass_binned_avrg, 10**(vrel_avrg_log - e_vrel_avrg_log), 10**(vrel_avrg_log + e_vrel_avrg_log), color='C3', edgecolor='none', alpha=0.5)
            
            # Model
            h4, = ax.plot(xs, 10**b_resample * xs**k_resample, color='k', label='Best Fit', zorder=3)
            
            plt.loglog()
        
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        if model_type=='power':
            if xlim[0] < 0.018: xlim = (0.018, xlim[1])
            if ylim[0] < 0.3: ylim = (0.3, ylim[1])
        
        # Plot KDE and contours
        # see https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_demo.html
        
        # Choose colormap
        cmap = plt.cm.Blues

        # set from white to half-blue
        my_cmap = cmap(np.linspace(0, 0.5, cmap.N))

        # Create new colormap
        my_cmap = ListedColormap(my_cmap)
        
        ax.set_facecolor(cmap(0))
        im = ax.imshow(Z, cmap=my_cmap, alpha=0.8, extent=[0, mass_value.max(), 0, vrel_value.max()], zorder=0, aspect='auto')
        cs = ax.contour(X, Y, np.flipud(Z), levels=np.percentile(Z, [kde_percentile]), alpha=0.5, zorder=1)
        
        # contour label
        fmt = {cs.levels[0]: f'{kde_percentile}%'}
        ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)
        h3 = Line2D([0], [0], color=cs.collections[0].get_edgecolor()[0])
        
        cax = fig.colorbar(im, fraction=0.1, shrink=1, pad=0.03)
        cax.set_label(label='KDE', size=12, labelpad=10)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        handles, labels = ax.get_legend_handles_labels()
        handles = [h1, (h2, f2), h3, h4]
        labels = [
            f'Sources - {model_name} Model',
            'Running Average',
            f"KDE's {kde_percentile}-th Percentile"
        ]
        
        labels.append(f'Best Fit:\n$k={k_resample:.2f}\pm{k_e:.2f}$\n$b={b_resample:.2f}\pm{b_e:.2f}$')
        
        ax.legend(handles, labels)
        
        
        def sci_notation(number, sig_fig=2):
            ret_string = "{0:.{1:d}e}".format(number, sig_fig)
            a, b = ret_string.split("e")
            # remove leading "+" and strip leading zeros
            b = int(b)
            return f'{a}*10^{{{str(b)}}}'
        
        at = AnchoredText(
            f'$p={sci_notation(p)}$\n$R={R_resample:.2f}\pm{R_e:.2f}$', 
            prop=dict(size=10), frameon=True, loc='lower right'
        )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        at.patch.set_alpha(0.8)
        at.patch.set_edgecolor((0.8, 0.8, 0.8))
        ax.add_artist(at)
        
        ax.set_xlabel('Mass $(M_\odot)$', fontsize=12)
        ax.set_ylabel('Relative Velocity (km$\cdot$s$^{-1}$)', fontsize=12)
        
        if save_path:
            if save_path.endswith('png'):
                plt.savefig(f'{save_path}/{model_name}-{model_type}-{radius.value:.2f}pc.pdf', bbox_inches='tight', transparent=True)
            else:
                plt.savefig(f'{save_path}/{model_name}-{model_type}-{radius.value:.2f}pc.pdf', bbox_inches='tight')
        
        if show_figure:
            plt.show()
        else:
            plt.close()
        
        ########## Updating the original DataFrame ##########
        if update_self:
            print('Updating self...')
            self.data[f'vrel_{model_name}'] = np.nan*vrel.unit
            self.data[f'vrel_{model_name}'][valid_idx] = vrel
            self.data[f'e_vrel_{model_name}'] = np.nan*e_vrel.unit
            self.data[f'e_vrel_{model_name}'][valid_idx] = e_vrel
            self.vrel = self.data[f'vrel_{model_name}']
            self.e_vrel = self.data[f'e_vrel_{model_name}']
        else:
            pass
        
        return mass, vrel, e_mass, e_vrel





class ONC(StarCluster):
    def __init__(self, table) -> None:
        super().__init__(table)
    
    def set_attr(self) -> None:
        """Set attributes of ra, dec, pmRA, pmDE rv, coord, teff, teff, mass of MIST, BHAC15, Feiden, Palla, and corresponding errors.
        """
        self.ra = self.data['RAJ2000']
        self.dec = self.data['DEJ2000']
        self.pmRA = self.data['pmRA']
        self.e_pmRA = self.data['e_pmRA']
        self.pmDE = self.data['pmDE']
        self.e_pmDE = self.data['e_pmDE']
        self.rv = self.data['rv']
        self.e_rv = self.data['e_rv']
        self.teff = self.data['teff']
        self.e_teff = self.data['e_teff']
        self.vRA    = self.data['vRA']
        self.e_vRA  = self.data['e_vRA']
        self.vDE    = self.data['vDE']
        self.e_vDE  = self.data['e_vDE']
        self.vt     = self.data['vt']
        self.e_vt   = self.data['e_vt']
        self.v      = self.data['v']
        self.e_v    = self.data['e_v']
        self.mass_MIST = self.data['mass_MIST']
        self.e_mass_MIST = self.data['e_mass_MIST']
        self.mass_BHAC15 = self.data['mass_BHAC15']
        self.e_mass_BHAC15 = self.data['e_mass_BHAC15']
        self.mass_Feiden = self.data['mass_Feiden']
        self.e_mass_Feiden = self.data['e_mass_Feiden']
        self.mass_Palla = self.data['mass_Palla']
        self.e_mass_Palla = self.data['e_mass_Palla']
        self.mass_Hillenbrand = self.data['mass_Hillenbrand']
        self.sep_to_trapezium = self.data['sep_to_trapezium']
    
    
    def preprocessing(self):
        trapezium_only = (self.data['sci_frames'].mask) & (self.data['APOGEE'].mask)
        unique_HC2000, counts_HC2000 = np.unique(self.data['HC2000'][~self.data['sci_frames'].mask], return_counts=True)
        unique_APOGEE, counts_APOGEE = np.unique(self.data['APOGEE'][~self.data['APOGEE'].mask], return_counts=True)
        print('Before any constraint:\nNIRSPAO: {} (multiples: {}->{})\nAPOGEE: {} (multiples: {}->{})\nMatched: {}\nTotal: {} (including multiples: {})'.format(
            len(unique_HC2000),
            list(unique_HC2000[counts_HC2000 > 1]),
            list(counts_HC2000[counts_HC2000 > 1]),
            len(unique_APOGEE),
            list(unique_APOGEE[counts_APOGEE > 1]),
            list(counts_APOGEE[counts_APOGEE > 1]),
            sum((~self.data['sci_frames'].mask) & (~self.data['APOGEE'].mask)),
            self.len - sum(trapezium_only) - (sum(counts_HC2000[counts_HC2000 > 1]) - len(counts_HC2000[counts_HC2000 > 1])),
            self.len - sum(trapezium_only)
        ))
        
        max_rv_e = 5*u.km/u.s
        rv_constraint = ((
            (self.data['e_rv_nirspao']   <= max_rv_e) |
            (self.data['e_rv_apogee']    <= max_rv_e)
        ) | (
            trapezium_only
        ))
        unique_HC2000_after_rv, counts_HC2000_after_rv = np.unique(self.data['HC2000'][rv_constraint & ~self.data['HC2000'].mask], return_counts=True)
        print(f"Multiples after RV constraint: {list(unique_HC2000_after_rv[counts_HC2000_after_rv > 1])}->{list(counts_HC2000_after_rv[counts_HC2000_after_rv > 1])}")
        print(f"Maximum RV error of {max_rv_e} constraint: {sum(rv_constraint) - sum(trapezium_only) - (sum(counts_HC2000_after_rv[counts_HC2000_after_rv > 1]) - len(counts_HC2000_after_rv[counts_HC2000_after_rv > 1]))} out of {len(rv_constraint) - sum(trapezium_only) - (sum(counts_HC2000[counts_HC2000 > 1]) - len(counts_HC2000[counts_HC2000 > 1]))} remaining.")
        self.data = self.data[rv_constraint]
        
        # Plot RV comparison
        # read multiepoch rv of HC2000 546
        sources = QTable.read(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog.ecsv')
        idx = np.where(sources['HC2000']==546)[0]
        parenago_rv_apogee = sources['rv_apogee'][idx[0]].unmasked * np.ones_like(idx)
        parenago_e_rv_apogee = sources['e_rv_apogee'][idx[0]].unmasked * np.ones_like(idx)
        parenago_rv_nirspao = sources['rv_helio'][idx].unmasked
        parenago_e_rv_nirspao = sources['e_rv_nirspao'][idx].unmasked

        parenago_idx = (self.data['HC2000']==546).filled(False)
        V1337_idx = (self.data['HC2000']==214).filled(False)
        brun_idx = (self.data['HC2000']==172).filled(False)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.errorbar(self.data['rv_helio'][~parenago_idx & ~V1337_idx & ~brun_idx].value, self.data['rv_apogee'][~parenago_idx & ~V1337_idx & ~brun_idx].value, xerr=self.data['e_rv_nirspao'][~parenago_idx & ~V1337_idx & ~brun_idx].value, yerr=self.data['e_rv_apogee'][~parenago_idx & ~V1337_idx & ~brun_idx].value, fmt='o', color=(.2, .2, .2, .8), alpha=0.4, markersize=3)
        ax.errorbar(parenago_rv_nirspao.value, parenago_rv_apogee.value, xerr=parenago_e_rv_nirspao.value, yerr=parenago_e_rv_apogee.value, ls='none', color='C0', alpha=0.5, marker='.', markersize=3, label='Parenago 1837')
        ax.errorbar(self.data['rv_helio'][V1337_idx].value, self.data['rv_apogee'][V1337_idx].value, xerr=self.data['e_rv_nirspao'][V1337_idx].value, yerr=self.data['e_rv_apogee'][V1337_idx].value, ls='none', color='C1', alpha=0.5, marker='s', markersize=6, label='V* V1337 Ori')
        ax.errorbar(self.data['rv_helio'][brun_idx].value, self.data['rv_apogee'][brun_idx].value, xerr=self.data['e_rv_nirspao'][brun_idx].value, yerr=self.data['e_rv_apogee'][V1337_idx].value, ls='none', color='C2', alpha=0.5, marker='d', markersize=6, label='Brun 590')
        ax.plot([23, 38], [23, 38], color='C3', linestyle='--', label='Equal Line')
        ax.set_xlabel(r'$\mathrm{RV}_\mathrm{NIRSPAO} \quad \left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$', fontsize=12)
        ax.set_ylabel(r'$\mathrm{RV}_\mathrm{APOGEE} \quad \left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$', fontsize=12)
        ax.legend()
        plt.savefig(f'{user_path}/ONC/figures/RV Comparison.pdf')
        plt.show()
        
        # update rv_nirspao with rv_apogee where the uncertainty is smaller in apogee
        rv_use_apogee = (self.data['e_rv_nirspao'] > max_rv_e) & (self.data['e_rv_apogee'] <= max_rv_e)
        self.data['rv_helio'][rv_use_apogee] = self.data['rv_apogee'][rv_use_apogee]
        self.data['e_rv_nirspao'][rv_use_apogee] = self.data['e_rv_apogee'][rv_use_apogee]
        
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
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.errorbar(
            (self.data['pmRA_gaia'] - self.data['pmRA_kim'] - offset_RA).value,
            (self.data['pmDE_gaia'] - self.data['pmDE_kim'] - offset_DE).value,
            xerr = ((self.data['e_pmRA_gaia']**2 + self.data['e_pmRA_kim']**2)**0.5).value,
            yerr = ((self.data['e_pmDE_gaia']**2 + self.data['e_pmDE_kim']**2)**0.5).value,
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
        self.data['rv']     = merge(self.data['rv_helio'], self.data['rv_apogee'])
        self.data['e_rv']   = merge(self.data['e_rv_nirspao'], self.data['e_rv_apogee'])
        self.data['dist']   = (1000/self.data['plx'].to(u.mas).value) * u.pc
        self.data['e_dist'] = self.data['e_plx'] / self.data['plx'] * self.data['dist']
        
        dist_constraint = distance_cut(self.data['dist'], self.data['e_dist'])
        unique_HC2000_after_dist, counts_HC2000_after_dist = np.unique(self.data['HC2000'][dist_constraint & ~self.data['HC2000'].mask], return_counts=True)
        print(f'Multiples after distance constraint: {list(unique_HC2000_after_dist[counts_HC2000_after_dist > 1])}->{list(counts_HC2000_after_dist[counts_HC2000_after_dist > 1])}')
        print(f'{300*u.pc}~{500*u.pc} distance range constraint: {sum(dist_constraint) - sum(trapezium_only) - (sum(counts_HC2000_after_dist[counts_HC2000_after_dist > 1]) - len(counts_HC2000_after_dist[counts_HC2000_after_dist > 1]))} sources out of {len(dist_constraint) - sum(trapezium_only) - (sum(counts_HC2000_after_rv[counts_HC2000_after_rv > 1]) - len(counts_HC2000_after_rv[counts_HC2000_after_rv > 1]))} remains.')
        self.data.remove_rows(~dist_constraint)
        
        # Calculate velocity
        self.calculate_velocity(self.data['pmRA'], self.data['e_pmRA'], self.data['pmDE'], self.data['e_pmDE'], self.data['rv'], self.data['e_rv'], dist=389*u.pc, e_dist=3*u.pc)
        
        super().set_coord(ra=self.data['RAJ2000'], dec=self.data['DEJ2000'], pmRA=self.data['pmRA'], pmDE=self.data['pmDE'], rv=self.data['rv'], distance=389*u.pc)
        self.data['sep_to_trapezium'] = self.coord.separation(trapezium)
        
        trapezium_only = (self.data['sci_frames'].mask) & (self.data['APOGEE'].mask)
        unique_HC2000, counts_HC2000 = np.unique(self.data['HC2000'][~self.data['sci_frames'].mask], return_counts=True)
        unique_APOGEE, counts_APOGEE = np.unique(self.data['APOGEE'][~self.data['APOGEE'].mask], return_counts=True)
        print('After all constraint:\nNIRSPAO: {} (multiples: {}->{})\nAPOGEE: {} (multiples: {}->{})\nMatched: {}\nTotal: {} (including multiples: {})'.format(
            len(unique_HC2000),
            list(unique_HC2000[counts_HC2000 > 1]),
            list(counts_HC2000[counts_HC2000 > 1]),
            len(unique_APOGEE),
            list(unique_APOGEE[counts_APOGEE > 1]),
            list(counts_APOGEE[counts_APOGEE > 1]),
            sum((~self.data['sci_frames'].mask) & (~self.data['APOGEE'].mask)),
            self.len - sum(trapezium_only) - (sum(counts_HC2000[counts_HC2000 > 1]) - len(counts_HC2000[counts_HC2000 > 1])),
            self.len - sum(trapezium_only)
        ))

        self.data.write(f'{user_path}/ONC/starrynight/catalogs/sources post-processing.csv', overwrite=True)
        self.data.write(f'{user_path}/ONC/starrynight/catalogs/sources post-processing.ecsv', overwrite=True)
    
    
    
    def plot_skymap(self, circle=4.*u.arcmin, zoom=False, background_path=None, show_figure=True, label='Sources', color='C6', lw=1.25, zorder=1, constraint=None):
        if zoom:
            hdu = fits.open(background_path)[0]
            wcs = WCS(background_path)
            box_size=5500
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
            
            r = SphericalCircle((trapezium.ra, trapezium.dec), circle,
                linestyle='dashed', linewidth=1.5, 
                edgecolor='w', facecolor='none', alpha=0.8, zorder=4, 
                transform=ax.get_transform('icrs'))
            ax.add_patch(r)
            
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
        fig = super().plot_3d(scale, show_figure=False, label=label)
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


    def plot_pm_rv(self, background_path=f'{user_path}/ONC/figures/skymap/hlsp_orion_hst_acs_colorimage_r_v1_drz.fits', save_path=None):
        """Plot proper motion and radial velocity

        Parameters
        ----------
        background_path : str, optional
            background path, by default f'{user_path}/ONC/figures/skymap/hlsp_orion_hst_acs_colorimage_r_v1_drz.fits'
        save_path : str, optional
            Save path, by default None
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
        
        # if constraint is None: constraint = np.ones(self.len, dtype=bool)
        im = ax.quiver(
            ra_wcs,
            dec_wcs,
            -self.pmRA.value,   # minius sign because ra increases from right to left
            self.pmDE.value,
            self.rv.value,
            cmap='coolwarm',
            width=0.006,
            scale=25
        )
        im.set_clim(vmin=19, vmax=35)
        
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
            labels=['MIST ($M_\odot$)', 'BHAC15 ($M_\odot$)', 'Feiden ($M_\odot$)', 'Palla ($M_\odot$)', 'Hillenbrand ($M_\odot$)'], 
            titles=['MIST', 'BHAC15', 'Feiden', 'Palla', 'Hillenbrand'], 
            limit=limit,
            save_path=save_path
        )
    
    
    def compare_chris(self, save_path=None):
        parenago_idx = list(self.data['HC2000']).index(546)
        idx_other = np.delete(np.array(range(self.len)), np.array(parenago_idx))

        teff_diff = self.teff - self.data['teff_chris']
        rv_diff = self.rv[idx_other] - self.data['rv_chris'][idx_other]
        print(f'Median absolute Teff difference: {np.nanmedian(abs(teff_diff)):.2f}.')
        print(f'Max. absolute Teff difference: {np.nanmax(abs(teff_diff)):.2f}.')
        print(f'Standard deviation of Teff difference: {np.nanstd(teff_diff):.2f}.')
        print(f'Median absolute RV difference: {np.nanmedian(abs(rv_diff)):.2f}.')
        print(f'Standard deviation RV difference: {np.nanstd(teff_diff):.2f}.')
        
        # compare veiling parameter of order 33    
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
        
        # compare teff
        ax1.plot([3000, 5000], [3000, 5000], linestyle='--', color='C3', label='Equal Line')
        ax1.errorbar(
            self.data['teff'][idx_other].value, 
            self.data['teff_chris'][idx_other].value, 
            xerr=self.data['e_teff'][idx_other].value, 
            yerr=self.data['e_teff_chris'][idx_other].value, 
            color=(.2, .2, .2, .8), fmt='o', alpha=0.5, markersize=3
        )
        ax1.errorbar(
            self.data['teff'][parenago_idx].value, 
            self.data['teff_chris'][parenago_idx].value, 
            xerr=self.data['e_teff'][parenago_idx].value, 
            yerr=self.data['e_teff_chris'][parenago_idx].value, 
            color='C0', label='Parenago 1837', 
            fmt='o', alpha=0.5, markersize=3
        )
        ax1.legend()
        ax1.set_xlabel(r'$T_\mathrm{eff, This\ Work}$ (K)', fontsize=12)
        ax1.set_ylabel(r'$T_\mathrm{eff, T22}$ (K)', fontsize=12)
        
        # compare rv
        ax2.plot([21, 34], [21, 34], linestyle='--', color='C3', label='Equal Line')
        ax2.errorbar(
            self.data['rv'][idx_other].value, 
            self.data['rv_chris'][idx_other].value, 
            xerr=self.data['e_rv'][idx_other].value, 
            yerr=self.data['e_rv_chris'][idx_other].value,  
            color=(.2, .2, .2, .8), fmt='o', alpha=0.5, markersize=3
        )
        ax2.errorbar(
            self.data['rv'][parenago_idx].value, 
            self.data['rv_chris'][parenago_idx].value, 
            xerr=self.data['e_rv'][parenago_idx].value, 
            yerr=self.data['e_rv_chris'][parenago_idx].value, 
            color='C0', label='Parenago 1837', 
            fmt='o', alpha=0.5, markersize=3
        )
        ax2.legend()
        ax2.set_xlabel(r'$\mathrm{RV}_\mathrm{This\ Work}$ $\left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$', fontsize=12)
        ax2.set_ylabel(r'$\mathrm{RV}_\mathrm{T22}$ $\left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$', fontsize=12)
        
        # compare veiling param    
        ax3.hist(self.data['veiling_param_O33_chris'], bins=20, range=(0, 1), histtype='step', color='C0', label='T22', lw=2)
        ax3.hist(self.data['veiling_param_O33_nirspao'], bins=20, range=(0, 1), histtype='step', color='C3', label='This work', lw=1.2)
        ax3.legend()
        ax3.set_xlabel('Veiling Param O33', fontsize=12)
        ax3.set_ylabel('Counts', fontsize=12)
        
        plt.subplots_adjust(wspace=0.3)
        if save_path:
            if save_path.endswith('png'):
                plt.savefig(save_path, bbox_inches='tight', transparent=True)
            else:
                plt.savefig(save_path, bbox_inches='tight')
        plt.show()


    def compare_teff_with_apogee(self, constraint=None, save_path=None):
        if constraint is None: constraint=np.ones(self.len, dtype=bool)
        diffs = (self.data['teff_apogee'] - self.data['teff_nirspao'])[constraint]
        valid_idx = ~np.isnan(diffs)
        diffs = diffs[valid_idx]
        weights = 1/(self.data['e_teff_nirspao'].value[constraint]**2 + self.data['e_teff_apogee'].value[constraint]**2)[valid_idx]
        mean_diff = np.average(diffs, weights=weights)
        max_diff = max(diffs)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.errorbar(self.data['teff_nirspao'].value, self.data['teff_apogee'].value, xerr=self.data['e_teff_nirspao'].value, yerr=self.data['e_teff_apogee'].value, fmt='o', color=(.2, .2, .2, .8), alpha=0.5, markersize=3)
        ranges = np.array([3600, 4800]) 
        ax.plot(ranges, ranges, linestyle='--', color='C3', label='Equal Line')
        ax.plot(ranges - mean_diff.value/2, ranges + mean_diff.value/2, linestyle=':', color='C0', label=f'Median Difference: {mean_diff:.1f}')
        ax.legend()
        ax.set_xlabel('NIRSPAO Teff (K)')
        ax.set_ylabel('APOGEE Teff (K)')
        if save_path:
            if save_path.endswith('.png'):
                plt.savefig(save_path, bbox_inches='tight', transparent=True)
            else:
                plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        
        return mean_diff, max_diff



#################################################
################### Functions ###################
#################################################

def fillna(column1, column2):
    result = column1.copy()
    result[np.isnan(result)] = column2[np.isnan(result)]
    return result



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
    trapezium_ra, trapezium_dec = wcs.wcs_world2pix(trapezium.ra.value, trapezium.dec.value, 0)
    tobin_ra    += ra_offset
    tobin_dec   += dec_offset
    apogee_ra   += ra_offset
    apogee_dec  += dec_offset
    trapezium_ra    += ra_offset
    trapezium_dec   += dec_offset
    
    ax.scatter(trapezium_ra, trapezium_dec, s=80, marker='*', edgecolor='royalblue', linewidth=1.5, facecolor='none', label='Center', zorder=4)
    ax.scatter(apogee_ra, apogee_dec, s=10, marker='s', edgecolor='C9', linewidths=1, facecolor='none', label='APOGEE', zorder=2)
    ax.scatter(tobin_ra, tobin_dec, s=10, marker='^', edgecolor='C1', linewidths=1, facecolor='none', label='Tobin et al. 2009', zorder=1)
    handles, labels = ax.get_legend_handles_labels()
    order = np.arange(len(labels))
    order [0] = 1
    order [1] = 0
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], framealpha=0.75, loc='upper right')
    plt.savefig(f'{user_path}/ONC/figures/skymap/Skymap Wide.pdf')
    plt.show()

    # Zoom-in figure
    parenago_idx = (orion.data['HC2000']==546).filled(False)
    V1337_idx = (orion.data['HC2000']==214).filled(False)
    V1279_idx = (orion.data['HC2000']==170).filled(False)
    brun_idx = (orion.data['HC2000']==172).filled(False)
    
    parenago_coord = SkyCoord(ra=orion.ra[parenago_idx], dec=orion.dec[parenago_idx])
    V1337_coord = SkyCoord(ra=orion.ra[V1337_idx], dec=orion.dec[V1337_idx])
    brun_coord = SkyCoord(ra=orion.ra[brun_idx], dec=orion.dec[brun_idx])
    apogee = apogee[(apogee['sep_to_trapezium'] <= 4*u.arcmin)]
    apogee_coord = SkyCoord(ra=apogee['RAJ2000'], dec=apogee['DEJ2000'])
    apogee = apogee[
        (apogee_coord.separation(parenago_coord) > 1.2*u.arcsec) & (apogee_coord.separation(V1337_coord) > 1.2*u.arcsec) & (apogee_coord.separation(brun_coord) > 1.2*u.arcsec)
    ]
    fig, ax, wcs = orion.plot_skymap(background_path=background_path, show_figure=False, label='NIRSPAO', zoom=True, constraint=~orion.data['HC2000'].mask & ~parenago_idx & ~V1337_idx & ~V1279_idx & ~brun_idx, zorder=3)
    apogee_ra, apogee_dec = wcs.wcs_world2pix(apogee['RAJ2000'].value, apogee['DEJ2000'].value, 0)
    parenago_ra, parenago_dec = wcs.wcs_world2pix(orion.ra[parenago_idx].value, orion.dec[parenago_idx].value, 0)
    V1337_ra, V1337_dec = wcs.wcs_world2pix(orion.ra[V1337_idx].value, orion.dec[V1337_idx].value, 0)
    V1279_ra, V1279_dec = wcs.wcs_world2pix(orion.ra[V1279_idx].value, orion.dec[V1279_idx].value, 0)
    brun_ra, brun_dec = wcs.wcs_world2pix(orion.ra[brun_idx].value, orion.dec[brun_idx].value, 0)
    trapezium_ra, trapezium_dec = wcs.wcs_world2pix(trapezium.ra.value, trapezium.dec.value, 0)
    apogee_ra       += ra_offset
    apogee_dec      += dec_offset
    parenago_ra     += ra_offset
    parenago_dec    += dec_offset
    V1337_ra        += ra_offset
    V1337_dec       += dec_offset
    V1279_ra        += ra_offset
    V1279_dec       += dec_offset
    brun_ra         += ra_offset
    brun_dec        += dec_offset
    trapezium_ra    += ra_offset
    trapezium_dec   += dec_offset
    ax.scatter(trapezium_ra, trapezium_dec, s=100, marker='*', edgecolor='royalblue', linewidth=1.5, facecolor='none', label='Center', zorder=4)
    ax.scatter(apogee_ra, apogee_dec, s=15, marker='s', edgecolor='C9', linewidths=1.25, facecolor='none', label='APOGEE', zorder=2)
    ax.scatter(parenago_ra, parenago_dec, s=50, marker='X', edgecolor='lightgreen', linewidth=1.5, facecolor='none', label='Parenago 1837', zorder=4)
    ax.scatter(V1337_ra, V1337_dec, s=50, marker='P', edgecolor='gold', linewidth=1.5, facecolor='none', label='V* V1337 Ori', zorder=4)
    ax.scatter(V1279_ra, V1279_dec, s=30, marker='D', edgecolor='orange', linewidth=1.5, facecolor='none', label='V* V1279 Ori', zorder=4)
    ax.scatter(brun_ra, brun_dec, s=50, marker='H', edgecolor='red', linewidth=1.5, facecolor='none', label='Brun 590', zorder=4)
    handles, labels = ax.get_legend_handles_labels()
    order = np.arange(len(labels))
    order [0] = 1
    order [1] = 0
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], framealpha=0.75, loc='upper right')
    plt.savefig(f'{user_path}/ONC/figures/skymap/Skymap Zoom.pdf')
    plt.show()




def corner_plot(data, labels, limit, titles=None, save_path=None):
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
                if titles is not None:
                    axs[i, j].set_title(titles[i], fontsize=fontsize)
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



def vdisp_all(sources, save_path, MCMC=True):
    '''Fit velocity dispersion for all sources, kim, and gaia respectively.

    Parameters
    ----------
    sources : astropy QTable
        Table containing pmRA, e_pmRA, pmDE, e_pmDE, rv, e_rv
    save_path : str
        Folder under which to save.
    MCMC : bool
        Run MCMC or read from existing fitting results (default True).
    constraint : array-like
        Constraint
    
    Returns
    -------
    vdisps_all: dict
        velocity dispersion for all sources.
        vdisps_all[key] = [value, error].
        keys: mu_RA, mu_DE, mu_rv, sigma_RA, sigma_DE, sigma_rv, rho_RA, rho_DE, rho_rv.
    '''
    # all velocity dispersions
    print('Fitting for all velocity dispersion...')
    sources_new = copy.deepcopy(sources)
    
    vdisps_all = fit_vdisp(
        sources_new,
        save_path=f'{save_path}/all/', 
        MCMC=MCMC
    )
    
    vdisp_1d = [0, 0]
    
    vdisp_1d[0] = ((vdisps_all.sigma_RA[0]**2 + vdisps_all.sigma_DE[0]**2 + vdisps_all.sigma_rv[0]**2)/3)**(1/2)
    vdisp_1d[1] = np.sqrt(1 / (3*vdisp_1d[0])**2 * ((vdisps_all.sigma_RA[0] * vdisps_all.sigma_RA[1])**2 + (vdisps_all.sigma_DE[0] * vdisps_all.sigma_DE[1])**2 + (vdisps_all.sigma_rv[0] * vdisps_all.sigma_rv[1])**2))
    
    with open(f'{save_path}/all/mcmc_params.txt', 'r') as file:
        raw = file.readlines()
    if not any([line.startswith('σ_1D:') for line in raw]):
        raw.insert(6, f'σ_1D:\t{vdisp_1d[0]}, {vdisp_1d[1]}\n')
        with open(f'{save_path}/all/mcmc_params.txt', 'w') as file:
            file.writelines(raw)
    
    # kim velocity dispersions
    print('Fitting for Kim velocity dispersion...')
    fit_vdisp(
        sources_new[~sources_new['ID_kim'].mask],
        save_path=f'{save_path}/kim', 
        MCMC=MCMC
    )

    # gaia velocity dispersions
    print('Fitting for Gaia velocity dispersion...')
    fit_vdisp(
        sources_new[~sources_new['Gaia DR3'].mask],
        save_path=f'{save_path}/gaia',
        MCMC=MCMC
    )
    
    return vdisps_all*u.km/u.s



def vdisp_vs_sep_binned(sources, separations, save_path, MCMC=True):
    '''Velocity dispersion within a bin vs separation from Trapezium in arcmin.
    
    Parameters
    ------------------------
    sources : astropy QTable
    separations : astropy quantity
        Separation borders, e.g., np.array([0, 1, 2, 3, 4]) * u.arcmin
    save_path: str
        Save path
    MCMC: boolean
        Run MCMC or load existing fitting results.
    
    
    Returns
    ------------------------
    vdisps: list of velocity dispersion results of each bin. length = len(separations) - 1
        vdisps[i] = pd.DataFrame({'mu_RA': [value, error], 'mu_DE': [value, error], 'mu_rv': [value, error], ...})
        keys: mu_RA, mu_DE, mu_rv, sigma_RA, sigma_DE, sigma_rv, rho_RA, rho_DE, rho_rv.
    '''
    sources = sources[~(np.isnan(sources['vRA']) | np.isnan(sources['vDE']) | np.isnan(sources['rv']))]
    
    separations_binned = (separations[:-1] + separations[1:])/2
    
    # binned velocity dispersions
    vdisps = []
    sources_in_bins = []
    for i, min_sep, max_sep in zip(range(len(separations_binned)), separations[:-1], separations[1:]):
        if MCMC:
            print()
            print(f'Start fitting for bin {i}...')
        
        sources_in_bins.append(sum((sources['sep_to_trapezium'] > min_sep) & (sources['sep_to_trapezium'] <= max_sep)))
        vdisps.append(fit_vdisp(
            sources[(sources['sep_to_trapezium'] > min_sep) & (sources['sep_to_trapezium'] <= max_sep)],
            save_path=f'{save_path}/bin {i}',
            MCMC=MCMC
        ))
    
    return vdisps



def vdisp_vs_sep_equally_spaced(sources, nbins:int, save_path:str, MCMC:bool) -> Tuple[u.Quantity, dict]:
    """Velocity dispersion vs separations, equally spaced.

    Parameters
    ----------
    sources : astropy quantity
        Sources.
    nbins : int
        Number of bins.
    save_path : str
        Save path.
    MCMC : bool
        Run MCMC or not.

    Returns
    -------
    separation_borders : astropy.Quantity
        Separation borders of each bin. e.g., np.array([0, 1, 2, 3, 4]) * u.arcmin
    vdisps : list
        Velocity dispersion fitting results of length nbins.
        Each element is a dictionary with keys ['mu_RA', 'mu_DE', 'mu_rv', 'sigma_RA', 'sigma_DE', 'sigma_rv', 'rho_RA', 'rho_DE', 'rho_rv'].
    """
    # filter nans.
    sources = sources[~(np.isnan(sources['vRA']) | np.isnan(sources['vDE']) | np.isnan(sources['rv']))]

    separation_borders = np.linspace(0, 4, nbins+1)*u.arcmin
    return separation_borders, vdisp_vs_sep_binned(sources, separation_borders, f'{save_path}/equally spaced/{nbins}-binned/', MCMC=MCMC)



def vdisp_vs_sep_equally_grouped(sources:pd.DataFrame, ngroups:int, save_path:str, MCMC:bool):
    """Velocity dispersion vs separations, equally grouped.
    Prioritize higher numbers at closer distance to trapezium. e.g., 10 is divided into 4+3+3.
    
    Parameters
    ----------
    sources : pd.DataFrame
        Sources.
    ngroups : int
        Number of groups.
    save_path : str
        Save path.
    MCMC : bool
        Run MCMC or not.

    Returns
    -------
    separation_borders : astropy.Quantity
        Separation borders of each bin. e.g., np.array([0, 1, 2, 3, 4]) * u.arcmin
    vdisps : list
        List of velocity dispersion fitting results of length nbins.
        Each element is a dictionary with keys ['mu_RA', 'mu_DE', 'mu_rv', 'sigma_RA', 'sigma_DE', 'sigma_rv', 'rho_RA', 'rho_DE', 'rho_rv'].
    """
    # filter nans.
    sources = sources[~(np.isnan(sources['vRA']) | np.isnan(sources['vDE']) | np.isnan(sources['rv']))]

    sources_in_bins = [len(sources) // ngroups + (1 if x < len(sources) % ngroups else 0) for x in range (ngroups)]
    division_idx = np.cumsum(sources_in_bins)[:-1] - 1
    separation_sorted = np.sort(sources['sep_to_trapezium'])
    separation_borders = np.array([0, *(separation_sorted[division_idx] + separation_sorted[division_idx+1]).value/2, 4]) * separation_sorted.unit
    return separation_borders, vdisp_vs_sep_binned(sources, separation_borders, f'{save_path}/equally grouped/{ngroups}-binned', MCMC=MCMC)



def vdisp_vs_sep(sources, nbins, ngroups, save_path, MCMC):
    """Velocity dispersion vs separation.
    Function call graph:
    - vdisp_vs_sep
        - vdisp_vs_sep_equally_spaced, vdisp_vs_sep_equally_grouped
            - vdisp_vs_sep_binned
                - fit_velocity_dispersion
    
    
    Parameters
    ----------
    sources : astropy Table
        sources
    nbins : int
        number of bins for equally spaced case.
    ngroups : int
        number of groups for equally grouped case.
    save_path : str
        save path.
    MCMC : bool
        run mcmc or not.
    """
    # filter nans.
    sources = sources[~(np.isnan(sources['vRA']) | np.isnan(sources['vDE']) | np.isnan(sources['rv']))]
    
    # virial equilibrium model
    model_separations_pc = np.logspace(-4, np.log10(4/60*np.pi/180*389), 100)   # separations in pc
    model_separations_arcmin = model_separations_pc / 389 * 180/np.pi * 60      # separations in arcmin
    sigma = np.sqrt(1/37*(70/0.8 * model_separations_pc**0.8 + 22/3 * model_separations_pc**3) / model_separations_pc)
    sigma_hi = np.sqrt(1/37*(70/0.8 * model_separations_pc**0.8 + 22/3 * model_separations_pc**3) * 1.3 / model_separations_pc)
    sigma_lo = np.sqrt(1/37*(70/0.8 * model_separations_pc**0.8 + 22/3 * model_separations_pc**3) * 0.7 / model_separations_pc)
    
    # Left: equally spaced    
    if MCMC:
        print(f'{nbins}-binned equally spaced velocity dispersion vs separation fitting...')
    
    separation_borders, vdisps = vdisp_vs_sep_equally_spaced(sources, nbins, save_path, MCMC)
    
    if MCMC:
        print(f'{nbins}-binned equally spaced velocity dispersion vs separation fitting finished!\n')
    
    
    # average separation within each bin.
    separation_sources = np.array([np.mean(sources['sep_to_trapezium'][(sources['sep_to_trapezium'] > min_sep) & (sources['sep_to_trapezium'] <= max_sep)].to(u.arcmin).value) for min_sep, max_sep in zip(separation_borders[:-1], separation_borders[1:])])
    
    sources_in_bins = [sum((sources['sep_to_trapezium'] > min_sep) & (sources['sep_to_trapezium'] <= max_sep)) for min_sep, max_sep in zip(separation_borders[:-1], separation_borders[1:])]
    
    # sigma_xx: 2*N array. sigma_xx[0] = value, sigma_xx[1] = error.
    sigma_RA = np.array([vdisp.sigma_RA for vdisp in vdisps]).transpose()
    sigma_DE = np.array([vdisp.sigma_DE for vdisp in vdisps]).transpose()
    sigma_rv = np.array([vdisp.sigma_rv for vdisp in vdisps]).transpose()

    sigma_pm = np.empty_like(sigma_RA)
    sigma_pm[0] = np.sqrt((sigma_RA[0]**2 + sigma_DE[0]**2)/2)
    sigma_pm[1] = np.sqrt(1/4*((sigma_RA[0]/sigma_pm[0]*sigma_RA[1])**2 + (sigma_DE[0]/sigma_pm[0]*sigma_DE[1])**2))

    sigma_1d = np.empty_like(sigma_RA)
    sigma_1d[0] = np.sqrt((sigma_RA[0]**2 + sigma_DE[0]**2 + sigma_rv[0]**2)/3)
    sigma_1d[1] = np.sqrt(1/9*((sigma_RA[0]/sigma_1d[0]*sigma_RA[1])**2 + (sigma_DE[0]/sigma_1d[0]*sigma_DE[1])**2 + (sigma_rv[0]/sigma_1d[0]*sigma_rv[1])**2))    
    
    fig, axs = plt.subplots(3, 2, figsize=(8, 9), dpi=300, sharex='col', sharey='row')
    
    for ax, direction, sigma_xx in zip(axs[:, 0], ['1d', 'pm', 'rv'], [sigma_1d, sigma_pm, sigma_rv]):
        # arcmin axis
        ax.set_xlim((0, 4))
        ax.set_ylim((0.5, 5.5))
        ax.tick_params(axis='both', labelsize=12)
        solid_line, = ax.plot(model_separations_arcmin, sigma, color='k')
        dotted_line, = ax.plot(model_separations_arcmin, sigma_lo, color='k', linestyle='dotted')
        ax.plot(model_separations_arcmin, sigma_hi, color='k', linestyle='dotted')
        ax.fill_between(model_separations_arcmin, y1=sigma_lo, y2=sigma_hi, edgecolor='none', facecolor='C7', alpha=0.4)
        if direction=='1d':
            ax.set_ylabel(r'$\sigma_{\mathrm{1D, rms}}$ $\left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$', fontsize=15)
        elif direction=='pm':
            ax.set_ylabel(r'$\sigma_{\mathrm{pm, rms}}$ $\left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$', fontsize=15)
        elif direction=='rv':
            ax.set_xlabel('Separation from Trapezium (arcmin)', fontsize=15, labelpad=10)
            ax.set_ylabel(r'$\sigma_{\mathrm{RV}}$ $\left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$', fontsize=15)
        
        errorbar = ax.errorbar(separation_sources, sigma_xx[0], yerr=sigma_xx[1], color='C3', fmt='o-', markersize=5, capsize=5)
        
        for i in range(len(sources_in_bins)):
            ax.annotate(f'{sources_in_bins[i]}', (separation_sources[i], sigma_xx[0, i] + sigma_xx[1, i] + 0.15), fontsize=12, horizontalalignment='center')
        ax.fill_between(separation_sources, y1=sigma_xx[0]-sigma_xx[1], y2=sigma_xx[0]+sigma_xx[1], edgecolor='none', facecolor='C3', alpha=0.4)
        
        # pc axis
        ax2 = ax.twiny()
        ax2.tick_params(axis='both', labelsize=12)
        ax2.set_xlim((0, 4/60 * np.pi/180 * 389))
        if direction=='1d':
            ax2.set_xlabel('Separation from Trapezium (pc)', fontsize=15, labelpad=10)
            ax2.set_title('Equally Spaced\n', fontsize=15)
        else:
            ax2.tick_params(
                axis='x',
                top=False,
                labeltop=False
            )
    
    
    # Right: Equally Grouped
    
    if MCMC:
        print(f'{ngroups}-binned equally grouped velocity dispersion vs separation fitting...')

    separation_borders, vdisps = vdisp_vs_sep_equally_grouped(sources, ngroups, save_path, MCMC)    

    if MCMC:
        print(f'{ngroups}-binned equally grouped velocity dispersion vs separation fitting finished!')

    # average separation within each bin.
    separation_sources = np.array([np.mean(sources['sep_to_trapezium'][(sources['sep_to_trapezium'] > min_sep) & (sources['sep_to_trapezium'] <= max_sep)].to(u.arcmin).value) for min_sep, max_sep in zip(separation_borders[:-1], separation_borders[1:])])

    sources_in_bins = [len(sources) // ngroups + (1 if x < len(sources) % ngroups else 0) for x in range (ngroups)]
    
    # sigma_xx: 2*N array. sigma_xx[0] = value, sigma_xx[1] = error.
    sigma_RA = np.array([vdisp.sigma_RA for vdisp in vdisps]).transpose()
    sigma_DE = np.array([vdisp.sigma_DE for vdisp in vdisps]).transpose()
    sigma_rv = np.array([vdisp.sigma_rv for vdisp in vdisps]).transpose()

    sigma_pm = np.empty_like(sigma_RA)
    sigma_pm[0] = np.sqrt((sigma_RA[0]**2 + sigma_DE[0]**2)/2)
    sigma_pm[1] = np.sqrt(1/4*((sigma_RA[0]/sigma_pm[0]*sigma_RA[1])**2 + (sigma_DE[0]/sigma_pm[0]*sigma_DE[1])**2))

    sigma_1d = np.empty_like(sigma_RA)
    sigma_1d[0] = np.sqrt((sigma_RA[0]**2 + sigma_DE[0]**2 + sigma_rv[0]**2)/3)
    sigma_1d[1] = np.sqrt(1/9*((sigma_RA[0]/sigma_1d[0]*sigma_RA[1])**2 + (sigma_DE[0]/sigma_1d[0]*sigma_DE[1])**2 + (sigma_rv[0]/sigma_1d[0]*sigma_rv[1])**2))    

    for ax, direction, sigma_xx in zip(axs[:, 1], ['1d', 'pm', 'rv'], [sigma_1d, sigma_pm, sigma_rv]):
        # arcmin axis
        ax.set_xlim((0, 4))
        ax.set_ylim((0.5, 5.5))
        ax.tick_params(axis='both', labelsize=12)
        solid_line, = ax.plot(model_separations_arcmin, sigma, color='k')
        dotted_line, = ax.plot(model_separations_arcmin, sigma_hi, color='k', linestyle='dotted')
        ax.plot(model_separations_arcmin, sigma_lo, color='k', linestyle='dotted')
        gray_fill = ax.fill_between(model_separations_arcmin, y1=sigma_lo, y2=sigma_hi, edgecolor='none', facecolor='C7', alpha=0.4)
        if direction=='rv':
            ax.set_xlabel('Separation from Trapezium (arcmin)', fontsize=15, labelpad=10)
        
        errorbar = ax.errorbar(separation_sources, sigma_xx[0], yerr=sigma_xx[1], color='C3', fmt='o-', markersize=5, capsize=5)
        
        for i in range(len(sources_in_bins)):
            ax.annotate(f'{sources_in_bins[i]}', (separation_sources[i], sigma_xx[0, i] + sigma_xx[1, i] + 0.15), fontsize=12, horizontalalignment='center')
        red_fill = ax.fill_between(separation_sources, y1=sigma_xx[0]-sigma_xx[1], y2=sigma_xx[0]+sigma_xx[1], edgecolor='none', facecolor='C3', alpha=0.4)
        
        # arcmin axis
        ax2 = ax.twiny()
        ax2.set_xlim((0, 4/60 * np.pi/180 * 389))
        ax2.tick_params(axis='both', labelsize=12)
        if direction=='1d':
            ax2.set_xlabel('Separation from Trapezium (pc)', fontsize=15, labelpad=10)
            ax2.set_title('Equally Grouped\n', fontsize=15)
        else:
            ax2.tick_params(
                axis='x',
                top=False,
                labeltop=False
            )
    
    axs[0, 1].legend(handles=[(errorbar, red_fill), solid_line, dotted_line], labels=['Measured Velocity Dispersion', 'Virial Equilibrium Model', '30% Total Mass Error'], fontsize=12)
    
    axs[-1, 0].set_xticks([0, 1, 2, 3])
    axs[-1, 1].set_xticks([0, 1, 2, 3, 4])
    fig.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(f'{user_path}/ONC/figures/vdisp vs sep.pdf', bbox_inches='tight')
    plt.savefig(f'{user_path}/ONC/figures/vdisp vs sep.png', bbox_inches='tight', transparent=True)
    plt.show()



#################################################
################ Vdisps vs Masses ###############
#################################################

def vdisp_vs_mass_binned(sources, model_name, mass_borders, save_path, MCMC=True):
    '''Velocity dispersion within a bin vs mass.
    
    Parameters
    ------------------------
    sources: pandas.DataFrame.
    mass_borders: np.array.
        Mass borders, e.g., np.array([0, 1, 2, 3, 4])*u.solMass.
    model_name: str.
        evolutionary model name.
    save_path: str.
        save path.
    MCMC: boolean.
        run MCMC or load existing fitting results.
    
    
    Returns
    ------------------------
    vdisps: list of velocity dispersion results of each bin. length = len(masses) - 1
        vdisps[i] = pd.DataFrame({'mu_RA': [value, error], 'mu_DE': [value, error], 'mu_rv': [value, error], ...})
        keys: mu_RA, mu_DE, mu_rv, sigma_RA, sigma_DE, sigma_rv, rho_RA, rho_DE, rho_rv.
    '''
    sources = sources[~(np.isnan(sources['vRA']) | np.isnan(sources['vDE']) | np.isnan(sources['rv']))]
    
    masses_binned = (mass_borders[:-1] + mass_borders[1:])/2
    
    # binned velocity dispersions
    vdisps = []
    sources_in_bins = []
    for i, min_mass, max_mass in zip(range(len(masses_binned)), mass_borders[:-1], mass_borders[1:]):
        if MCMC:
            print()
            print(f'Start fitting for bin {i}...')
        
        sources_in_bins.append(sum((sources[f'mass_{model_name}'] > min_mass) & (sources[f'mass_{model_name}'] <= max_mass)))
        vdisps.append(fit_vdisp(
            sources[(sources[f'mass_{model_name}'] > min_mass) & (sources[f'mass_{model_name}'] <= max_mass)],
            save_path=f'{save_path}/bin {i}',
            MCMC=MCMC
        ))
    
    return vdisps



def vdisp_vs_mass_equally_grouped(sources:QTable, model_name:str, ngroups:int, save_path:str, MCMC:bool):
    """Velocity dispersion vs masses, equally grouped.
    Prioritize higher numbers at closer distance to trapezium. e.g., 10 is divided into 4+3+3.
    
    Parameters
    ----------
    sources : astropy QTable
        Sources
    model_name: str.
        Model name
    ngroups : int
        Number of groups.
    save_path : str
        Save path.
    MCMC : bool
        Run MCMC or not.

    Returns
    -------
    masses : np.array
        Dividing masses. e.g., [0, 1, 2, 3, 4]*u.solMass.
    vdisps : list
        List of velocity dispersion fitting results of length nbins.
        Each element is a dictionary with keys ['mu_RA', 'mu_DE', 'mu_rv', 'sigma_RA', 'sigma_DE', 'sigma_rv', 'rho_RA', 'rho_DE', 'rho_rv'].
    """
    # filter nans.
    sources = sources[~(np.isnan(sources['vRA']) | np.isnan(sources['vDE']) | np.isnan(sources['rv']))]

    sources_in_bins = [len(sources) // ngroups + (1 if x < len(sources) % ngroups else 0) for x in range (ngroups)]
    division_idx = np.cumsum(sources_in_bins)[:-1] - 1
    mass_sorted = np.sort(sources[f'mass_{model_name}'])
    mass_borders = np.array([0, *(mass_sorted[division_idx] + mass_sorted[division_idx+1]).value/2, 4])*mass_sorted.unit
    return mass_borders, vdisp_vs_mass_binned(sources, model_name, mass_borders, f'{save_path}/equally grouped/{ngroups}-binned/', MCMC=MCMC)



def vdisp_vs_mass(sources, model_name, ngroups, save_path, MCMC):
    """Velocity dispersion vs mass.
    Function call graph:
    - vdisp_vs_mass
        - vdisp_vs_mass_equally_grouped
            - vdisp_vs_mass_binned
                - fit_vdisp
    
    
    Parameters
    ----------
    sources : astropy QTable
        sources
    model_name: str
        Model name.
    ngroups : int
        Number of groups for equally grouped case.
    save_path : str
        Save path.
    MCMC : bool
        Run mcmc or not.
    """
    # filter nans.
    sources = sources[~(np.isnan(sources['vRA']) | np.isnan(sources['vDE']) | np.isnan(sources['rv']))]
    
        
    # Equally Grouped
    
    if MCMC:
        print(f'{ngroups}-binned equally grouped velocity dispersion vs mass fitting...')

    mass_borders, vdisps = vdisp_vs_mass_equally_grouped(sources, model_name, ngroups, save_path, MCMC)    

    if MCMC:
        print(f'{ngroups}-binned equally grouped velocity dispersion vs mass fitting finished!')

    # average mass within each bin
    mass_sources = []
    for min_mass, max_mass in zip(mass_borders[:-1], mass_borders[1:]):
        bin_idx = (sources[f'mass_{model_name}'] > min_mass) & (sources[f'mass_{model_name}'] <= max_mass)
        mass_sources.append(np.average(sources[f'mass_{model_name}'][bin_idx].value, weights=1/sources[f'e_mass_{model_name}'][bin_idx].value**2))
    mass_sources = np.array(mass_sources)
    
    sources_in_bins = [len(sources) // ngroups + (1 if x < len(sources) % ngroups else 0) for x in range (ngroups)]
    
    # sigma_xx: 2*N array. sigma_xx[0] = value, sigma_xx[1] = error.
    sigma_RA = np.array([vdisp.sigma_RA for vdisp in vdisps]).transpose()
    sigma_DE = np.array([vdisp.sigma_DE for vdisp in vdisps]).transpose()
    sigma_rv = np.array([vdisp.sigma_rv for vdisp in vdisps]).transpose()
    
    sigma_pm = np.empty_like(sigma_RA)
    sigma_pm[0] = np.sqrt((sigma_RA[0]**2 + sigma_DE[0]**2)/2)
    sigma_pm[1] = np.sqrt(1/4*((sigma_RA[0]/sigma_pm[0]*sigma_RA[1])**2 + (sigma_DE[0]/sigma_pm[0]*sigma_DE[1])**2))

    sigma_1d = np.empty_like(sigma_RA)
    sigma_1d[0] = np.sqrt((sigma_RA[0]**2 + sigma_DE[0]**2 + sigma_rv[0]**2)/3)
    sigma_1d[1] = np.sqrt(1/9*((sigma_RA[0]/sigma_1d[0]*sigma_RA[1])**2 + (sigma_DE[0]/sigma_1d[0]*sigma_DE[1])**2 + (sigma_rv[0]/sigma_1d[0]*sigma_rv[1])**2))    
    
    
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, direction, sigma_xx in zip(axs, ['1d', 'pm', 'rv'], [sigma_1d, sigma_pm, sigma_rv]):
        if direction=='1d':
            ax.set_title('$\sigma_{\mathrm{1D, rms}}$', fontsize=15)
            ax.set_ylabel(r'$\sigma$ $\left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$', fontsize=15)
        elif direction=='pm':
            ax.set_title('$\sigma_{\mathrm{pm, rms}}$', fontsize=15)
            ax.set_xlabel(f'{model_name} Mass ($M_\odot$)', fontsize=15, labelpad=10)
        elif direction=='rv':
            ax.set_title('$\sigma_{\mathrm{RV}}$', fontsize=15)

        errorbar = ax.errorbar(mass_sources, sigma_xx[0], yerr=sigma_xx[1], color='C3', fmt='o-', markersize=5, capsize=5)
        
        for i in range(len(sources_in_bins)):
            ax.annotate(f'{sources_in_bins[i]}', (mass_sources[i], sigma_xx[0, i] + sigma_xx[1, i] + 0.15), fontsize=12, horizontalalignment='center')
        fill = ax.fill_between(mass_sources, y1=sigma_xx[0]-sigma_xx[1], y2=sigma_xx[0]+sigma_xx[1], edgecolor='none', facecolor='C3', alpha=0.4)

        ax.set_ylim((1.3, 5.2))
        ax.loglog()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter()) # set to regular format
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1])
        ax.tick_params(axis='both', which='major', labelsize=12)

    axs[0].legend(handles=[(errorbar, fill)], labels=['Measured Velocity Dispersion'], fontsize=12, loc='upper left')
    fig.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(f'{user_path}/ONC/figures/vdisp vs mass.pdf', bbox_inches='tight')
    plt.savefig(f'{user_path}/ONC/figures/vdisp vs mass.png', bbox_inches='tight', transparent=True)
    plt.show()



def merge_multiple_mass(sources:QTable):
    sources = copy.deepcopy(sources)
    unique_ids, counts = np.unique(sources['HC2000'], return_counts=True)
    multiple_ids = unique_ids[(counts > 1) & (~unique_ids.mask)]
    for multiple_id in multiple_ids:
        multiple_idx = np.where(sources['HC2000']==multiple_id)[0]
        for model_name in ['MIST', 'BHAC15', 'Palla', 'Feiden']:
            # if trapezium stars: replace latest one with mass literature
            # if all(~sources['theta_orionis'][sources['HC2000']==multiple_id].mask):
            #         sources[f'mass_{model_name}'][multiple_idx[-1]] = sources['mass_literature'][multiple_idx[-1]]
            #         sources[f'e_mass_{model_name}'][multiple_idx[-1]] = sources['e_mass_literature'][multiple_idx[-1]]
            # else:
            sources[f'mass_{model_name}'][multiple_idx[-1]] = sum(sources[f'mass_{model_name}'][multiple_idx])
            sources[f'e_mass_{model_name}'][multiple_idx[-1]] = (sum(sources[f'e_mass_{model_name}'][multiple_idx]**2))**0.5
        
        sources.remove_rows(multiple_idx[:-1])
    return sources
            
        
    
    
def mass_segregation_ratio(sources:QTable, model_name:str, save_path:str, Nmst_min=5, Nmst_max=40, step=1, use_literature_trapezium_mass=True):
    '''Mass segregation ratio. See https://doi.org/10.1111/j.1365-2966.2009.14508.x.
    
    Parameters:
        sources : astropy QTable
            Sources
        model : str
            Model name
        save_path : str
            Path to save the figure
        Nmst_min : int
            Minimum number of sources selected to construct the minimum spanning tree (mst).
        Nmst_max : int
            Maximum number of sources selected to construct the minimum spanning tree (mst).
        step : ste
    
    Returns:
        lambda_msr: N-by-2 array of mass segregation ratio in the form of [[value, error], ...].
    '''
    
    # Merge binaries.
    sources = merge_multiple_mass(sources)
    if use_literature_trapezium_mass:
        idx_mass_literature = ~np.isnan(sources['mass_literature'])
        sources[f'mass_{model_name}'][idx_mass_literature] = sources['mass_literature'][idx_mass_literature]
        sources[f'e_mass_{model_name}'][idx_mass_literature] = sources['e_mass_literature'][idx_mass_literature]
    sources.remove_rows(np.isnan(sources[f'mass_{model_name}']))
    # binary_hc2000 = sources['HC2000'][(~sources['m_HC2000'].mask) & sources['APOGEE'].mask]
    # binary_hc2000_unique = [_.split('_')[0] for _ in binary_hc2000 if _.endswith('_A')]
    # # binary_idx_pairs = [[a_idx, b_idx], [a_idx, b_idx], ...]
    # binary_idx_pairs = [[binary_hc2000.loc[binary_hc2000==f'{_}_A'].index[0], binary_hc2000.loc[binary_hc2000==f'{_}_B'].index[0]] for _ in binary_hc2000_unique]
    
    # for binary_idx_pair in binary_idx_pairs:
    #     nan_flag = sources.loc[binary_idx_pair, [f'mass_{model_name}', f'e_mass_{model_name}']].isna()
    
    #     # if any value is valid, update the first place with m=m1+m2, m_e = sqrt(m1_e**2 + m2_e**2) (valid values only). Else (all values are nan), do nothing.
    #     if any(~nan_flag[f'mass_{model_name}']):
    #         sources.loc[binary_idx_pair[0], f'mass_{model_name}'] = sum(sources.loc[binary_idx_pair, f'mass_{model_name}'][~nan_flag[f'mass_{model_name}']])
        
    #     if any(~nan_flag[f'e_mass_{model_name}']):
    #         sources.loc[binary_idx_pair[0], f'e_mass_{model_name}'] = sum(sources.loc[binary_idx_pair, f'e_mass_{model_name}'][~nan_flag[f'mass_{model_name}']].pow(2))**0.5
            
    #     # update names to remove '_A', '_B' suffix.
    #     sources.loc[binary_idx_pair[0], 'HC2000'] = sources.loc[binary_idx_pair[0], 'HC2000'].split('_')[0]
    #     # remove values in the second place.
    #     sources = sources.drop(binary_idx_pair[1])

    # sources = sources.reset_index(drop=True)

    # Construct separation matrix
    sources_coord = SkyCoord(ra=sources['RAJ2000'], dec=sources['DEJ2000'])
    sep_matrix = np.zeros((len(sources), len(sources)))

    for i in range(len(sources)):
        sep_matrix[i, :] = sources_coord[i].separation(sources_coord).arcsec
        sep_matrix[i, 0:i] = 0


    # Construct MST for the Nmst most massive sources.
    # lambda: [[value, error], [value, error], ...]
    lambda_msr = np.empty((len(np.arange(Nmst_min, Nmst_max, step)), 2))
    for i, Nmst in enumerate(np.arange(Nmst_min, Nmst_max, step)):
        massive_idx = np.sort(np.argpartition(sources[f'mass_{model_name}'], -Nmst)[-Nmst:])  # sorted idx of most massive Nmst sources
        massive_sep_matrix = sep_matrix[massive_idx][:, massive_idx]
        massive_mst = minimum_spanning_tree(csr_matrix(massive_sep_matrix)).toarray()
        l_massive = massive_mst.sum()

        # construct MST for the Nmst random sources.
        if Nmst>=5 and Nmst<=10:
            repetition = 1000
        else:
            repetition = 50
        l_norm = np.empty(repetition)
        for j in range(repetition):
            random_idx = np.random.choice(len(sources), Nmst)
            random_sep_matrix = sep_matrix[random_idx][:, random_idx]
            random_mst = minimum_spanning_tree(csr_matrix(random_sep_matrix)).toarray()
            l_norm[j] = random_mst.sum()

        lambda_msr[i, :] = np.array([l_norm.mean()/l_massive, l_norm.std()/l_massive])

    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=300)
    h1, = ax.plot(np.arange(Nmst_min, Nmst_max, step), lambda_msr[:, 0], marker='o', markersize=5, label=r'$\Lambda_{MSR}$')
    f1 = ax.fill_between(np.arange(Nmst_min, Nmst_max, step), lambda_msr[:, 0] - lambda_msr[:, 1], lambda_msr[:, 0] + lambda_msr[:, 1], edgecolor='none', facecolor='C0', alpha=0.4, label='Uncertainty')
    h2 = ax.hlines(1, Nmst_min, Nmst_max, colors='C3', linestyles='--', label='No Segregation')
    # automatic log scale
    if max(lambda_msr[:, 0]) / min(lambda_msr[:, 0]) > 3:
        ax.set_yscale('log')
    
    ax.set_xlabel(r'$N_{MST}$', fontsize=15)
    ax.set_ylabel(r'$\Lambda_{MSR}$', fontsize=15)
    if max(lambda_msr[:, 0]) > 1:
        ax.legend([(h1, f1), h2], [r'$\Lambda_{MSR}$', 'No Segregation'], fontsize=12)
    else:
        ax.legend([(h1, f1), h2], [r'$\Lambda_{MSR}$', 'No Segregation'], fontsize=12, loc='lower right')
    fig.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
    return lambda_msr


def mean_mass_binned(sources, separations, model_name):
    """Mean mass of sources within a bin vs separation from the Trapezium.

    Parameters
    ----------
    sources : pandas.DataFrame
    separations : astropy.Quantity
        separations.
    model_name : str
        model name.
    save_path : str
        save path.
    
    Returns
    -------
    mass_mean: ndarray
        mean mass within each bin.
    mass_std: ndarray
        standard deviation within each bin.
    """
    # sources = sources.sort_values('sep_to_trapezium').reset_index(drop=True)
    sources = sources.copy()
    sources.sort('sep_to_trapezium')
    
    mass_mean = np.array([np.mean(sources[f'mass_{model_name}'][(sources['sep_to_trapezium'] > sep_min) & (sources['sep_to_trapezium'] <= sep_max)].to(u.solMass).value) for sep_min, sep_max in zip(separations[:-1], separations[1:])])*u.solMass
    mass_std = np.array([np.std(sources[f'mass_{model_name}'][(sources['sep_to_trapezium'] > sep_min) & (sources['sep_to_trapezium'] <= sep_max)].to(u.solMass).value) for sep_min, sep_max in zip(separations[:-1], separations[1:])])*u.solMass
    
    return mass_mean, mass_std


def mean_mass_equally_spaced(sources, nbins, model_name):
    """Mean mass vs separation from the Trapezium, equally spaced.

    Parameters
    ----------
    sources : pd.DataFrame
    nbins : int
        number of bins.
    model_name : str
        mass model name
    save_path : str
        save path.

    Returns
    -------
    separations : astropy.Quantity
        calculated separations, e.g. [0, 1, 2, 3, 4]*u.arcmin.
    (mass_mean, mass_std): (ndarray, ndarray)
        mean mass and the associated standard deviation within each bin.
    """
    # filter nans.
    sources.remove_rows(np.isnan(sources[f'mass_{model_name}']))
    separations = np.linspace(0, 4, nbins+1) * u.arcmin
    # sources_in_bins = [len(_) for _ in [sources.loc[(sources.sep_to_trapezium > r_min) & (sources.sep_to_trapezium <= r_max)] for r_min, r_max in zip(separation_arcmin[:-1], separation_arcmin[1:])]]
    return separations, mean_mass_binned(sources, separations, model_name)


def mean_mass_equally_grouped(sources, ngroups, model_name):
    """Mean mass vs separation from the Trapezium, equally grouped.

    Parameters
    ----------
    sources : pd.DataFrame
    nbins : int
        number of bins.
    model_name : str
        mass model name
    save_path : str
        save path.

    Returns
    -------
    separations : astropy.Quantity
        calculated separations, e.g. [0, 1, 2, 3, 4]*u.arcmin.
    (mass_mean, mass_std): (ndarray, ndarray)
        mean mass and the associated standard deviation within each bin.
    """
    # filter nans.
    sources.remove_rows(np.isnan(sources[f'mass_{model_name}']))
    
    sources_in_bins = [len(sources) // ngroups + (1 if x < len(sources) % ngroups else 0) for x in range (ngroups)]
    division_idx = np.cumsum(sources_in_bins)[:-1] - 1
    separation_sorted = np.sort(sources['sep_to_trapezium']).to(u.arcmin).value
    separations = np.array([0, *(separation_sorted[division_idx] + separation_sorted[division_idx+1])/2, 4]) * u.arcmin
    
    return separations, mean_mass_binned(sources, separations, model_name)


def mean_mass_vs_separation(sources:QTable, nbins:int, ngroups:int, model_name:str, save_path:str):
    """Mean mass vs separation, equally spaced and equally grouped

    Parameters
    ----------
    sources : QTable
        Sources
    nbins : int
        Number of bins
    ngroups : int
        Number of groups
    model_name : str
        Model name for stellar mass
    save_path : str
        Save path
    """
    
    # filter nans.
    sources = copy.deepcopy(sources)
    sources.remove_rows(np.isnan(sources[f'mass_{model_name}']))
    
    # Left: equally spaced.
    separation_borders, (mass_mean, mass_std) = mean_mass_equally_spaced(sources, nbins, model_name)
    
    # average separation within each bin.
    separation_sources = np.array([np.mean(sources['sep_to_trapezium'][(sources['sep_to_trapezium'] > min_sep) & (sources['sep_to_trapezium'] <= max_sep)]).to(u.arcmin).value for min_sep, max_sep in zip(separation_borders[:-1], separation_borders[1:])])
    
    sources_in_bins = [sum((sources['sep_to_trapezium'] > sep_min) & (sources['sep_to_trapezium'] <= sep_max)) for sep_min, sep_max in zip(separation_borders[:-1], separation_borders[1:])]
    
    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.5), dpi=300, sharey=True)
    
    # Left figure
    ax1.set_xlim((0, 4))
    ax1.set_xticks([0, 1, 2, 3])
    ax1.plot(separation_sources, mass_mean.to(u.solMass).value, marker='o', markersize=5, label=r"$\overline{M}$")
    ax1.fill_between(separation_sources, y1=(mass_mean-mass_std).to(u.solMass).value, y2=(mass_mean+mass_std).to(u.solMass).value, edgecolor='none', facecolor='C0', alpha=0.3, label='$\sigma_M$')
    ylim = ax1.get_ylim()
    for i in range(len(sources_in_bins)):
        ax1.annotate(f'{sources_in_bins[i]}', (separation_sources[i], mass_mean.to(u.solMass).value[i] + (ylim[1] - ylim[0])/20), fontsize=10, horizontalalignment='center')
    ax1.legend(fontsize=12, loc='upper left')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel('Separation from Trapezium (arcmin)', fontsize=15)
    ax1.set_ylabel(r'Mean Stellar Mass $(M_{\odot})$', fontsize=15)
    
    ax1_upper = ax1.twiny()
    ax1_upper.tick_params(axis='both', labelsize=12)
    ax1_upper.set_xlim((0, 4/60*np.pi/180*389))
    ax1_upper.set_xlabel('Separation from Trapezium (pc)\n', fontsize=15)
    ax1_upper.set_title('Equally Spaced\n', fontsize=15)
    
    
    # Right: equally grouped.
    separation_borders, (mass_mean, mass_std) = mean_mass_equally_grouped(sources, ngroups, model_name)
    
    # average separation within each bin.
    separation_sources = np.array([np.mean(sources['sep_to_trapezium'][(sources['sep_to_trapezium'] > min_sep) & (sources['sep_to_trapezium'] <= max_sep)]).to(u.arcmin).value for min_sep, max_sep in zip(separation_borders[:-1], separation_borders[1:])])
        
    sources_in_bins = [len(sources) // ngroups + (1 if x < len(sources) % ngroups else 0) for x in range (ngroups)]
    
    # Right figure
    ax2.set_xlim((0, 4))
    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.plot(separation_sources, mass_mean.to(u.solMass).value, marker='o', markersize=5)
    ax2.fill_between(separation_sources, y1=(mass_mean-mass_std).to(u.solMass).value, y2=(mass_mean+mass_std).to(u.solMass).value, edgecolor='none', facecolor='C0', alpha=0.3)
    for i in range(len(sources_in_bins)):
        ax2.annotate(f'{sources_in_bins[i]}', (separation_sources[i], mass_mean[i].to(u.solMass).value + (ylim[1] - ylim[0])/20), fontsize=10, horizontalalignment='center')
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xlabel('Separation from Trapezium (arcmin)', fontsize=15)
    
    ax2_upper = ax2.twiny()
    ax2_upper.tick_params(axis='both', labelsize=12)
    ax2_upper.set_xlim((0, 4/60*np.pi/180*389))
    ax2_upper.set_xlabel('Separation from Trapezium (pc)\n', fontsize=15)
    ax2_upper.set_title('Equally Grouped\n', fontsize=15)

    fig.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


#################################################
################# Main Function #################
#################################################
MCMC = False
resampling = False

C0 = '#1f77b4'
C1 = '#ff7f0e'
C3 = '#d62728'
C4 = '#9467bd'
C6 = '#e377c2'
C7 = '#7f7f7f'
C9 = '#17becf'

save_path = f'{user_path}/ONC/starrynight/codes/analysis'

orion = ONC(QTable.read(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog - epoch combined.ecsv'))
orion.preprocessing()
orion.set_attr()

trapezium_only = (orion.data['sci_frames'].mask) & (orion.data['APOGEE'].mask)

plot_skymaps(orion)

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

orion.plot_pm_rv(save_path=f'{user_path}/ONC/figures/3D kinematics.pdf')
orion.pm_angle_distribution(save_path=f'{user_path}/ONC/figures/pm direction.pdf')
orion.compare_mass(save_path=f'{user_path}/ONC/figures/mass comparison.pdf')
orion.compare_chris(save_path=f'{user_path}/ONC/figures/compare T22.pdf')


#################################################
########### Relative Velocity vs Mass ###########
#################################################

model_names = ['MIST', 'BHAC15', 'Feiden', 'Palla']
radii = [0.05, 0.1, 0.15, 0.2, 0.25]*u.pc
model_type = 'linear'

# Remove the high relative velocity sources.
max_rv = 40*u.km/u.s
min_rv = 0*u.km/u.s
print(f'Removed {sum((orion.rv < min_rv) | (orion.rv > max_rv))} sources with radial velocity outside {min_rv} ~ {max_rv}.')
mean_diff, maximum_diff = orion.compare_teff_with_apogee(constraint= orion.rv <= max_rv)
print(f'Mean difference in teff: {mean_diff:.2f}.')
print(f'Maximum difference in teff: {maximum_diff:.2f}.')

# teff offset simulation
orion_mean_offset = copy.deepcopy(orion)
orion_mean_offset.data['teff_nirspao'] += mean_diff
orion_mean_offset.teff = fillna(orion_mean_offset.data['teff_nirspao'], orion_mean_offset.data['teff_apogee'])

from starrynight import fit_mass
mean_offset_masses = fit_mass(orion_mean_offset.teff, orion_mean_offset.e_teff).filled(np.nan)
columns = mean_offset_masses.keys()
for column in columns:
    orion_mean_offset.data[column] = mean_offset_masses[column]
orion_mean_offset.set_attr()


# for radius in radii:
#     for model_name in model_names:
        
#         if radius == 0.1*u.pc:
#             update_self = True
#         else:
#             update_self = False
        
#         mass, vrel, e_mass, e_vrel = orion.vrel_vs_mass(
#             model_name=model_name,
#             model_type=model_type,
#             radius=radius,
#             resampling=resampling,
#             min_rv=min_rv,
#             max_rv=max_rv,
#             update_self=update_self,
#             kde_percentile=84,
#             show_figure=False,
#             save_path=f'{save_path}/vrel_results/{model_type}-{radius.value:.2f}pc'
#         )
        
#         mass, vrel, e_mass, e_vrel = orion_mean_offset.vrel_vs_mass(
#             model_name=model_name,
#             model_type=model_type,
#             radius=radius,
#             resampling=resampling,
#             min_rv=min_rv,
#             max_rv=max_rv,
#             update_self=False,
#             kde_percentile=84,
#             show_figure=False,
#             save_path=f'{save_path}/vrel_results/{model_type}-mean-offset-{radius.value:.2f}pc'
#         )

# orion.data.write(f'{user_path}/ONC/starrynight/catalogs/sources with vrel.ecsv', overwrite=True)


model_name = 'MIST'
model_type = 'linear'
ks = np.empty((2, len(radii)))
ks_mean_offset = np.empty((2, len(radii)))

for i, radius in enumerate(radii):
    with open(f'{user_path}/ONC/starrynight/codes/analysis/vrel_results/{model_type}-{radius.value:.2f}pc/{model_name}-{model_type}-{radius.value:.2f}pc params.txt', 'r') as file:
        raw = file.readlines()
    with open(f'{user_path}/ONC/starrynight/codes/analysis/vrel_results/{model_type}-mean-offset-{radius.value:.2f}pc/{model_name}-{model_type}-{radius.value:.2f}pc params.txt', 'r') as file:
        raw_mean_offset = file.readlines()
    
    for line, line_mean_offset in zip(raw, raw_mean_offset):
        if line.startswith('k_resample:\t'):
            ks[:, i] = np.array([float(_) for _ in line.strip('k_resample:\t\n').split('± ')])
        if line_mean_offset.startswith('k_resample:\t'):
            ks_mean_offset[:, i] = np.array([float(_) for _ in line_mean_offset.strip('k_resample:\t\n').split('± ')])


colors = ['C0', 'C3']
fig, ax = plt.subplots()
blue_errorbar  = ax.errorbar(radii.value, ks[0], yerr=ks[1], color=colors[0], fmt='o-', markersize=5, capsize=5, zorder=2)
red_errorbar   = ax.errorbar(radii.value, ks_mean_offset[0], yerr=ks_mean_offset[1], color=colors[1], fmt='o--', markersize=5, capsize=5, zorder=3)
blue_fill      = ax.fill_between(radii.value, y1=ks[0]-ks[1], y2=ks[0]+ks[1], edgecolor='none', facecolor=colors[0], alpha=0.4, zorder=1)
red_fill       = ax.fill_between(radii.value, y1=ks_mean_offset[0]-ks_mean_offset[1], y2=ks_mean_offset[0] + ks_mean_offset[1], edgecolor='none', facecolor=colors[1], alpha=0.4, zorder=4)

hline = ax.hlines(0, xmin=min(radii.value), xmax=max(radii.value), linestyles=':', lw=2, colors='k', zorder=0)
ax.legend(handles=[(blue_errorbar, blue_fill), (red_errorbar, red_fill), hline], labels=[f'Original {model_name} Model', 'Average Offset NIRSPAO Teff', 'Zero Slope'], fontsize=12)
ax.set_xlabel('Separation Limits of Neighbors (pc)')
ax.set_ylabel('Slope of Linear Fit (k)')
plt.savefig(f'{user_path}/ONC/figures/slope vs sep.pdf', bbox_inches='tight', transparent=True)
plt.show()


#################################################
############## Velocity Dispersion ##############
#################################################
# Apply rv constraint
rv_constraint = ((
    abs(orion.rv - np.nanmean(orion.rv)) <= 3*np.nanstd(orion.rv)
    ) | (
        trapezium_only
))
print(f'3σ RV constraint for velocity dispersion: {sum(rv_constraint) - sum(trapezium_only)} out of {len(rv_constraint) - sum(trapezium_only)} remains.')
print(f'Accepted radial velocity range: {np.nanmean(orion.rv):.3f} ± {3*np.nanstd(orion.rv):.3f} km/s.')

if not os.path.exists(f'{user_path}/ONC/starrynight/codes/analysis/vdisp_results'):
    os.mkdir(f'{user_path}/ONC/starrynight/codes/analysis/vdisp_results')

with open(f'{user_path}/ONC/starrynight/codes/analysis/vdisp_results/mean_rv.txt', 'w') as file:
    file.write(str(np.nanmean(orion.rv[rv_constraint])))

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(orion.data['sep_to_trapezium'].to(u.arcmin).value, orion.rv.value, yerr=orion.e_rv.value, fmt='.', label='Measurements')
ax.hlines([np.nanmean(orion.rv.value) - 3*np.nanstd(orion.rv.value), np.nanmean(orion.rv.value) + 3*np.nanstd(orion.rv.value)], xmin=min(orion.data['sep_to_trapezium'].to(u.arcmin).value), xmax=max(orion.data['sep_to_trapezium'].to(u.arcmin).value), linestyles='--', colors='C1', label='3σ range')
ax.set_xlabel('Separation From Trapezium (arcmin)')
ax.set_ylabel('Radial Velocity')
ax.legend()
plt.show()

# # vdisp for all
# vdisps_all = vdisp_all(orion.data[rv_constraint], save_path=f'{save_path}/vdisp_results', MCMC=MCMC)

# # vdisp vs sep
# vdisp_vs_sep(orion.data[rv_constraint], nbins=8, ngroups=8, save_path=f'{save_path}/vdisp_results/vdisp_vs_sep', MCMC=MCMC)

# vdisp vs mass
# vdisp_vs_mass(orion.data[rv_constraint], model_name='MIST', ngroups=8, save_path=f'{save_path}/vdisp_results/vdisp_vs_mass', MCMC=MCMC)



# #################################################
# ################ Mass Segregation ###############
# #################################################

lambda_msr_with_trapezium = mass_segregation_ratio(orion.data, model_name='MIST', use_literature_trapezium_mass=True, save_path=f'{user_path}/ONC/figures/MSR-MIST-all.pdf')
lambda_msr_no_trapezium = mass_segregation_ratio(orion.data, model_name='MIST', use_literature_trapezium_mass=False, save_path=f'{user_path}/ONC/figures/MSR-MIST-no trapezium.pdf')

# mean_mass_vs_separation(orion.data[~trapezium_only], nbins=10, ngroups=10, model_name='MIST', save_path=f'{user_path}/ONC/figures/mass vs separation - MIST.pdf')

print('--------------------Finished--------------------')