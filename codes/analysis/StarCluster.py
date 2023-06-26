import os
import copy
import numpy as np
from numpy import ma
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats
from scipy.optimize import curve_fit
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
        
        mass = getattr(self, f'mass_{model_name}')
        e_mass = getattr(self, f'e_mass_{model_name}')
        
        constraint = \
            (~np.isnan(self.pmRA)) & (~np.isnan(self.pmDE)) & \
            (~np.isnan(mass)) & \
            (e_mass <= max_mass_error) & \
            (self.rv >= min_rv) & (self.rv <= max_rv) & \
            (self.e_v <= max_v_error)
        
        sources_coord = self.coord[constraint]
        
        mass = mass[constraint]
        e_mass = e_mass[constraint]
        e_v = self.e_v[constraint]
        
        ############# calculate vcom within radius #############
        # v & vcom: n-by-3 velocity in cartesian coordinates
        v = self.v_xyz.T[constraint, :]
        vcom = np.empty((sum(constraint), 3))*u.km/u.s
        e_vcom = np.empty(sum(constraint))*u.km/u.s
        
        # is_neighbor: boolean symmetric neighborhood matrix
        is_neighbor = np.empty((len(sources_coord), len(sources_coord)), dtype=bool)
        
        for i, star in enumerate(sources_coord):
            sep = star.separation_3d(sources_coord)
            if self_included:
                is_neighbor[i] = sep < radius
            else:
                is_neighbor[i] = (sep > 0*u.pc) & (sep < radius)
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
        
        if model_type=='linear':
            def model_func(x, k, b):
                return k*x + b
        elif model_type=='power':
            def model_func(x, k, b):
                return b*x**k
        else:
            raise ValueError(f"model_func must be one of 'linear' or 'power', not {model_func}.")
        
        R = np.corrcoef(mass.to(u.solMass).value, vrel.to(u.km/u.s).value)[1, 0]   # Pearson's R
        
        # Resampling
        if resampling is True:
            resampling = 100000
        if resampling:
            ks = np.empty(resampling)
            ebs = np.empty(resampling)
            Rs = np.empty(resampling)
            
            for i in range(resampling):
                mass_resample = np.random.normal(loc=mass, scale=e_mass)
                vrel_resample = np.random.normal(loc=vrel, scale=e_vrel)
                valid_resample_idx = (mass_resample > 0) & (vrel_resample > 0)
                mass_resample = mass_resample[valid_resample_idx]
                vrel_resample = vrel_resample[valid_resample_idx]
                popt, _ = curve_fit(model_func, mass_resample, vrel_resample)
                ks[i] = popt[0]
                ebs[i] = popt[1]        
                Rs[i] = np.corrcoef(mass_resample, vrel_resample)[1, 0]
            
            k_resample = np.median(ks)
            k_e = np.diff(np.percentile(ks, [16, 84]))[0]/2
            b_resample = np.median(ebs)
            b_e = np.diff(np.percentile(ebs, [16, 84]))[0]/2
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
        
        print(f'k_resample = {k_resample:.2f} ± {k_e:.2f}')
        print(f'R = {R:.2f}, R_resample = {R_resample:.2f}')
        
        
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
        avrg_vrel_binned    = np.empty(nbins)
        e_vrel_binned       = np.empty(nbins)
        vrel_weight = 1 / e_vrel_value**2
        
        for i, min_mass, max_mass in zip(range(nbins), mass_borders[:-1]*u.solMass, mass_borders[1:]*u.solMass):
            idx = (mass > min_mass) & (mass <= max_mass)
            mass_weight_sum = sum(mass_weight[idx])
            mass_binned_avrg[i] = np.average(mass_value[idx], weights=mass_weight[idx])
            e_mass_binned[i] = 1/mass_weight_sum * sum(mass_weight[idx] * e_mass_value[idx])
            
            vrel_weight_sum = sum(vrel_weight[idx])
            avrg_vrel_binned[i] = np.average(vrel_value[idx], weights=vrel_weight[idx])
            e_vrel_binned[i] = 1/vrel_weight_sum * sum(vrel_weight[idx] * e_vrel_value[idx])
        
        # write params
        if save_path:
            with open(f'{save_path}/{model_name}-{model_type}-{radius.value:.2f}pc params.txt', 'w') as file:
                file.write(f'Median of neighbors in a group:\t{np.median(n_neighbors):.0f}\n')
                file.write(f'k_resample:\t{k_resample} ± {k_e}\n')
                file.write(f'b_resample:\t{b_resample} ± {b_e}\n')
                file.write(f'R_resample:\t{R_resample} ± {R_e}\n')
                file.write(f'R:\t{R}\n')
        
        
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
            mass_binned_avrg, avrg_vrel_binned, 
            xerr=e_mass_binned, 
            yerr=e_vrel_binned, 
            fmt='.', 
            elinewidth=1.2, ecolor='C3', 
            markersize=8, markeredgecolor='none', markerfacecolor='C3', 
            alpha=0.8,
            zorder=4
        )
        
        # Running Average Fill
        f2 = ax.fill_between(mass_binned_avrg, avrg_vrel_binned - e_vrel_binned, avrg_vrel_binned + e_vrel_binned, color='C3', edgecolor='none', alpha=0.5)
            
        h4, = ax.plot(xs, model_func(xs, k_resample, b_resample), color='k', label='Best Fit', zorder=3)
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
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
        
        if model_type=='linear':
            labels.append(f'Best Linear Fit:\n$k={k_resample:.2f}\pm{k_e:.2f}$\n$b={b_resample:.2f}\pm{b_e:.2f}$')
        elif model_type=='power':
            labels.append(f'Best Fit:\n$k={k_resample:.2f}\pm{k_e:.2f}$\n$A={b_resample:.2f}\pm{b_e:.2f}$')
        
        ax.legend(handles, labels)
        
        
        at = AnchoredText(
            f'$R={R_resample:.2f}\pm{R_e:.2f}$', 
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
        self.mass_BHAC15 = self.data['mass_MIST']
        self.e_mass_BHAC15 = self.data['e_mass_MIST']
        self.mass_Feiden = self.data['mass_Feiden']
        self.e_mass_Feiden = self.data['e_mass_Feiden']
        self.mass_Palla = self.data['mass_Palla']
        self.e_mass_Palla = self.data['e_mass_Palla']
        self.mass_Hillenbrand = self.data['mass_Hillenbrand']
    
    
    def preprocessing(self):
        trapezium_only = (self.data['sci_frames'].mask) & (self.data['APOGEE'].mask)
        max_rv_e = 5*u.km/u.s
        rv_constraint = ((
            (self.data['e_rv_nirspao']   <= max_rv_e) |
            (self.data['e_rv_apogee']    <= max_rv_e)
        ) | (
            trapezium_only
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
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.errorbar(
            (self.data['pmRA_gaia'] - self.data['pmRA_kim'] - offset_RA).value,
            (self.data['pmDE_gaia'] - self.data['pmDE_kim'] - offset_DE).value,
            xerr = ((self.data['e_pmRA_gaia']**2 + self.data['e_pmDE_kim']**2)**0.5).value,
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
        self.data.remove_rows(~dist_constraint)
        
        # Calculate velocity
        self.calculate_velocity(self.data['pmRA'], self.data['e_pmRA'], self.data['pmDE'], self.data['e_pmDE'], self.data['rv'], self.data['e_rv'], dist=389*u.pc, e_dist=3*u.pc)
        
        super().set_coord(ra=self.data['RAJ2000'], dec=self.data['DEJ2000'], pmRA=self.data['pmRA'], pmDE=self.data['pmDE'], rv=self.data['rv'], distance=389*u.pc)
        self.data['sep_to_trapezium'] = self.coord.separation(trapezium)
        
        print('After all constraint:\nNIRSPAO:\t{}\nAPOGEE:\t{}\nMatched:\t{}\nTotal:\t{}'.format(
            sum(~self.data['sci_frames'].mask),
            sum(~self.data['APOGEE'].mask),
            sum((~self.data['sci_frames'].mask) & (~self.data['APOGEE'].mask)),
            self.len - sum((self.data['sci_frames'].mask) & (self.data['APOGEE'].mask))
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
            labels=['MIST', 'BHAC15', 'Feiden', 'Palla', 'Hillenbrand'], 
            limit=limit,
            save_path=save_path
        )
    
    
    def compare_chris(self, save_path=None):
        idx_binary = list(self.data['HC2000']).index(546)
        idx_other = np.delete(np.array(range(self.len)), np.array(idx_binary))

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
            self.data['teff'][idx_binary].value, 
            self.data['teff_chris'][idx_binary].value, 
            xerr=self.data['e_teff'][idx_binary].value, 
            yerr=self.data['e_teff_chris'][idx_binary].value, 
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
            self.data['rv'][idx_binary].value, 
            self.data['rv_chris'][idx_binary].value, 
            xerr=self.data['e_rv'][idx_binary].value, 
            yerr=self.data['e_rv_chris'][idx_binary].value, 
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
                plt.savefig(f'{save_path}/compare Chris.pdf', bbox_inches='tight', transparent=True)
            else:
                plt.savefig(f'{save_path}/compare Chris.pdf', bbox_inches='tight')
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
    
    
    def vdisp_all(self, save_path, MCMC=True):
        '''Fit velocity dispersion for all sources, kim, and gaia respectively.

        Parameters
        ----------
        sources: pd.DataFrame
            sources dataframe with keys: vRA, vRA_e, vDE, vDE_e, rv, rv_e.
        save_path: str
            Folder under which to save.
        MCMC: bool
            Run MCMC or read from existing fitting results (default True).
        
        Returns
        -------
        vdisps_all: dict
            velocity dispersion for all sources.
            vdisps_all[key] = [value, error].
            keys: mu_RA, mu_DE, mu_rv, sigma_RA, sigma_DE, sigma_rv, rho_RA, rho_DE, rho_rv.
        '''
        # all velocity dispersions
        print('Fitting for all velocity dispersion...')
        vdisps_all = fit_vdisp(
            self,
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
            self.loc[~self.ID_kim.isna()].reset_index(drop=True),
            save_path=f'{save_path}/kim', 
            MCMC=MCMC
        )

        # gaia velocity dispersions
        print('Fitting for Gaia velocity dispersion...')
        fit_vdisp(
            self.loc[~self.ID_gaia.isna()].reset_index(drop=True),
            save_path=f'{save_path}/gaia',
            MCMC=MCMC
        )
        
        return vdisps_all


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

save_path = f'{user_path}/ONC/starrynight/codes/analysis'

orion = ONC(QTable.read(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog - epoch combined.ecsv'))
orion.preprocessing()
orion.set_attr()

trapezium_only = (orion.data['sci_frames'].mask) & (orion.data['APOGEE'].mask)

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
# orion.plot_pm_rv(constraint=~trapezium_only)
# orion.pm_angle_distribution()
# orion.compare_mass()
# orion.compare_chris()


#################################################
########### Relative Velocity vs Mass ###########
#################################################

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

model_names = ['MIST', 'BHAC15', 'Feiden', 'Palla']
radii = [0.05, 0.1, 0.15, 0.2, 0.25]*u.pc
base_path = 'linear'
base_path_mean_offset = 'linear-mean-offset'

for radius in radii:
    for model_name in model_names:
        
        if (radius == 0.1*u.pc) and (model_name == 'MIST'):
            update_self = True
        else:
            update_self = False
        
        mass, vrel, e_mass, e_vrel = orion.vrel_vs_mass(
            model_name=model_name,
            model_type='linear',
            radius=radius,
            resampling=False,
            min_rv=min_rv,
            max_rv=max_rv,
            update_self=update_self,
            kde_percentile=84,
            show_figure=False,
            save_path=f'{save_path}/vrel_results/{base_path}-{radius.value:.2f}pc'
        )
        
        mass, vrel, e_mass, e_vrel = orion_mean_offset.vrel_vs_mass(
            model_name=model_name,
            model_type='linear',
            radius=radius,
            resampling=False,
            min_rv=min_rv,
            max_rv=max_rv,
            update_self=False,
            kde_percentile=84,
            show_figure=False,
            save_path=f'{save_path}/vrel_results/{base_path_mean_offset}-{radius.value:.2f}pc'
        )

orion.data.write(f'{user_path}/ONC/starrynight/catalogs/sources with vrel.ecsv', overwrite=True)


model_name = 'MIST'
ks = np.empty((2, len(radii)))
ks_mean_offset = np.empty((2, len(radii)))

for i, radius in enumerate(radii):
    with open(f'{user_path}/ONC/starrynight/codes/analysis/vrel_results/{base_path}-{radius.value:.2f}pc/{model_name}-linear-{radius.value:.2f}pc params.txt', 'r') as file:
        raw = file.readlines()
    with open(f'{user_path}/ONC/starrynight/codes/analysis/vrel_results/{base_path_mean_offset}-{radius.value:.2f}pc/{model_name}-linear-{radius.value:.2f}pc params.txt', 'r') as file:
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
ax.errorbar(orion.data['sep_to_trapezium'].value, orion.rv.value, yerr=orion.e_rv.value, fmt='.', label='Measurements')
ax.hlines([np.nanmean(orion.rv.value) - 3*np.nanstd(orion.rv.value), np.nanmean(orion.rv.value) + 3*np.nanstd(orion.rv.value)], xmin=min(orion.data['sep_to_trapezium'].value), xmax=max(orion.data['sep_to_trapezium'].value), linestyles='--', colors='C1', label='3σ range')
ax.set_xlabel('Separation From Trapezium (arcmin)')
ax.set_ylabel('Radial Velocity')
ax.legend()
plt.show()