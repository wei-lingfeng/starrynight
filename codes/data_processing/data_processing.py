import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import copy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
import plotly.figure_factory as ff
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from fit_vdisp import fit_vdisp
from typing import Tuple
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


#################################################
################### Functions ###################
#################################################

def corner_plot(data, labels, limit, save_path):
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


def merge(array1, array2):
    '''
    Union of two lists with nans.
    If value in array1 exists, take array1. Otherwise, take array2.
    '''
    merge = copy.copy(array1[:])
    
    if array1.dtype.name.startswith('float'):
        nan_idx = np.isnan(array1)
    # else:
    #     nan_idx = array1=='nan'
    merge[nan_idx] = array2[nan_idx]
    return merge


def weighted_avrg_and_merge(array1, array2, error1=None, error2=None):
    '''
    Calculate weighted avearge of two 1-D arrays
    ----------------
    - Parameters:
        - array1, array2: 1-D value arrays.
        - error1, error2: 1-D error arrays.
    ----------------
    - Returns:
        - result: 1-D array of weighted average
        - result_e: 1-D array of uncertainty.
    '''
    array1 = np.array([array1]).flatten()
    array2 = np.array([array2]).flatten()
    error1 = np.array([error1]).flatten()
    error2 = np.array([error2]).flatten()
    
    N_stars = len(array1)
    avrg = np.empty(N_stars)
    avrg_e = np.empty(N_stars)
    value_in_1 = np.logical_and(~np.isnan(array1),  np.isnan(array2))   # value in 1 ONLY
    value_in_2 = np.logical_and( np.isnan(array1), ~np.isnan(array2))   # value in 2 ONLY
    value_both = np.logical_and(~np.isnan(array1), ~np.isnan(array2))   # value in both
    value_none = np.logical_and( np.isnan(array1),  np.isnan(array2))   # value in none

    avrg[value_in_1] = array1[value_in_1]
    avrg[value_in_2] = array2[value_in_2]
    avrg[value_none] = np.nan
    
    if not ((error1 is None) or (error2 is None)):
        avrg[value_both] = np.average(
            [array1[value_both], array2[value_both]], 
            axis=0, 
            weights=[1 / error1[value_both]**2, 1 / error2[value_both]**2]
        )
        
        avrg_e[value_in_1] = error1[value_in_1]
        avrg_e[value_in_2] = error2[value_in_2]
        avrg_e[value_both] = 1 / np.sqrt(
            1 / error1[value_both]**2 + 1 / error2[value_both]**2
        )
        avrg_e[value_none] = np.nan
        
        if N_stars==1:
            return avrg[0], avrg_e[0]
        else:
            return avrg, avrg_e
        
    else:
        avrg[value_both] = np.average(
            [array1[value_both], array2[value_both]], 
            axis=0
        )
        
        if N_stars==1:
            return avrg[0]
        else:
            return avrg


def hex_to_rgb(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in range(0, len(value), 2))


#################################################
################## Parallax Cut #################
#################################################

def normal(x, mu=0, sigma=1, amplitude=1):
    return amplitude * np.exp(-1/2 * ((x-mu)/sigma)**2)
    
# def fit_dist(sources):
#     '''Fit a Gaussian profile to the distance distribution
#     '''
#     constraint = ~sources.dist.isna() & (sources.dist < 600)
#     mu = np.average(sources.dist[constraint], weights=1/sources.plx_e[constraint]**2)
#     sigma = np.sqrt(np.average((sources.dist[constraint] - mu)**2, weights=1/sources.plx_e[constraint]**2))
    
#     fig, ax = plt.subplots()
#     n, bins, patches = ax.hist(sources.dist, range=(350, 450), bins=30)
#     x = np.linspace(350, 450, 100)
#     plt.plot(x, normal(x, mu=mu, sigma=sigma, amplitude=max(n)))
#     return mu, sigma

# Depricated:
def apply_dist_constraint(dist, dist_e, dist_range=15, dist_error_range=15, min_plx_over_e=5):
    '''Apply a distance constraint on sources.
    ----------
    - Parameters:
        - dist: sources distance in pc.
        - dist_e: sources distance error in pc.
        - dist_range: allowed distance range in +/- pc. Default is 15 pc.
        - dist_error_range: allowed distance error in pc. Default is 15 pc.
        - min_plx_over_e: minimum parallax over error. Default is 5, as per https://www.aanda.org/articles/aa/full_html/2021/05/aa39834-20/aa39834-20.html.
    
    - Returns:
        - center_dist: center distance that gives the most matches.
        - plx_constraint: boolean list of filter result.
    '''
    
    plx = 1000/dist
    plx_e = dist_e / dist * plx
    
    trial_dists = np.arange(np.nanmedian(dist) - dist_range, np.nanmedian(dist) + dist_range)
    n_matches = [sum(
        (abs(dist - trial_dist) < dist_range) & 
        (dist_e < dist_error_range)
    ) for trial_dist in trial_dists]

    center_dist = trial_dists[np.argmax(n_matches)]
    dist_constraint = (abs(dist - center_dist) < dist_range) & (dist_e < dist_error_range) & (plx/plx_e > min_plx_over_e)
    
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(dist, range=(center_dist - 3*dist_range, center_dist + 3*dist_range), bins=21, alpha=0.5)
    # x = np.linspace(350, 450, 100)
    # ax.plot(x, normal(x, mu=center_dist, sigma=dist_range, amplitude=max(n)))
    ax.vlines([center_dist-dist_range, center_dist+dist_range], ymin=0, ymax=max(n), colors='C3', linestyles='--', linewidth=1.5)
    ax.set_xlabel('Distance (pc)')
    ax.set_ylabel('Counts')
    plt.show()
    
    print(f'Distance constraint applied: {center_dist} \u00B1 {dist_range} pc, maximum uncertainty: {dist_error_range} pc.')
    print(f'Number of matches: {sum(dist_constraint)}')
    return center_dist, dist_constraint

def distance_cut(dist, dist_e, min_dist=300, max_dist=500, min_plx_over_e=5):
    plx = 1000/dist
    plx_e = dist_e / dist * plx
    dist_constraint = (dist >= min_dist) & (dist <= max_dist) & (plx/plx_e >= min_plx_over_e) | np.isnan(dist)
    print(f'{min_dist}~{max_dist} pc distance range constraint: {sum(dist_constraint) - 5} sources out of {len(dist_constraint) - 5} remains.')
    return dist_constraint

#################################################
############## Calculate velocity ###############
#################################################

def calculate_velocity(sources, dist=389, dist_e=3):
    '''Calculate velocity
    ------------------------------
    - Parameters:
        - sources: pandas DataFrame containing keys: pmRA, pmRA_e, pmDE, pmDE_e, rv, rv_e.
        - dist: distance to target in parsecs. Default is 389 pc uniform.
        - dist_e: distance uncertainty in parsecs.
    
    - Returns:
        - sources: pandas DataFrame with tangential & total velocity and errors updated in km/s.
    '''
    vRA = 4.74e-3 * sources.pmRA * dist
    vRA_e = 4.74e-3 * np.sqrt((sources.pmRA_e * dist)**2 + (dist_e * sources.pmRA)**2)
    
    vDE = 4.74e-3 * sources.pmDE * dist
    vDE_e = 4.74e-3 * np.sqrt((sources.pmDE_e * dist)**2 + (dist_e * sources.pmDE)**2)
    
    pm = np.sqrt(sources.pmRA**2 + sources.pmDE**2)
    pm_e = 1/pm * np.sqrt((sources.pmRA * sources.pmRA_e)**2 + (sources.pmDE * sources.pmDE_e)**2)
    
    vt = 4.74e-3 * pm * dist
    vt_e = vt * np.sqrt((pm_e / pm)**2 + (dist_e / dist)**2)
    
    vr = sources.rv
    vr_e = sources.rv_e
    
    v = np.sqrt(vt**2 + vr**2)
    v_e = 1/v * np.sqrt((vt * vt_e)**2 + (vr * vr_e)**2)
    
    sources = copy.deepcopy(sources)
    sources['vRA'] = vRA
    sources['vRA_e'] = vRA_e
    sources['vDE'] = vDE
    sources['vDE_e'] = vDE_e
    sources['vt'] = vt
    sources['vt_e'] = vt_e
    sources['v'] = v
    sources['v_e'] = v_e
    return sources


#################################################
######### Compare Proper Motion and RV ##########
#################################################

def compare_velocity(sources, save_path=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4))
    ax1.errorbar(sources.pmRA_kim.values, sources.pmRA_gaia.values, xerr=sources.pmRA_e_kim.values, yerr=sources.pmRA_e_gaia.values, fmt='o', color=(.2, .2, .2, .8), alpha=0.4, markersize=3)
    ax1.plot([-2, 3], [-2, 3], color='C3', linestyle='--', label='Equal Line')
    ax1.set_xlabel(r'$\mu_{\alpha^*, HK} \quad \left(\mathrm{mas}\cdot\mathrm{yr}^{-1}\right)$')
    ax1.set_ylabel(r'$\mu_{\alpha^*, DR3} - \widetilde{\Delta\mu_{\alpha^*}} \quad \left(\mathrm{mas}\cdot\mathrm{yr}^{-1}\right)$')
    ax1.legend()
    
    ax2.errorbar(sources.pmDE_kim.values, sources.pmDE_gaia.values, xerr=sources.pmDE_e_kim.values, yerr=sources.pmDE_e_gaia.values, fmt='o', color=(.2, .2, .2, .8), alpha=0.4, markersize=3)
    ax2.plot([-2, 3], [-2, 3], color='C3', linestyle='--', label='Equal Line')
    ax2.set_xlabel(r'$\mu_{\delta, HK} \quad \left(\mathrm{mas}\cdot\mathrm{yr}^{-1}\right)$')
    ax2.set_ylabel(r'$\mu_{\delta, DR3} - \widetilde{\Delta\mu_{\alpha^*}} \quad \left(\mathrm{mas}\cdot\mathrm{yr}^{-1}\right)$')
    ax2.legend()
    
    ax3.errorbar(sources.rv_helio.values, sources.rv_apogee.values, xerr=sources.rv_e_nirspec.values, yerr=sources.rv_e_apogee.values, fmt='o', color=(.2, .2, .2, .8), alpha=0.4, markersize=3)
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


#################################################
#################### Plot 2D ####################
#################################################

def plot_2d(sources, scale=0.004):
    '''Generate 2D plots of position and velocity.
    - Parameters:
        - sources_coord: astropy coordinates with ra, dec, pm_ra_cosdec, pm_dec.
        - scale: scale of quiver, 0.004 by default.
    - Returns:
        - fig: figure handle.
    '''
    
    line_width=2
    opacity=0.8
    marker_size = 6

    nirspec_flag    = np.logical_and(~sources.HC2000.isna(), sources.ID_apogee.isna())
    apogee_flag     = np.logical_and(sources.HC2000.isna(), ~sources.ID_apogee.isna())
    matched_flag    = np.logical_and(~sources.HC2000.isna(), ~sources.ID_apogee.isna())

    trapezium = SkyCoord("05h35m16.26s", "-05d23m16.4s")

    fig_data = [
        # nirspec quiver
        ff.create_quiver(
            sources.loc[nirspec_flag, 'RAJ2000'],
            sources.loc[nirspec_flag, 'DEJ2000'],
            sources.loc[nirspec_flag, 'pmRA'],
            sources.loc[nirspec_flag, 'pmDE'],
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
            sources.loc[apogee_flag, 'RAJ2000'],
            sources.loc[apogee_flag, 'DEJ2000'],
            sources.loc[apogee_flag, 'pmRA'],
            sources.loc[apogee_flag, 'pmDE'],
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
            sources.loc[matched_flag, 'RAJ2000'], 
            sources.loc[matched_flag, 'DEJ2000'], 
            sources.loc[matched_flag, 'pmRA'], 
            sources.loc[matched_flag, 'pmDE'], 
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
            x=sources.loc[nirspec_flag, 'RAJ2000'], 
            y=sources.loc[nirspec_flag, 'DEJ2000'], 
            mode='markers',
            marker=dict(
                size=marker_size,
                color=C0
            )
        ),
        
        # apogee scatter
        go.Scatter(
            name='APOGEE Sources',
            x=sources.loc[apogee_flag, 'RAJ2000'], 
            y=sources.loc[apogee_flag, 'DEJ2000'], 
            mode='markers',
            marker=dict(
                size=marker_size,
                color=C4
            )
        ),
        
        # matched scatter
        go.Scatter(
            name='Matched Sources',
            x=sources.loc[matched_flag, 'RAJ2000'], 
            y=sources.loc[matched_flag, 'DEJ2000'], 
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


def plot_pm_rv(sources, save_path=None):
    center = SkyCoord("05h35m17.5s", "-05d23m16.4s")
    image_path = f'{user_path}/ONC/figures/skymap/hlsp_orion_hst_acs_colorimage_r_v1_drz.fits'
    hdu = fits.open(image_path)[0]
    wcs = WCS(image_path)
    box_size = 5200
    # Cutout. See https://docs.astropy.org/en/stable/nddata/utils.html.
    cutout = Cutout2D(hdu.data, position=center, size=(box_size, box_size), wcs=wcs)
    a = 1e4
    image_data = ((np.power(a, cutout.data/255) - 1)/a)*255
    # image_data_zoom = (cutout.data/255)**2*255
    image_wcs = cutout.wcs
    ra_wcs, dec_wcs = image_wcs.wcs_world2pix(sources.RAJ2000, sources.DEJ2000, 0)
    
    fig = plt.figure(figsize=(7, 6), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=image_wcs)
    ax.imshow(image_data, cmap='gray')
    
    im = ax.quiver(
        ra_wcs,
        dec_wcs,
        -sources.pmRA,
        sources.pmDE,
        sources.rv,
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


    
#################################################
#################### Plot 3D ####################
#################################################

def plot_3d(sources_coord_3d, scale=3):
    '''Generate 3D plots of position and velocity.
    ----------
    - Parameters:
        - sources_3d: astropy SkyCoord.
        - scale: scale of cone.
    - Returns:
        - fig1: scatter plot.
        - fig2: cone plot with velocity.
    '''
    
    marker_size = 3
    opacity = 0.7
    
    # ~HC2000 322
    trapezium = SkyCoord("05h35m16.26s", "-05d23m16.4s", distance=1000/2.59226*u.pc)
    
    # v = np.linalg.norm(sources_coord_3d.velocity.d_xyz.value, axis=0)
    
    fig1_data = [
        go.Scatter3d(
            mode='markers',
            name='NIRSPEC + APOGEE',
            x=sources_coord_3d.cartesian.x.value,
            y=sources_coord_3d.cartesian.y.value,
            z=sources_coord_3d.cartesian.z.value,
            marker=dict(
                size=marker_size,
                color=C0,
                opacity=opacity
            )
        ),
        
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
    ]

    fig1 = go.Figure(fig1_data)
    
    fig1.update_layout(
        width=700,
        height=700,
        scene = dict(
            xaxis_title='X (pc)',
            yaxis_title='Y (pc)',
            zaxis_title='Z (pc)',
            aspectratio=dict(x=1, y=1, z=1)
        )
    )
    fig1.show()


    fig2_data = go.Cone(
        x=sources_coord_3d.cartesian.x.value,
        y=sources_coord_3d.cartesian.y.value,
        z=sources_coord_3d.cartesian.z.value,
        u=sources_coord_3d.velocity.d_x.value,
        v=sources_coord_3d.velocity.d_y.value,
        w=sources_coord_3d.velocity.d_z.value,
        sizeref=scale,
        colorscale='Blues',
        colorbar=dict(title=r'Velocity $\left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$'), #, thickness=20),
        colorbar_title_side='right'
    )
    
    fig2 = go.Figure(fig2_data)
    
    fig2.update_layout(
        width=700,
        height=700,
        scene = dict(
            xaxis_title='X (pc)',
            yaxis_title='Y (pc)',
            zaxis_title='Z (pc)',
            aspectratio=dict(x=1, y=1, z=1)
        )
    )
    
    fig2.show()

    return fig1, fig2


#################################################
########### Relative Velocity vs Mass ###########
#################################################

def vrel_vs_mass(sources, model_name, radius=0.1*u.pc, model_type='linear', resampling=100000, self_included=True, max_rv=np.inf, max_v_error=5., max_mass_error=0.5, kde_percentile=84, update_sources=False, save_path=None, **kwargs):
    """Velocity relative to the neighbors of each source within a radius vs mass.

    Parameters
    ----------
    sources : pd.DataFrame
        sources
    model_name : str
        one of ['MIST', 'BHAC15', 'Feiden', 'Palla']
    radius : astropy.Quantity, optional
        radius within which count as neighbors, by default 0.1*u.pc
    model_func : str, optional
        format of model function: 'linear' or 'power'. V=k*M + b or V=A*M**k, by default 'linear'.
    resampling : any, optional
        whether resample or not, and number of resamples, 100000 by default. If False or None, will try to read from previous results
    self_included : bool, optional
        include the source itself or not when calculating the center of mass velocity of its neighbors, by default True
    max_rv : float, optional
        maximum radial velocity, by default inf.
    max_rv_error : float, optional
        maximum radial velocity error, by default 5
    max_mass_error : float, optional
        maximum mass error, by default 0.5
    update_sources : bool, optional
        update the original sources dataframe or not, by default False
    kde_percentile : int, optional
        Percentile of KDE contour, 84 by default
    show_figure : bool, optional
        whether to show the figure
    save_path : str, optional
        save path, by default None
    kwargs:
        bin_method: str, optional
            binning method when calculating running average, 'equally spaced' or 'equally grouped', by default 'equally grouped'
        nbins: int, optional
            number of bins, by default 7 for 'equally grouped' and 5 for 'equally spaced'.

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
    
    bin_method = kwargs.get('bin_method', 'equally grouped')
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    # Avoid changing original data.
    # sources = copy.deepcopy(sources)    # only valid entries.
    
    constraint = \
        (~sources.pmRA.isna()) & (~sources.pmDE.isna()) & \
        (~sources[f'mass_{model_name}'].isna()) & \
        (sources[f'mass_e_{model_name}'] < max_mass_error) & \
        (sources.rv < max_rv) & (sources.v_e < max_v_error)
    
    # sources = sources.loc[constraint].reset_index(drop=True)
    sources_coord = SkyCoord(
        ra=sources.loc[constraint, 'RAJ2000'].to_numpy()*u.degree,
        dec=sources.loc[constraint, 'DEJ2000'].to_numpy()*u.degree,
        distance=sources.loc[constraint, 'dist'].to_numpy()*u.pc,
        pm_ra_cosdec=sources.loc[constraint, 'pmRA'].to_numpy()*u.mas/u.yr,
        pm_dec=sources.loc[constraint, 'pmDE'].to_numpy()*u.mas/u.yr,
        radial_velocity=sources.loc[constraint, 'rv'].to_numpy()*u.km/u.s
    )
    
    mass = sources.loc[constraint, f'mass_{model_name}']
    mass_e = sources.loc[constraint, f'mass_e_{model_name}']
    v_e = sources.loc[constraint, 'v_e']
    
    ############# calculate vcom within radius #############
    # vel & vel_com: n-by-3 velocity in cartesian coordinates
    v = np.array([coord.velocity.d_xyz.value for coord in sources_coord])
    # vel_rel = np.empty((len(sources_coord), 3))
    vcom = np.empty((len(sources_coord), 3))
    vcom_e = np.empty(len(sources_coord))
    
    # neighbors[i] = list of boolean. Self included.
    neighbors = []
    
    for i, star in enumerate(sources_coord):
        sep = star.separation_3d(sources_coord)
        if self_included:
            neighbors.append(sep < radius)
        else:
            neighbors.append((sep > 0*u.pc) & (sep < radius))
        # vel_com[i]: 1-by-3 center of mass velocity
        vcom[i] = (mass[neighbors[i]] @ v[neighbors[i]]) / sum(mass[neighbors[i]])
    
    neighbors = np.array(neighbors)
    n_neighbors = np.sum(neighbors, axis=1)
    
    # delete those without any neighbors
    if self_included:
        no_neighbor = n_neighbors==1
    else:
        no_neighbor = n_neighbors==0
        
    v = np.delete(v, no_neighbor, axis=0)
    vcom = np.delete(vcom, no_neighbor, axis=0)
    vcom_e = np.delete(vcom_e, no_neighbor, axis=0)
    neighbors = np.delete(neighbors, no_neighbor, axis=1)
    n_neighbors = np.delete(n_neighbors, no_neighbor)
    # sources = sources.drop(no_neighbor).reset_index(drop=True)
    valid_idx = np.where(constraint)[0][~no_neighbor]
    
    sources_coord = SkyCoord(
        ra=sources.loc[valid_idx, 'RAJ2000'].to_numpy()*u.degree,
        dec=sources.loc[valid_idx, 'DEJ2000'].to_numpy()*u.degree,
        distance=sources.loc[valid_idx, 'dist'].to_numpy()*u.pc,
        pm_ra_cosdec=sources.loc[valid_idx, 'pmRA'].to_numpy()*u.mas/u.yr,
        pm_dec=sources.loc[valid_idx, 'pmDE'].to_numpy()*u.mas/u.yr,
        radial_velocity=sources.loc[valid_idx, 'rv'].to_numpy()*u.km/u.s
    )
    
    mass = sources.loc[valid_idx, f'mass_{model_name}'].to_numpy()
    mass_e = sources.loc[valid_idx, f'mass_e_{model_name}'].to_numpy()

    v_e = sources.loc[valid_idx, 'v_e'].to_numpy()
    
    print(f'Median neighbors in a group: {np.median(n_neighbors):.0f}')
    
    vrel_vector = v - vcom
    vrel = np.linalg.norm(vrel_vector, axis=1)
    
    ############# calculate vrel error #############
    for i in range(len(sources_coord)):
        vcom_e_j = np.sqrt(
            [sum((vrel_vector[neighbors[i], j]/sum(mass[neighbors[i]]) * mass_e[neighbors[i]])**2 + (mass[neighbors[i]] / sum(mass[neighbors[i]]) * v_e[neighbors[i]])**2) for j in range(3)]
        )
        
        vcom_e[i] = np.sqrt(sum([(vcom[i,j] / np.linalg.norm(vcom[i]) * vcom_e_j[j])**2 for j in range(3)]))
    
    vrel_e = np.sqrt(v_e**2 + vcom_e**2)
    
    
    ############# Resampling #############
    
    if model_type=='linear':
        def model_func(x, k, b):
            return k*x + b
    elif model_type=='power':
        def model_func(x, k, b):
            return b*x**k
    else:
        raise ValueError(f"model_func must be one of 'linear' or 'power', not {model_func}.")
    
    R = np.corrcoef(mass, vrel)[1, 0]   # Pearson's R
    
    # Resampling
    if resampling is True:
        resampling = 100000
    if resampling:
        ks = np.empty(resampling)
        ebs = np.empty(resampling)
        Rs = np.empty(resampling)
        
        for i in range(resampling):
            mass_resample = np.random.normal(loc=mass, scale=mass_e)
            vrel_resample = np.random.normal(loc=vrel, scale=vrel_e)
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
        mass_borders = np.array([np.nanmin(mass) - 1e-3, *(mass_sorted[division_idx] + mass_sorted[division_idx + 1])/2, np.nanmax(mass)])
    
    # equally spaced
    elif bin_method == 'equally spaced':
        nbins = kwargs.get('nbins', 5)
        mass_borders = np.linspace(np.nanmin(mass) - 1e-3, np.nanmax(mass), nbins + 1)
    
    else:
        raise ValueError("bin_method must be one of the following: ['equally grouped', 'equally spaced']")
    
    mass_binned_avrg    = np.empty(nbins)
    mass_binned_e       = np.empty(nbins)
    mass_weight = 1 / mass_e**2
    vrel_binned_avrg    = np.empty(nbins)
    vrel_binned_e       = np.empty(nbins)
    vrel_weight = 1 / vrel_e **2
    
    for i, min_mass, max_mass in zip(range(nbins), mass_borders[:-1], mass_borders[1:]):
        idx = (mass > min_mass) & (mass <= max_mass)
        mass_weight_sum = sum(mass_weight[idx])
        mass_binned_avrg[i] = np.average(mass[idx], weights=mass_weight[idx])
        mass_binned_e[i] = 1/mass_weight_sum * sum(mass_weight[idx] * mass_e[idx])
        
        vrel_weight_sum = sum(vrel_weight[idx])
        vrel_binned_avrg[i] = np.average(vrel[idx], weights=vrel_weight[idx])
        vrel_binned_e[i] = 1/vrel_weight_sum * sum(vrel_weight[idx] * vrel_e[idx])
    
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
    X, Y = np.meshgrid(np.linspace(0, mass.max(), resolution), np.linspace(0, vrel.max(), resolution))
    positions = np.vstack([X.T.ravel(), Y.T.ravel()])
    values = np.vstack([mass, vrel])
    kernel = stats.gaussian_kde(values)
    Z = np.rot90(np.reshape(kernel(positions).T, X.shape))
    
    
    ########## Linear Fit Plot - Original Error ##########
    xs = np.linspace(mass.min(), mass.max(), 100)
    
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    
    # Errorbar with uniform transparency
    h1 = ax.errorbar(
        mass, vrel, xerr=mass_e, yerr=vrel_e,
        fmt='.',
        markersize=6, markeredgecolor='none', markerfacecolor='C0', 
        elinewidth=1, ecolor='C0', alpha=0.5,
        zorder=2
    )
    
    # Running Average
    h2 = ax.errorbar(
        mass_binned_avrg, vrel_binned_avrg, 
        xerr=mass_binned_e, 
        yerr=vrel_binned_e, 
        fmt='.', 
        elinewidth=1.2, ecolor='C3', 
        markersize=8, markeredgecolor='none', markerfacecolor='C3', 
        alpha=0.8,
        zorder=4
    )
    
    # Running Average Fill
    f2 = ax.fill_between(mass_binned_avrg, vrel_binned_avrg - vrel_binned_e, vrel_binned_avrg + vrel_binned_e, color='C3', edgecolor='none', alpha=0.5)
        
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
    im = ax.imshow(Z, cmap=my_cmap, alpha=0.8, extent=[0, mass.max(), 0, vrel.max()], zorder=0, aspect='auto')
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
    plt.close()
    
    ########## Updating the original DataFrame ##########
    if update_sources:
        print('Changing sources...')
        sources.loc[valid_idx, f'vrel_{model_name}'] = vrel
        sources.loc[valid_idx, f'vrel_e_{model_name}'] = vrel_e
    else:
        pass
    
    return mass, vrel, mass_e, vrel_e


#################################################
############## Velocity Dispersion ##############
#################################################

def vdisp_all(sources, save_path, MCMC=True):
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
        sources,
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
        sources.loc[~sources.ID_kim.isna()].reset_index(drop=True),
        save_path=f'{save_path}/kim', 
        MCMC=MCMC
    )

    # gaia velocity dispersions
    print('Fitting for Gaia velocity dispersion...')
    fit_vdisp(
        sources.loc[~sources.ID_gaia.isna()].reset_index(drop=True),
        save_path=f'{save_path}/gaia',
        MCMC=MCMC
    )
    
    return vdisps_all
    


def vdisp_vs_sep_binned(sources, separations, save_path, MCMC=True):
    '''Velocity dispersion within a bin vs separation from Trapezium in arcmin.
    
    Parameters
    ------------------------
    sources: pandas.DataFrame.
    separations: astropy quantity array.
        separation borders, e.g., np.array([0, 1, 2, 3, 4]) * u.arcmin.
    save_path: str.
        save path.
    MCMC: boolean.
        run MCMC or load existing fitting results.
    
    
    Returns
    ------------------------
    vdisps: list of velocity dispersion results of each bin. length = len(separations) - 1
        vdisps[i] = pd.DataFrame({'mu_RA': [value, error], 'mu_DE': [value, error], 'mu_rv': [value, error], ...})
        keys: mu_RA, mu_DE, mu_rv, sigma_RA, sigma_DE, sigma_rv, rho_RA, rho_DE, rho_rv.
    '''
    sources = sources.loc[~(sources.vRA.isna() | sources.vDE.isna() | sources.rv.isna())].reset_index(drop=True)
    
    separations_arcmin = separations.to(u.arcmin).value
    separations_arcmin_binned = (separations_arcmin[:-1] + separations_arcmin[1:])/2
    
    # binned velocity dispersions
    vdisps = []
    sources_in_bins = []
    for i, min_sep, max_sep in zip(range(len(separations_arcmin_binned)), separations_arcmin[:-1], separations_arcmin[1:]):
        if MCMC:
            print(f'Start fitting for bin {i}...')
        
        sources_in_bins.append(sum((sources.sep_to_trapezium > min_sep) & (sources.sep_to_trapezium <= max_sep)))
        vdisps.append(fit_vdisp(
            sources.loc[(sources.sep_to_trapezium > min_sep) & (sources.sep_to_trapezium <= max_sep)].reset_index(drop=True),
            save_path=f'{save_path}/bin {i}',
            MCMC=MCMC
        ))
    
    return vdisps



def vdisp_vs_sep_equally_spaced(sources:pd.DataFrame, nbins:int, save_path:str, MCMC:bool) -> Tuple[u.Quantity, dict]:
    """Velocity dispersion vs separations, equally spaced.

    Parameters
    ----------
    sources : pd.DataFrame
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
    sources = sources.loc[~(sources.vRA.isna() | sources.vDE.isna() | sources.rv.isna())].reset_index(drop=True)

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
    sources = sources.loc[~(sources.vRA.isna() | sources.vDE.isna() | sources.rv.isna())].reset_index(drop=True)

    sources_in_bins = [len(sources) // ngroups + (1 if x < len(sources) % ngroups else 0) for x in range (ngroups)]
    division_idx = np.cumsum(sources_in_bins)[:-1] - 1
    separation_sorted = np.sort(sources.sep_to_trapezium)
    separation_borders = np.array([0, *(separation_sorted[division_idx] + separation_sorted[division_idx+1])/2, 4]) * u.arcmin
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
    sources : pd.DataFrame
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
    sources = sources.loc[~(sources.vRA.isna() | sources.vDE.isna() | sources.rv.isna())].reset_index(drop=True)
    
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
    
        
    separations_arcmin = separation_borders.to(u.arcmin).value
    # average separation within each bin.
    separation_sources = np.array([np.mean(sources.loc[(sources.sep_to_trapezium > min_sep) & (sources.sep_to_trapezium <= max_sep), 'sep_to_trapezium']) for min_sep, max_sep in zip(separations_arcmin[:-1], separations_arcmin[1:])])
    
    sources_in_bins = [sum((sources.sep_to_trapezium > min_sep) & (sources.sep_to_trapezium <= max_sep)) for min_sep, max_sep in zip(separations_arcmin[:-1], separations_arcmin[1:])]
    
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

    separations_arcmin = separation_borders.to(u.arcmin).value
    # average separation within each bin.
    separation_sources = np.array([np.mean(sources.loc[(sources.sep_to_trapezium > min_sep) & (sources.sep_to_trapezium <= max_sep), 'sep_to_trapezium']) for min_sep, max_sep in zip(separations_arcmin[:-1], separations_arcmin[1:])])

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

def vdisp_vs_mass_binned(sources, model_name, masses, save_path, MCMC=True):
    '''Velocity dispersion within a bin vs mass.
    
    Parameters
    ------------------------
    sources: pandas.DataFrame.
    masses: np.array.
        separation borders, e.g., np.array([0, 1, 2, 3, 4]).
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
    sources = sources.loc[~(sources.vRA.isna() | sources.vDE.isna() | sources.rv.isna())].reset_index(drop=True)
    
    masses_binned = (masses[:-1] + masses[1:])/2
    
    # binned velocity dispersions
    vdisps = []
    sources_in_bins = []
    for i, min_mass, max_mass in zip(range(len(masses_binned)), masses[:-1], masses[1:]):
        if MCMC:
            print(f'Start fitting for bin {i}...')
        
        sources_in_bins.append(sum((sources[f'mass_{model_name}'] > min_mass) & (sources[f'mass_{model_name}'] <= max_mass)))
        vdisps.append(fit_vdisp(
            sources.loc[(sources[f'mass_{model_name}'] > min_mass) & (sources[f'mass_{model_name}'] <= max_mass)].reset_index(drop=True),
            save_path=f'{save_path}/bin {i}',
            MCMC=MCMC
        ))
    
    return vdisps



def vdisp_vs_mass_equally_grouped(sources:pd.DataFrame, model_name:str, ngroups:int, save_path:str, MCMC:bool):
    """Velocity dispersion vs masses, equally grouped.
    Prioritize higher numbers at closer distance to trapezium. e.g., 10 is divided into 4+3+3.
    
    Parameters
    ----------
    sources : pd.DataFrame
        sources.
    model_name: str.
        model name
    ngroups : int
        number of groups.
    save_path : str
        save path.
    MCMC : bool
        run MCMC or not.

    Returns
    -------
    masses : np.array
        Dividing masses. e.g., [0, 1, 2, 3, 4].
    vdisps : list
        List of velocity dispersion fitting results of length nbins.
        Each element is a dictionary with keys ['mu_RA', 'mu_DE', 'mu_rv', 'sigma_RA', 'sigma_DE', 'sigma_rv', 'rho_RA', 'rho_DE', 'rho_rv'].
    """
    # filter nans.
    sources = sources.loc[~(sources.vRA.isna() | sources.vDE.isna() | sources.rv.isna())].reset_index(drop=True)

    sources_in_bins = [len(sources) // ngroups + (1 if x < len(sources) % ngroups else 0) for x in range (ngroups)]
    division_idx = np.cumsum(sources_in_bins)[:-1] - 1
    mass_sorted = np.sort(sources[f'mass_{model_name}'])
    mass_borders = np.array([0, *(mass_sorted[division_idx] + mass_sorted[division_idx+1])/2, 4])
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
    sources : pd.DataFrame
        sources
    model_name: str
        model name.
    ngroups : int
        number of groups for equally grouped case.
    save_path : str
        save path.
    MCMC : bool
        run mcmc or not.
    """
    # filter nans.
    sources = sources.loc[~(sources.vRA.isna() | sources.vDE.isna() | sources.rv.isna())].reset_index(drop=True)
    
        
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
        mass_sources.append(np.average(sources.loc[bin_idx, f'mass_{model_name}'], weights=1/sources.loc[bin_idx, f'mass_e_{model_name}']**2))
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
    
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, direction, sigma_xx in zip(axs, ['1d', 'pm', 'rv'], [sigma_1d, sigma_pm, sigma_rv]):
        # arcmin axis
        ax.set_ylim((1.3, 6))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=12)
        ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        
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
        ax.tick_params(axis='both', labelsize=12)
    
    axs[0].legend(handles=[(errorbar, fill)], labels=['Measured Velocity Dispersion'], fontsize=12, loc='upper left')
    fig.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(f'{user_path}/ONC/figures/vdisp vs mass.pdf', bbox_inches='tight')
    plt.savefig(f'{user_path}/ONC/figures/vdisp vs mass.png', bbox_inches='tight', transparent=True)
    plt.show()



#################################################
################ Mass Segregation ###############
#################################################

def mass_segregation_ratio(sources: pd.DataFrame, model_name: str, save_path: str, Nmst_min=5, Nmst_max=40, step=1):
    '''Mass segregation ratio. See https://doi.org/10.1111/j.1365-2966.2009.14508.x.
    
    Parameters:
        sources: pandas dataframe.
        model: model name.
        save_path: path to save the figure.
        Nmst_min: minimum Nmst.
        Nmst_max: maximum Nmst.
        step: step.
    
    Returns:
        lambda_msr: N-by-2 array of mass segregation ratio in the form of [[value, error], ...].
    '''
    
    sources = copy.deepcopy(sources)
    # Merge binaries.
    binary_hc2000 = sources.loc[[False if str(_).lower()=='nan' else '_' in _ for _ in list(sources.HC2000)], 'HC2000']
    binary_hc2000_unique = [_.split('_')[0] for _ in binary_hc2000 if _.endswith('_A')]
    # binary_idx_pairs = [[a_idx, b_idx], [a_idx, b_idx], ...]
    binary_idx_pairs = [[binary_hc2000.loc[binary_hc2000==f'{_}_A'].index[0], binary_hc2000.loc[binary_hc2000==f'{_}_B'].index[0]] for _ in binary_hc2000_unique]
    
    for binary_idx_pair in binary_idx_pairs:
        nan_flag = sources.loc[binary_idx_pair, [f'mass_{model_name}', f'mass_e_{model_name}']].isna()
    
        # if any value is valid, update the first place with m=m1+m2, m_e = sqrt(m1_e**2 + m2_e**2) (valid values only). Else (all values are nan), do nothing.
        if any(~nan_flag[f'mass_{model_name}']):
            sources.loc[binary_idx_pair[0], f'mass_{model_name}'] = sum(sources.loc[binary_idx_pair, f'mass_{model_name}'][~nan_flag[f'mass_{model_name}']])
        
        if any(~nan_flag[f'mass_e_{model_name}']):
            sources.loc[binary_idx_pair[0], f'mass_e_{model_name}'] = sum(sources.loc[binary_idx_pair, f'mass_e_{model_name}'][~nan_flag[f'mass_{model_name}']].pow(2))**0.5
            
        # update names to remove '_A', '_B' suffix.
        sources.loc[binary_idx_pair[0], 'HC2000'] = sources.loc[binary_idx_pair[0], 'HC2000'].split('_')[0]
        # remove values in the second place.
        sources = sources.drop(binary_idx_pair[1])

    sources = sources.reset_index(drop=True)

    # Construct separation matrix
    sources_coord = SkyCoord(ra=sources.RAJ2000.to_numpy()*u.degree, dec=sources.DEJ2000.to_numpy()*u.degree)
    sep_matrix = np.zeros((len(sources), len(sources)))

    for i in range(len(sources)):
        sep_matrix[i, :] = sources_coord[i].separation(sources_coord).arcsec
        sep_matrix[i, 0:i] = 0


    # Construct MST for the Nmst most massive sources.
    # lambda: [[value, error], [value, error], ...]
    lambda_msr = np.empty((len(np.arange(Nmst_min, Nmst_max, step)), 2))
    for i, Nmst in enumerate(np.arange(Nmst_min, Nmst_max, step)):
        massive_idx = np.sort(sources.sort_values(f'mass_{model_name}', ascending=False).index[:Nmst])
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
    if max(lambda_msr[:, 0]) / min(lambda_msr[:, 0]) > 10:
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
    sources = sources.sort_values('sep_to_trapezium').reset_index(drop=True)
    
    separations_arcmin = separations.to(u.arcmin).value
    
    mass_mean = np.array([sources.loc[(sources.sep_to_trapezium > sep_min) & (sources.sep_to_trapezium <= sep_max), f'mass_{model_name}'].mean() for sep_min, sep_max in zip(separations_arcmin[:-1], separations_arcmin[1:])])
    mass_std = np.array([sources.loc[(sources.sep_to_trapezium > sep_min) & (sources.sep_to_trapezium <= sep_max), f'mass_{model_name}'].std() for sep_min, sep_max in zip(separations_arcmin[:-1], separations_arcmin[1:])])
    
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
    sources = sources.loc[~sources[f'mass_{model_name}'].isna()].reset_index(drop=True)
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
    sources = sources.loc[~sources[f'mass_{model_name}'].isna()].reset_index(drop=True)
    
    sources_in_bins = [len(sources) // ngroups + (1 if x < len(sources) % ngroups else 0) for x in range (ngroups)]
    division_idx = np.cumsum(sources_in_bins)[:-1] - 1
    separation_sorted = np.sort(sources.sep_to_trapezium)
    separations = np.array([0, *(separation_sorted[division_idx] + separation_sorted[division_idx+1])/2, 4]) * u.arcmin
    
    return separations, mean_mass_binned(sources, separations, model_name)


def mean_mass_vs_separation(sources, nbins, ngroups, model_name, save_path):    
    """Mean mass vs separation, equally spaced and equally grouped.

    Parameters
    ----------
    sources : pd.DataFrame
    nbins : int
        number of bins.
    ngroups : int
        number of groups.
    model_name : str
        model name of stellar mass
    save_path : str
        save path.
    """
    # filter nans.
    sources = sources.loc[~sources[f'mass_{model_name}'].isna()].reset_index(drop=True)
    
    # Left: equally spaced.
    separation_borders, (mass_mean, mass_std) = mean_mass_equally_spaced(sources, nbins, model_name)
    
    separations_arcmin = separation_borders.to(u.arcmin).value
    # average separation within each bin.
    separation_sources = np.array([np.mean(sources.loc[(sources.sep_to_trapezium > min_sep) & (sources.sep_to_trapezium <= max_sep), 'sep_to_trapezium']) for min_sep, max_sep in zip(separations_arcmin[:-1], separations_arcmin[1:])])
    
    sources_in_bins = [sum((sources.sep_to_trapezium > sep_min) & (sources.sep_to_trapezium <= sep_max)) for sep_min, sep_max in zip(separations_arcmin[:-1], separations_arcmin[1:])]
    
    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.5), dpi=300, sharey=True)
    
    # Left figure
    ax1.set_xlim((0, 4))
    ax1.set_xticks([0, 1, 2, 3])
    ax1.plot(separation_sources, mass_mean, marker='o', markersize=5, label=r"$\overline{M}$")
    ax1.fill_between(separation_sources, y1=mass_mean-mass_std, y2=mass_mean+mass_std, edgecolor='none', facecolor='C0', alpha=0.3, label='$\sigma_M$')
    ylim = ax1.get_ylim()
    for i in range(len(sources_in_bins)):
        ax1.annotate(f'{sources_in_bins[i]}', (separation_sources[i], mass_mean[i] + (ylim[1] - ylim[0])/20), fontsize=10, horizontalalignment='center')
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
    
    separations_arcmin = separation_borders.to(u.arcmin).value
    # average separation within each bin.
    separation_sources = np.array([np.mean(sources.loc[(sources.sep_to_trapezium > min_sep) & (sources.sep_to_trapezium <= max_sep), 'sep_to_trapezium']) for min_sep, max_sep in zip(separations_arcmin[:-1], separations_arcmin[1:])])
        
    sources_in_bins = [len(sources) // ngroups + (1 if x < len(sources) % ngroups else 0) for x in range (ngroups)]
    
    # Right figure
    ax2.set_xlim((0, 4))
    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.plot(separation_sources, mass_mean, marker='o', markersize=5)
    ax2.fill_between(separation_sources, y1=mass_mean-mass_std, y2=mass_mean+mass_std, edgecolor='none', facecolor='C0', alpha=0.3)
    for i in range(len(sources_in_bins)):
        ax2.annotate(f'{sources_in_bins[i]}', (separation_sources[i], mass_mean[i] + (ylim[1] - ylim[0])/20), fontsize=10, horizontalalignment='center')
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


def pm_angle_distribution(sources, save_path=None):
    def angle_between(v1, v2):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        return np.arctan2(np.linalg.det(np.array([v1, v2])), np.dot(v1, v2))

    position = np.array([
        -(sources.RAJ2000 - trapezium.ra.degree),
        sources.DEJ2000 - trapezium.dec.degree
    ])

    pm = np.array([
        -sources.pmRA,
        sources.pmDE
    ])

    angles = -np.array([angle_between(position[:, i], pm[:, i]) for i in range(len(sources))])
    # angles += np.pi/12
    # angles[angles > np.pi] = angles[angles > np.pi] - np.pi

    nbins = 12
    hist, bin_edges = np.histogram(angles, nbins, range=(-np.pi, np.pi))

    theta = (bin_edges[:-1] + bin_edges[1:])/2
    colors = plt.cm.viridis(hist / max(hist))

    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(projection='polar')
    ax.bar(theta, hist, width=2*np.pi/nbins, bottom=0.0, color=colors, alpha=0.5)
    ax.set_xticks(np.linspace(np.pi, -np.pi, 8, endpoint=False))
    ax.set_yticks([5, 10, 15, 20])
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


def compare_mass(sources, save_path):
    fltr = ~(
        sources.mass_MIST.isna() | \
        sources.mass_BHAC15.isna() | \
        sources.mass_Feiden.isna() | \
        sources.mass_Palla.isna() | \
        sources.mass_Hillenbrand.isna() | \
        ~sources.theta_orionis.isna()
    )

    mass_corner = np.array([
        sources.loc[fltr, 'mass_MIST'],
        sources.loc[fltr, 'mass_BHAC15'],
        sources.loc[fltr, 'mass_Feiden'],
        sources.loc[fltr, 'mass_Palla'],
        sources.loc[fltr, 'mass_Hillenbrand']
    ])
    
    limit = (0.09, 1.35)
    
    fig = corner_plot(
        data=mass_corner.transpose(),
        labels=['MIST', 'BHAC15', 'Feiden', 'Palla', 'Hillenbrand'], 
        limit=limit,
        save_path=save_path
    )


def compare_chris(sources, save_path=None):
    idx_binary = sources.index[sources.HC2000=='546'].to_list()
    idx_other = np.delete(sources.index.to_numpy(), np.array(idx_binary))

    teff_diff = sources.teff - sources.teff_chris
    rv_diff = sources.loc[idx_other, 'rv'] - sources.loc[idx_other, 'rv_chris']
    print(f'Median absolute Teff difference: {abs(teff_diff).median():.2f} K')
    print(f'Max. absolute Teff difference: {abs(teff_diff).max():.2f} K')
    print(f'Standard deviation of Teff difference: {teff_diff.std():.2f} K')
    print(f'Median absolute RV difference: {abs(rv_diff).median():.2f} km/s')
    print(f'Standard deviation RV difference: {rv_diff.std():.2f} km/s')
    
    # compare veiling parameter of order 33    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
    
    # compare teff
    ax1.plot([3000, 5000], [3000, 5000], linestyle='--', color='C3', label='Equal Line')
    ax1.errorbar(
        sources.loc[idx_other, 'teff'].values, 
        sources.loc[idx_other, 'teff_chris'].values, 
        xerr=sources.loc[idx_other, 'teff_e'].values, 
        yerr=sources.loc[idx_other, 'teff_e_chris'].values, 
        color=(.2, .2, .2, .8), fmt='o', alpha=0.5, markersize=3
    )
    ax1.errorbar(
        sources.loc[idx_binary, 'teff'].values, 
        sources.loc[idx_binary, 'teff_chris'].values, 
        xerr=sources.loc[idx_binary, 'teff_e'].values, 
        yerr=sources.loc[idx_binary, 'teff_e_chris'].values, 
        color='C0', label='Parenago 1837', 
        fmt='o', alpha=0.5, markersize=3
    )
    ax1.legend()
    ax1.set_xlabel(r'$T_\mathrm{eff, This\ Work}$ (K)', fontsize=12)
    ax1.set_ylabel(r'$T_\mathrm{eff, Theissen}$ (K)', fontsize=12)
    
    # compare rv
    ax2.plot([21, 34], [21, 34], linestyle='--', color='C3', label='Equal Line')
    ax2.errorbar(
        sources.loc[idx_other, 'rv'].values, 
        sources.loc[idx_other, 'rv_chris'].values, 
        xerr=sources.loc[idx_other, 'rv_e'].values, 
        yerr=sources.loc[idx_other, 'rv_e_chris'].values, 
        color=(.2, .2, .2, .8), fmt='o', alpha=0.5, markersize=3
    )
    ax2.errorbar(
        sources.loc[idx_binary, 'rv'].values, 
        sources.loc[idx_binary, 'rv_chris'].values, 
        xerr=sources.loc[idx_binary, 'rv_e'].values, 
        yerr=sources.loc[idx_binary, 'rv_e_chris'].values, 
        color='C0', label='Parenago 1837', 
        fmt='o', alpha=0.5, markersize=3
    )
    ax2.legend()
    ax2.set_xlabel(r'$\mathrm{RV}_\mathrm{This\ Work}$ $\left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$', fontsize=12)
    ax2.set_ylabel(r'$\mathrm{RV}_\mathrm{Theissen}$ $\left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$', fontsize=12)
    
    # compare veiling param    
    ax3.hist(sources.veiling_param_O33_chris, bins=20, range=(0, 1), histtype='step', color='C0', label='T22', lw=2)
    ax3.hist(sources.veiling_param_O33, bins=20, range=(0, 1), histtype='step', color='C3', label='This work', lw=1.2)
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


def compare_teff_with_apogee(sources, save_path=None):
    diffs = sources.teff_apogee - sources.teff_nirspec
    valid_idx = ~diffs.isna()
    diffs = diffs[valid_idx]
    weights = 1/(sources.teff_e_nirspec**2 + sources.teff_e_apogee**2)[valid_idx]
    
    mean_diff = np.average(diffs, weights=weights)
    max_diff = max(diffs)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(sources.teff_nirspec.values, sources.teff_apogee.values, xerr=sources.teff_e_nirspec.values, yerr=sources.teff_e_apogee.values, fmt='o', color=(.2, .2, .2, .8), alpha=0.5, markersize=3)
    ranges = np.array([3600, 4800]) 
    ax.plot(ranges, ranges, linestyle='--', color='C3', label='Equal Line')
    ax.plot(ranges - mean_diff/2, ranges + mean_diff/2, linestyle=':', color='C0', label=f'Median Difference: {mean_diff:.2f} K')
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


def pm_to_v(pm, dist):
    '''Convert proper motion in mas/yr to km/s.
    Parameters:
        pm: proper motion in mas/yr.
        dist: distance in pc.
    Returns:
        vt: transverse velocity in km/s.
    Note:
        It can be approximated as pm(mas) * dist(pc) * 4.74*10^(-3) km/s.
        Or: 4.74 * pm(mas) * dist(kpc).
    '''
    vt = pm/1000/60/60*np.pi/180*dist * 30856775814913.673 / 31557600
    return vt


#################################################
############## Data Pre-processing ##############
#################################################

def preprocessing(sources):
    trapezium_names = ['A', 'B', 'C', 'D', 'E']

    # Replace Trapezium stars fitting results with literature values.
    for i in range(len(trapezium_names)):
        trapezium_index = sources.loc[sources.theta_orionis == trapezium_names[i]].index[-1]
        for model_name in ['BHAC15', 'MIST', 'Feiden', 'Palla']:
            sources.loc[trapezium_index, [f'mass_{model_name}', f'mass_e_{model_name}']] = [sources.loc[trapezium_index, 'mass_literature'], sources.loc[trapezium_index, 'mass_e_literature']]


    # Apply rv error constraint.
    max_rv_e = 5

    rv_constraint = ((
        (sources.rv_e_nirspec <= max_rv_e) |
        (sources.rv_e_apogee <= max_rv_e)
    ) | (
        ~sources.theta_orionis.isna()
    ))

    print(f'Maximum RV error of {max_rv_e} km/s constraint: {sum(rv_constraint) - sum(~sources.theta_orionis.isna())} out of {len(rv_constraint) - sum(~sources.theta_orionis.isna())} remaining.')

    sources = sources.loc[rv_constraint].reset_index(drop=True)

    rv_use_apogee = (sources.rv_e_nirspec > max_rv_e) & (sources.rv_e_apogee <= max_rv_e)
    sources.loc[rv_use_apogee, ['rv_nirspec', 'rv_e_nirspec']] = sources.loc[rv_use_apogee, ['rv_apogee', 'rv_e_apogee']]

    # Apply gaia constraint.
    gaia_columns = [key for key in sources.keys() if (key.endswith('gaia') | key.startswith('plx') | key.startswith('Gmag') | key.startswith('astrometric') | (key=='ruwe') | (key=='bp_rp'))]
    # gaia_filter = (sources.astrometric_excess_noise <= 1) & (sources.ruwe <= 1.4)
    gaia_filter = (sources.astrometric_gof_al < 16) & (sources.Gmag < 16)
    sources.loc[~gaia_filter, gaia_columns] = np.nan

    offset_RA = np.nanmedian(sources.pmRA_gaia - sources.pmRA_kim)
    offset_DE = np.nanmedian(sources.pmDE_gaia - sources.pmDE_kim)
    print(f'offset in RA and DEC is {(offset_RA, offset_DE)} mas/yr.')
    with open(f'{user_path}/ONC/starrynight/codes/data_processing/pm_offset.txt', 'w') as file:
        file.write(f'pmRA_gaia - pmRA_kim = {offset_RA}\npmDE_gaia - pmDE_kim = {offset_DE}')

    # Plot pm comparison
    fig, ax = plt.subplots(figsize=(6, 6))
    markers, caps, bars = ax.errorbar(
        sources.pmRA_gaia - sources.pmRA_kim - offset_RA,
        sources.pmDE_gaia - sources.pmDE_kim - offset_DE,
        xerr = (sources.pmRA_e_gaia**2 + sources.pmRA_e_kim**2)**0.5,
        yerr = (sources.pmDE_e_gaia**2 + sources.pmDE_e_kim**2)**0.5,
        fmt='o', color=(.2, .2, .2, .8), markersize=3, ecolor='black', alpha=0.4
    )

    ax.hlines(0, -3.2, 1.4, colors='C3', ls='--')
    ax.vlines(0, -2.6, 1.6, colors='C3', ls='--')
    ax.set_xlabel(r'$\mu_{\alpha^*, DR3} - \mu_{\alpha^*, HK} - \widetilde{\Delta\mu_{\alpha^*}} \quad \left(\mathrm{mas}\cdot\mathrm{yr}^{-1}\right)$', fontsize=12)
    ax.set_ylabel(r'$\mu_{\delta, DR3} - \mu_{\delta, HK} - \widetilde{\Delta\mu_\delta} \quad \left(\mathrm{mas}\cdot\mathrm{yr}^{-1}\right)$', fontsize=12)
    ax.set_xlim((-3.2, 1.4))
    ax.set_ylim((-2.6, 1.6))
    plt.savefig(f'{user_path}/ONC/figures/Proper Motion Comparison.pdf')
    plt.show()

    # Correct Gaia values?
    sources.pmRA_gaia -= offset_RA
    sources.pmDE_gaia -= offset_DE
    # sources.pmRA_e_kim = np.sqrt(sources.pmRA_e_kim**2 + ((offset[2] - offset[1])/2)**2)

    # merge proper motion and rv
    # prioritize kim
    sources['pmRA'] = merge(sources.pmRA_kim, sources.pmRA_gaia)
    sources['pmRA_e'] = merge(sources.pmRA_e_kim, sources.pmRA_e_gaia)
    sources['pmDE'] = merge(sources.pmDE_kim, sources.pmDE_gaia)
    sources['pmDE_e'] = merge(sources.pmDE_e_kim, sources.pmDE_e_gaia)

    # choose one from the two options: weighted avrg or prioritize nirspec.
    # # weighted average
    # sources['rv'], sources['rv_e'] = weighted_avrg_and_merge(sources.rv_helio, sources.rv_apogee, error1=sources.rv_e_nirspec, error2=sources.rv_e_apogee)
    # prioritize nirspec values
    sources['rv'] = merge(sources.rv_helio, sources.rv_apogee)
    sources['rv_e'] = merge(sources.rv_e_nirspec, sources.rv_e_apogee)

    sources['dist'] = 1000/sources.plx
    sources['dist_e'] = sources.dist / sources.plx * sources.plx_e


    # Apply dist constraint
    dist_constraint = distance_cut(sources.dist, sources.dist_e)

    # Construct 2-D & 3-D Coordinates
    # 2-D
    sources_2d = sources.loc[dist_constraint].reset_index(drop=True)
    sources_2d = calculate_velocity(sources_2d)
    sources_2d.plx = 1000/389
    sources_2d.dist = 389
    sources_2d.dist_e = 3
    sources_coord_2d = SkyCoord(
        ra=sources_2d.RAJ2000.to_numpy()*u.degree,
        dec=sources_2d.DEJ2000.to_numpy()*u.degree, 
        pm_ra_cosdec=sources_2d.pmRA.to_numpy()*u.mas/u.yr,
        pm_dec=sources_2d.pmDE.to_numpy()*u.mas/u.yr
    )
    sources_2d['sep_to_trapezium'] = sources_coord_2d.separation(trapezium).arcmin  # arcmin


    # # 3-D
    # # center_dist, dist_constraint = apply_dist_constraint(sources.dist, sources.dist_e, min_plx_over_e=5)
    # dist_constraint = distance_cut(sources.dist, sources.dist_e)
    # sources_3d = sources.loc[dist_constraint].reset_index(drop=True)
    # sources_3d = calculate_velocity(sources_3d, dist=sources_3d.dist, dist_e=sources_3d.dist_e)
    # sources_coord_3d = SkyCoord(
    #     ra=sources_3d.RAJ2000.to_numpy()*u.degree, 
    #     dec=sources_3d.DEJ2000.to_numpy()*u.degree, 
    #     distance=1000/sources_3d.plx.to_numpy()*u.pc,
    #     pm_ra_cosdec=sources_3d.pmRA.to_numpy()*u.mas/u.yr,
    #     pm_dec=sources_3d.pmDE.to_numpy()*u.mas/u.yr,
    #     radial_velocity=sources_3d.rv.to_numpy()*u.km/u.s
    # )

    print('After all constraint:\nNIRSPEC:\t{}\nAPOGEE:\t{}\nMatched:\t{}\nTotal:\t{}'.format(
        sum((sources_2d.theta_orionis.isna()) & (~sources_2d.HC2000.isna())),
        sum((sources_2d.theta_orionis.isna()) & (~sources_2d.ID_apogee.isna())),
        sum((sources_2d.theta_orionis.isna()) & (~sources_2d.HC2000.isna()) & (~sources_2d.ID_apogee.isna())),
        sum((sources_2d.theta_orionis.isna()))
    ))

    sources_2d.to_csv(f'{user_path}/ONC/starrynight/catalogs/sources 2d.csv', index=False)
    return sources, sources_2d


#################################################
################# Main Function #################
#################################################
global user_path
user_path = os.path.expanduser('~')

MCMC = False
Multiprocess = True

np.seterr(all="ignore")

C0 = '#1f77b4'
C1 = '#ff7f0e'
C3 = '#d62728'
C4 = '#9467bd'
C6 = '#e377c2'
C7 = '#7f7f7f'
C9 = '#17becf'

sources = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/synthetic catalog - epoch combined.csv', dtype={'ID_gaia': str})
save_path = f'{user_path}/ONC/starrynight/codes/data_processing'

chris_table = pd.read_csv(f'{user_path}/ONC/starrynight/catalogs/Chris\'s Table.csv')

trapezium = SkyCoord("05h35m16.26s", "-05d23m16.4s")

print('Before any constraint:\nNIRSPEC:\t{}\nAPOGEE:\t{}\nMatched:\t{}\nTotal:\t{}'.format(
    sum((sources.theta_orionis.isna()) & (~sources.HC2000.isna())),
    sum((sources.theta_orionis.isna()) & (~sources.ID_apogee.isna())),
    sum((sources.theta_orionis.isna()) & (~sources.HC2000.isna()) & (~sources.ID_apogee.isna())),
    sum((sources.theta_orionis.isna()))
))

# preprocessing
sources, sources_2d = preprocessing(sources)



compare_velocity(sources_2d, save_path=f'{user_path}/ONC/figures/Velocity Comparison.pdf')
# Save html figures
fig_2d = plot_2d(sources_2d)
fig_2d.write_html(f'{user_path}/ONC/figures/sky 2d.html')
# fig_3d1, fig_3d2 = plot_3d(sources_coord_3d)
# fig_3d2.write_html(f'{user_path}/ONC/figures/sky 3d small.html')
plot_pm_rv(sources_2d.loc[sources_2d.theta_orionis.isna()].reset_index(drop=True), save_path=f'{user_path}/ONC/figures/3D kinematics.pdf')

# pm angle distribution
# pm_angle_distribution(sources_2d)
pm_angle_distribution(sources_2d.loc[sources_2d.theta_orionis.isna()].reset_index(drop=True), save_path=f'{user_path}/ONC/figures/pm direction.pdf')

# compare mass
compare_mass(sources_2d, save_path=f'{user_path}/ONC/figures/mass comparison.pdf')

# compare with Chris
compare_chris(sources_2d, save_path=f'{user_path}/ONC/figures')


#################################################
########### Relative Velocity vs Mass ###########
#################################################

# Local velocity
model_names = ['MIST', 'BHAC15', 'Feiden', 'Palla']
radii = [0.05, 0.1, 0.15, 0.2, 0.25]*u.pc
base_path = 'linear'
base_path_mean_offset = 'linear-mean-offset'

# Remove the high relative velocity sources.
rv_threshold = 40
print(f'Removed {sum(sources_2d.rv > rv_threshold)} sources with radial velocity exceeding {rv_threshold} km/s.')

mean_diff, maximum_diff = compare_teff_with_apogee(sources_2d.loc[sources_2d.rv <= rv_threshold].reset_index(drop=True))
print(f'Mean difference in teff: {mean_diff:.2f} K')
print(f'Maximum difference in teff: {maximum_diff:.2f} K')

# teff offset simulation
sources_mean_offset = copy.deepcopy(sources_2d)
sources_mean_offset.teff_nirspec += mean_diff
sources_mean_offset.teff = sources_mean_offset.teff_nirspec.fillna(sources_mean_offset.teff_apogee)

from starrynight import fit_mass
mean_offset_masses = fit_mass(sources_mean_offset.teff, sources_mean_offset.teff_e)
columns = mean_offset_masses.keys()
sources_mean_offset[columns] = mean_offset_masses
indices = ~sources_mean_offset.theta_orionis.isna()
sources_mean_offset.loc[indices, columns] = np.nan

for radius in radii:
    for model_name in model_names:
        
        if radius == 0.1*u.pc:
            update_sources = True
        else:
            update_sources = False
        
        mass, vrel, mass_e, vrel_e = vrel_vs_mass(
            sources_2d, 
            model_name, 
            model_type='linear',
            radius=radius, 
            resampling=False,
            max_rv=rv_threshold,
            update_sources=update_sources,
            kde_percentile=84,
            show_figure=False,
            save_path=f'{save_path}/vrel_results/{base_path}-{radius.value:.2f}pc'
        )

        mass, vrel, mass_e, vrel_e = vrel_vs_mass(
            sources_mean_offset,
            model_name,
            model_type='linear',
            radius=radius,
            resampling=False,
            max_rv=rv_threshold,
            update_sources=False,
            kde_percentile=84,
            show_figure=False,
            save_path=f'{save_path}/vrel_results/{base_path_mean_offset}-{radius.value:.2f}pc'
        )

# write sources_2d with vrel
sources_2d.to_csv(f'{user_path}/ONC/starrynight/catalogs/sources with vrel.csv', index=False)


model_name = 'MIST'
ks = np.empty((2, len(radii)))
ks_mean_offset = np.empty((2, len(radii)))

for i, radius in enumerate(radii):
    with open(f'{user_path}/ONC/starrynight/codes/data_processing/vrel_results/{base_path}-{radius.value:.2f}pc/{model_name}-linear-{radius.value:.2f}pc params.txt', 'r') as file:
        raw = file.readlines()
    with open(f'{user_path}/ONC/starrynight/codes/data_processing/vrel_results/{base_path_mean_offset}-{radius.value:.2f}pc/{model_name}-linear-{radius.value:.2f}pc params.txt', 'r') as file:
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
    abs(sources_2d.rv - np.nanmean(sources_2d.rv)) <= 3*np.nanstd(sources_2d.rv)
    ) | (
        ~sources_2d.theta_orionis.isna()
))
print(f'3σ RV constraint for velocity dispersion: {sum(rv_constraint) - sum(~sources_2d.theta_orionis.isna())} out of {len(rv_constraint) - sum(~sources_2d.theta_orionis.isna())} remains.')
print(f'Accepted radial velocity range: {np.nanmean(sources.rv):.3f} ± {3*np.nanstd(sources.rv):.3f} km/s.')

if not os.path.exists(f'{user_path}/ONC/starrynight/codes/data_processing/vdisp_results'):
    os.mkdir(f'{user_path}/ONC/starrynight/codes/data_processing/vdisp_results')

with open(f'{user_path}/ONC/starrynight/codes/data_processing/vdisp_results/mean_rv.txt', 'w') as file:
    file.write(str(np.nanmean(sources_2d.loc[rv_constraint, 'rv'])))

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(sources_2d.sep_to_trapezium, sources_2d.rv, yerr=sources_2d.rv_e, fmt='.', label='Measurements')
ax.hlines([np.nanmean(sources_2d.rv) - 3*np.nanstd(sources_2d.rv), np.nanmean(sources_2d.rv) + 3*np.nanstd(sources_2d.rv)], xmin=min(sources_2d.sep_to_trapezium), xmax=max(sources_2d.sep_to_trapezium), linestyles='--', colors='C1', label='3σ range')
ax.set_xlabel('Separation From Trapezium (arcmin)')
ax.set_ylabel('Radial Velocity')
ax.legend()
plt.show()

vdisps_all = vdisp_all(sources_2d.loc[(sources_2d.theta_orionis.isna()) & rv_constraint].reset_index(drop=True), save_path=f'{save_path}/vdisp_results', MCMC=MCMC)

# vdisp vs sep
vdisp_vs_sep(sources_2d.loc[(sources_2d.theta_orionis.isna()) & rv_constraint].reset_index(drop=True), 8, 8, save_path=f'{save_path}/vdisp_results/vdisp_vs_sep', MCMC=MCMC)

# vdisp vs mass
vdisp_vs_mass(sources_2d.loc[(sources_2d.theta_orionis.isna()) & rv_constraint].reset_index(drop=True), model_name='MIST', ngroups=8, save_path=f'{save_path}/vdisp_results/vdisp_vs_mass', MCMC=MCMC)

#################################################
################ Mass Segregation ###############
#################################################

lambda_msr_with_trapezium = mass_segregation_ratio(sources_2d, model_name='MIST', save_path=f'{user_path}/ONC/figures/MSR-MIST-all.pdf')
lambda_msr_no_trapezium = mass_segregation_ratio(sources_2d.loc[sources_2d.theta_orionis.isna()].reset_index(drop=True), model_name='MIST', save_path=f'{user_path}/ONC/figures/MSR-MIST-no trapezium.pdf')

mean_mass_vs_separation(sources_2d.loc[sources_2d.theta_orionis.isna()].reset_index(drop=True), nbins=10, ngroups=10, model_name='MIST', save_path=f'{user_path}/ONC/figures/mass vs separation - MIST.pdf')

print('--------------------Finished--------------------')