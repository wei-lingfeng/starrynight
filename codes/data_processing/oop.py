import numpy as np
import pandas as pd
import copy
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff


def hex_to_rgb(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in range(0, len(value), 2))


class Sources():
    def __init__(self, ra: u.Quantity, dec: u.Quantity, pm_ra_cosdec: u.Quantity, pm_dec: u.Quantity, radial_velocity: u.Quantity) -> None:
        """Initialize data.

        Parameters
        ----------
        ra : u.Quantity
            Right ascension.
        dec : u.Quantity
            Declination.
        pm_ra_cosdec : u.Quantity
            pm ra cosdec.
        pm_dec : u.Quantity
            pm dec.
        rv : u.Quantity
            radial velocity.
        """
        self.coord = SkyCoord(
            ra  = ra,
            dec = dec,
            pm_ra_cosdec = pm_ra_cosdec,
            pm_dec = pm_dec,
            radial_velocity = radial_velocity
        )
        
        self.HC2000 = []
        self.ID_apogee = []
    
    def plot_2d(self, scale=0.0025):
        '''Generate 2D plots of position and velocity.
        - Parameters:
            - sources_coord: astropy coordinates with ra, dec, pm_ra_cosdec, pm_dec.
            - scale: scale of quiver.
        - Returns:
            - fig: figure handle.
        '''
        
        C0 = '#1f77b4'
        C1 = '#ff7f0e'
        C3 = '#d62728'
        C4 = '#9467bd'
        C6 = '#e377c2'
        C7 = '#7f7f7f'
        C9 = '#17becf'

        line_width=2
        opacity=0.8
        marker_size = 6

        nirspec_flag    = np.logical_and(~self.HC2000.isna(), self.ID_apogee.isna())
        apogee_flag     = np.logical_and(self.HC2000.isna(), ~self.ID_apogee.isna())
        matched_flag    = np.logical_and(~self.HC2000.isna(), ~self.ID_apogee.isna())

        trapezium = SkyCoord("05h35m16.26s", "-05d23m16.4s")

        fig_data = [
            # nirspec quiver
            ff.create_quiver(
                self._RAJ2000[nirspec_flag],
                self._DEJ2000[nirspec_flag],
                self.pmRA[nirspec_flag],
                self.pmDE[nirspec_flag],
                name='NIRSPEC Velocity',
                scale=scale,
                line=dict(
                    color='rgba' + str(hex_to_rgb(C0) + (opacity,)),
                    width=line_width
                ),
                opacity=opacity,
                showlegend=False
            ).data[0],
            
            # apogee quiver
            ff.create_quiver(
                self._RAJ2000[apogee_flag],
                self._DEJ2000[apogee_flag],
                self.pmRA[apogee_flag],
                self.pmDE[apogee_flag],
                name='APOGEE Velocity',
                scale=scale,
                line=dict(
                    color='rgba' + str(hex_to_rgb(C4) + (opacity,)),
                    width=line_width
                ),
                opacity=opacity,
                showlegend=False
            ).data[0],
            
            # matched quiver
            ff.create_quiver(
                self._RAJ2000[matched_flag], 
                self._DEJ2000[matched_flag], 
                self.pmRA[matched_flag], 
                self.pmDE[matched_flag], 
                name='Matched Velocity', 
                scale=scale, 
                line=dict(
                    color='rgba' + str(hex_to_rgb(C3) + (opacity,)),
                    width=line_width
                ),
                opacity=opacity,
                showlegend=False
            ).data[0],
            
            
            # nirspec scatter
            go.Scatter(
                name='NIRSPEC Sources',
                x=self._RAJ2000[nirspec_flag], 
                y=self._DEJ2000[nirspec_flag], 
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=C0
                ),
                opacity=opacity
            ),
            
            # apogee scatter
            go.Scatter(
                name='APOGEE Sources',
                x=self._RAJ2000[apogee_flag], 
                y=self._DEJ2000[apogee_flag],  
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=C4
                ),
                opacity=opacity
            ),
            
            # matched scatter
            go.Scatter(
                name='Matched Sources',
                x=self._RAJ2000[matched_flag], 
                y=self._DEJ2000[matched_flag], 
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=C3
                ),
                opacity=opacity
            )
        ]

        fig = go.Figure()
        fig.add_traces(fig_data)
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        )

        fig.update_layout(
            width=700,
            height=700,
            xaxis_title="Right Ascension (degree)",
            yaxis_title="Declination (degree)",
        )

        fig.show()
    
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

def calculate_velocity(sources, dist=389, dist_e=3):
    '''Calculate velocity
    ------------------------------
    - Parameters:
        - sources: pandas DataFrame containing keys: pmRA, pmRA_e, pmDE, pmDE_e, vr, rv_e.
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
    
    vr = sources.vr
    vr_e = sources.vr_e
    
    v = np.sqrt(vt**2 + vr**2)
    v_e = 1/v * np.sqrt((vt * vt_e)**2 + (vr * vr_e)**2)
    
    sources_new = copy.deepcopy(sources)
    sources_new['vRA'] = vRA
    sources_new['vRA_e'] = vRA_e
    sources_new['vDE'] = vDE
    sources_new['vDE_e'] = vDE_e
    sources_new['vt'] = vt
    sources_new['vt_e'] = vt_e
    sources_new['v'] = v
    sources_new['v_e'] = v_e
    return sources_new


def apply_dist_constraint(dist, dist_e, dist_range=15, dist_error_range=15, min_plx_over_e=5):
    '''Apply a distance constraint on sources.
    ----------
    - Parameters:
        - dist: sources distance in pc.
        - dist_e: sources distance error in pc.
        - dist_range: allowed distance range in +/- pc. Default is 15 pc.
        - dist_error_range: allowed distance error in pc. Default is 15 pc.
        - min_plx_over_e: minimum parallax over error. Default is 5, as per https://www.aanda.org/articles/aa/full_html/2021/05/aa39834-20/aa39834-20.html.
    
    - Returns: dist, plx_cut
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
    ax.vlines([center_dist-dist_range, center_dist+dist_range], ymin=0, ymax=max(n), colors='C3', linestyles='dashed', linewidth=1.5)
    ax.set_xlabel('Distance (pc)')
    ax.set_ylabel('Counts')
    plt.show()
    
    print('Distance constraint applied: {} \u00B1 {} pc, maximum uncertainty: {} pc.'.format(center_dist, dist_range, dist_error_range))
    print('Number of matches: {}'.format(sum(dist_constraint)))
    return center_dist, dist_constraint


data = pd.read_csv('/home/l3wei/ONC/Catalogs/synthetic catalog - epoch combined.csv', dtype={'ID_gaia': str})

#################################################
############## Data Pre-processing ##############
#################################################

trapezium_names = ['A', 'B', 'C', 'D', 'E']

# Replace Trapezium stars fitting results with literature values.
for i in range(len(trapezium_names)):
    trapezium_index = data.loc[data.theta_orionis == trapezium_names[i]].index[-1]
    for model in ['Baraffe', 'Feiden_std', 'Feiden_mag', 'MIST', 'Palla']:
        data.loc[trapezium_index, ['mass_{}'.format(model), 'mass_e_{}'.format(model)]] = [data.loc[trapezium_index, 'mass_literature'], data.loc[trapezium_index, 'mass_e_literature']]


# Apply rv constraint.
max_rv = 20
max_rv_e = 5

rv_constraint = ((
    (abs(data.rv_corrected_nirspec) < max_rv) |
    (abs(data.rv_corrected_apogee) < max_rv)
) & (
    (data.rv_e_nirspec < max_rv_e) |
    (data.rv_e_apogee < max_rv_e)
) | (
    ~data.theta_orionis.isna()
))

data = data.loc[rv_constraint].reset_index(drop=True)

rv_use_apogee = (abs(data.rv_corrected_nirspec) > max_rv) & (abs(data.rv_corrected_apogee) <= max_rv)
data.loc[rv_use_apogee, ['rv_corrected_nirspec', 'rv_e_nirspec']] = data.loc[rv_use_apogee, ['rv_corrected_apogee', 'rv_e_apogee']]

# Apply gaia constraint.
gaia_columns = [key for key in data.keys() if (key.endswith('gaia') | key.startswith('plx') | key.startswith('Gmag') | key.startswith('astrometric'))]
gaia_columns.append('ruwe')
# gaia_filter = (data.plx_over_e < 5) | (data.astrometric_excess_noise > 1)
gaia_filter = data.astrometric_excess_noise > 1
data.loc[gaia_filter, gaia_columns] = np.nan

# Compare velocity and fit offset
# compare_velocity(data)
offset = [1.5706144258694983, 1.5556238124586341, 1.58540596409595]

# Correct Kim values
data.pmRA_kim += offset[0]
data.pmRA_e_kim = np.sqrt(data.pmRA_e_kim**2 + ((offset[2] - offset[1])/2)**2)

# merge proper motion and vr
# prioritize kim
data['pmRA'] = merge(data.pmRA_kim, data.pmRA_gaia)
data['pmRA_e'] = merge(data.pmRA_e_kim, data.pmRA_e_gaia)
data['pmDE'] = merge(data.pmDE_kim, data.pmDE_gaia)
data['pmDE_e'] = merge(data.pmDE_e_kim, data.pmDE_e_gaia)
data['vr'], data['vr_e'] = weighted_avrg_and_merge(data.rv_corrected_nirspec, data.rv_corrected_apogee, error1=data.rv_e_nirspec, error2=data.rv_e_apogee)

data['dist'] = 1000/data.plx
data['dist_e'] = data.dist / data.plx * data.plx_e


# Construct 2-D & 3-D Coordinates
# 2-D
data_2d = calculate_velocity(data)
data_2d.plx = 1000/389
data_2d.dist = 389
data_2d.dist_e = 3
data_coord_2d = SkyCoord(
    ra=data_2d._RAJ2000.to_numpy()*u.degree,
    dec=data_2d._DEJ2000.to_numpy()*u.degree, 
    pm_ra_cosdec=data_2d.pmRA.to_numpy()*u.mas/u.yr,
    pm_dec=data_2d.pmDE.to_numpy()*u.mas/u.yr
)
trapezium = SkyCoord("05h35m16.26s", "-05d23m16.4s")
data_2d['sep_to_trapezium'] = data_coord_2d.separation(trapezium).to(u.arcmin).value  # arcmin

# 3-D
center_dist, dist_constraint = apply_dist_constraint(data.dist, data.dist_e, min_plx_over_e=5)
data_3d = data.loc[dist_constraint].reset_index(drop=True)
data_3d = calculate_velocity(data_3d, dist=data_3d.dist, dist_e=data_3d.dist_e)
data_coord_3d = SkyCoord(
    ra=data_3d._RAJ2000.to_numpy()*u.degree, 
    dec=data_3d._DEJ2000.to_numpy()*u.degree, 
    distance=1000/data_3d.plx.to_numpy()*u.pc,
    pm_ra_cosdec=data_3d.pmRA.to_numpy()*u.mas/u.yr,
    pm_dec=data_3d.pmDE.to_numpy()*u.mas/u.yr,
    radial_velocity=data_3d.vr.to_numpy()*u.km/u.s
)




sources = Sources(
    ra=data._RAJ2000.to_numpy() * u.degree,
    dec=data._DEJ2000.to_numpy() * u.degree,
    pm_ra_cosdec=data.pmRA_kim * u.mas/u.yr,
    pm_dec=data.pmDE_kim * u.mas/u.yr,
    radial_velocity=data.vr
)

sources.HC2000 = data.HC2000
sources.ID_apogee = data.ID_apogee
sources._RAJ2000 = data._RAJ2000
sources._DEJ2000 = data._DEJ2000
sources.pmRA = data.pmRA
sources.pmDE = data.pmDE