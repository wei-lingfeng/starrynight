import copy
import collections
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS

# sys.path.insert(0, r'/home/l3wei/ONC')

from Models.BHAC15 import BHAC15_Fit
from Models.Feiden import Feiden_Fit
from Models.MIST import MIST_Fit
from Models.Palla import Palla_Fit
from Models.DAntona import DAntona_Fit


###############################################
################## Functions ##################
###############################################
def weighted_avrg(array, error=None):
    '''
    Calculate weighted avearge of a 1-D array with weights.
    --------------------
    - Parameters:
        - array: array-like 1-D array.
        - weight: array-like 1-D weight. None by default.
    - Returns:
        - result: 1-D array of weighted average
        - result_e: 1-D array of uncertainty. Not returned if errors are not provided.
    '''
    array = np.array([array]).flatten()
    if not (error is None):
        error = np.array([error]).flatten()
        idx = np.logical_and(~np.isnan(array), ~np.isnan(error))   # value in both
        if sum(idx)==0: 
            return np.nan, np.nan
        else:
            avrg = np.average(array[idx], weights=1/error[idx]**2)
            avrg_e = 1/sum(1/error[idx]**2) * np.sqrt(sum(1/error[idx]**2))
            return avrg, avrg_e
    else:
        avrg = np.nanmean(array)
        return avrg


def weighted_avrg_and_merge(array1, array2, error1=None, error2=None):
    '''
    Calculate weighted average of two 1-D arrays and merge into one.
    --------------------
    - Parameters:
        - array1: array-like 1-D values.
        - array2: array-like 1-D values.
        - error1: array-like 1-D errors. Default is None.
        - error2: array-like 1-D errors. Default is None.
    - Returns:
        - result: 1-D array of weighted average
        - result_e: 1-D array of uncertainty. Not returned if errors are not provided.
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


def cross_match(coord1, coord2, max_sep=1*u.arcsec, plot=True, label1='Sources 1', label2='Sources 2'):
    '''Cross match two coordinates.
    Parameters:
        coord1: astropy coordinates.
        coord2: astropy coordinates.
        max_sep: maximum allowed separation. Default is 1 arcsec.
        plot: boolean. Plot cross match results or not.
        label1: legend label of coord1. Default is 'Sources 1'.
        label2: legend label of coord2. Default is 'Sources 2'.
    Returns:
        idx, sep_constraint: coord1[sep_constraint] -> coord2[idx[sep_constraint]]. See https://docs.astropy.org/en/stable/coordinates/matchsep.html#matching-catalogs.
    '''
    
    idx, d2d, d3d = coord1.match_to_catalog_sky(coord2)
    sep_constraint = d2d < max_sep
    
    if plot:
        
        fig_data = [
            go.Scatter(
                mode='markers',
                name=label2,
                x=coord2.ra.deg,
                y=coord2.dec.deg,
                marker=dict(
                    size=3,
                    color='#1f77b4'
                ),
                legendrank=2
            ),
            
            go.Scatter(
                mode='markers',
                name=label1,
                x=coord1.ra.deg,
                y=coord1.dec.deg,
                marker=dict(
                    size=3,
                    color='#d62728'
                ),
                legendrank=1
            )
        ]
        
        ply_shapes = dict()
        for i in np.arange(len(coord1))[sep_constraint]:
            ply_shapes['shape_' + str(i)] = go.layout.Shape(
                type='circle',
                xref='x', yref='y',
                x0=(coord1.ra.deg[i]  + coord2.ra.deg[idx[i]])/2  - (max_sep.to(u.degree)).value/2 / np.cos((coord1.dec.deg[i] + coord2.dec.deg[idx[i]])/2*np.pi/180),
                y0=(coord1.dec.deg[i] + coord2.dec.deg[idx[i]])/2 - (max_sep.to(u.degree)).value/2,
                x1=(coord1.ra.deg[i]  + coord2.ra.deg[idx[i]])/2  + (max_sep.to(u.degree)).value/2 / np.cos((coord1.dec.deg[i] + coord2.dec.deg[idx[i]])/2*np.pi/180),
                y1=(coord1.dec.deg[i] + coord2.dec.deg[idx[i]])/2 + (max_sep.to(u.degree)).value/2,
                line_color='#2f2f2f',
                line_width=1,
                name='Matched Sources'
            )
        
        fig = go.Figure(fig_data)
        
        # Plot matched circles
        fig.update_layout(shapes=list(ply_shapes.values()))
        
        # Final configuration
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1
        )
        
        fig.update_layout(
            width=750,
            height=750,
            xaxis_title="Right Ascension (degree)",
            yaxis_title="Declination (degree)"
        )
        
        fig.show()
    
    print('{} pairs are found when cross matching between {} and {} with a radius of {} {}.'.format(
        sum(sep_constraint), 
        label1, 
        label2, 
        max_sep.value, 
        max_sep.unit.to_string())
    )
    return idx, sep_constraint


def match_and_merge(catalog1, catalog2, mode, max_sep=1.0*u.arcsec, plot=True, **kwargs):
    '''Cross match and merge two catalogs.
    --------------------
    For each star in catalog1, match the closest star in catalog2 that has a separation < max_sep.
    A suffix is added to the keys of catalog2 for duplicate columns. _RAJ2000 and _DEJ2000 are merged, prioritizing catalog1.
    - Parameters: 
        - catalog1: pandas dataframe with _RAJ2000 and _DEJ2000 columns in degrees.
        - catalog2: pandas dataframe with _RAJ2000 and _DEJ2000 columns in degrees.
        - mode: "and" or "or". Determine whether to keep unmatched sources in catalog2 or not.
        - max_sep: maximum separation (1.0 arcsec by default).
        - kwargs: 
            - label1: legend label of catalog1. Default is 'Sources 1'.
            - label2: legend label of catalog2. Default is 'Sources 2'.
            - suffix: a string to be added to the columns of catalog2 after merge if there are duplicate columns. Default is '_{}'.format(label2).
            - columns: list of desired keys in catalog2 that will appear in the result (default: every column in catalog2). Other columns will not be included in the result.
            - matches: user-provided list of matched index. catalog1[i] --> catalog2[matches[i]]. 
                np.nan for unmatched sources. Length is the same as catalog1.
    - Returns: 
        - result: pandas dataframe of cross-matched catalog.
    '''
    
    catalog1_coord = SkyCoord(ra=catalog1._RAJ2000*u.degree, dec=catalog1._DEJ2000*u.degree)
    catalog2_coord = SkyCoord(ra=catalog2._RAJ2000*u.degree, dec=catalog2._DEJ2000*u.degree)
    result = copy.deepcopy(catalog1)
    
    label1 = kwargs.get('label1', 'Sources 1')
    label2 = kwargs.get('label2', 'Sources 2')
    suffix = kwargs.get('suffix', '_{}'.format(label2))
    
    # match the closest among sep < max_sep
    # idx, d2d, d3d = catalog1_coord.match_to_catalog_sky(catalog2_coord)
    # sep_constraint = d2d < max_sep
    
    idx, sep_constraint = cross_match(catalog1_coord, catalog2_coord, max_sep, plot, label1=label1, label2=label2)
    
    matches = kwargs.get('matches', [i if constraint else np.nan for i, constraint in zip(idx, sep_constraint)])
    
    
    columns = kwargs.get('columns', catalog2.keys())
    if '_RAJ2000' not in columns:
        columns.append('_RAJ2000')
    if '_DEJ2000' not in columns:
        columns.append('_DEJ2000')
    
    # record if the column is provided or modified (combined with the suffix) within this function.
    new_column_flag = [True if column in catalog1.keys() else False for column in columns]
    new_columns = [''.join((column, suffix)) if column in catalog1.keys() else column for column in columns]
    
    # merge matched sources
    for flag, column in zip(new_column_flag, new_columns):
        if flag:
            result[column] = list(catalog2[column.replace(suffix, '')].reindex(matches))
        else:
            result[column] = list(catalog2[column].reindex(matches))
        
    
    # merge unmatched sources
    if mode=='or':
        unmatched = dict()
        for column in list(catalog1.keys()) + new_columns:
            unmatched[column] = list()
        
        for i in [_ for _ in range(len(catalog2)) if _ not in idx[sep_constraint]]:
            for column in catalog1.keys():
                unmatched[column].append(np.nan)
            
            for flag, column in zip(new_column_flag, new_columns):
                if flag:
                    unmatched[column].append(catalog2[column.replace(suffix, '')][i])
                else:
                    unmatched[column].append(catalog2[column][i])
        
        unmatched = pd.DataFrame.from_dict(unmatched)
        result = pd.concat((result, unmatched))
    elif mode=='and':
        pass
    else:
        raise ValueError("mode {} not recognized! Have to be 'and' or 'or'.".format(mode))
            
    result = result.reset_index(drop=True)
    # merge coordinates, prioritizing catalog1.
    result._RAJ2000 = result._RAJ2000.fillna(result['_RAJ2000{}'.format(suffix)])
    result._DEJ2000 = result._DEJ2000.fillna(result['_DEJ2000{}'.format(suffix)])
    result.pop('_RAJ2000{}'.format(suffix))
    result.pop('_DEJ2000{}'.format(suffix))
    return result


def cross_match_gaia(a_coord, b_coord, plx, max_sep=1*u.arcsec):
    '''On-sky cross match for ALL candidates within a certain radius.
    User can configure the matches with the help of the printed information for multiple candidates within the maximum allowed separation.
    The closest is chosen by default.
    - Parameters: 
        - a_coord: Coordinates to be matched. astropy SkyCoord.
        - b_coord: Gaia Coordinates.
        - plx: Gaia parallax
        - max_sep: maximum separation. Default is 1 arcsec.
    - Returns: 
        matches: list of closest index in b_coord or np.nan. i.e.: a_coord[i] --> b_coord[matches[i]].
        If there's no match: np.nan.
    - Prints:
        When there are multiple matches within max_sep, print out their distances and parallax.
        The closest match is chosen by defaut. However, users can easily change the match with the help of printed information.
    '''
    
    b_idx = np.arange(0, len(b_coord))
    matches = []
    
    print('Multiple Cross Match Results with Gaia:')
    
    for i, star in enumerate(a_coord):
        
        circle_match = b_idx[star.separation(b_coord) < max_sep]
        
        if len(circle_match) > 1:
            separation = star.separation(b_coord[circle_match])
            circle_match = circle_match[np.argsort(separation)]
            separation.sort()
            matches.append(circle_match[0])
            print('{} --> {}'.format(i, circle_match))
            print('Distances from Earth:\t{} pc'.format(1000/np.array(plx[circle_match])))
            print('Mutual separations:\t{}\n'.format(separation.to(u.arcsec)))
        elif len(circle_match) == 1:
            matches.append(circle_match[0])
        else:
            matches.append(np.nan)
    
    exception = input('Exceptions? y/n:')
    
    while True:    
        if exception=='y':
            index1 = eval(input('Index of the first catalog:'))
            index2 = eval(input('Matching index of the second catalog:'))
            matches[index1] = index2
            exception = input("Any other exceptions? Enter 'y' to input the next exception, or 'n' to continue:")
            continue
        elif exception=='n' or exception=='': 
            break
        else:
            exception = input("Unrecognized input. Please input 'y' or 'n'. Exceptions? y/n:")
            continue
            
    return matches


def merge_multiepoch(sources, **kwargs):
    '''Merge multiepoch observation data. The latest row is kept.
    --------------------
    - Parameters:
        - sources: pandas dataframe containing column 'HC2000' and mcmc fitting results.
        - kwargs: column_pairs: dictionary of the form value --> error. e.g. {'teff': 'teff_e'}.
            If not provided, search for all keyword pairs of the form 'KEYWORDS' --> 'KEYWORDS_e' by default.
    - Returns:
        - sources_epoch_combined: combined pandas dataframe. Each object is unique.
    '''
    if 'column_pairs' not in kwargs.keys():
        column_pairs = {}
        keywords = [keyword for keyword in sources.keys() if '{}_e'.format(keyword) in sources.keys()]
        for keyword in keywords:
            column_pairs[keyword] = '{}_e'.format(keyword)
    
    column_pairs = kwargs.get('column_pairs', column_pairs)
    
    counts = collections.Counter(sources.HC2000)
    multiepoch = {k: v for k, v in counts.items() if ((v > 1) & (str(k).lower()!='nan'))}   # multi epoch objects
    sources_epoch_combined = copy.deepcopy(sources)
    for object in multiepoch.keys():
        indices = sources.index[sources.HC2000 == object].tolist()
        for value_column, error_column in column_pairs.items():
            sources_epoch_combined.loc[indices[-1], value_column], sources_epoch_combined.loc[indices[-1], error_column] = weighted_avrg(
                sources.loc[indices, value_column], error=sources.loc[indices, error_column]
            )
        sources_epoch_combined.loc[indices[-1], 'rv_helio'] = weighted_avrg(
            sources.loc[indices, 'rv_helio'], error=sources.loc[indices, 'rv_e_nirspec']
        )[0]
        sources_epoch_combined = sources_epoch_combined.drop(indices[:-1])
    sources_epoch_combined = sources_epoch_combined.reset_index(drop=True)
    return sources_epoch_combined


def configurable_cross_match(a_coord, b_coord, max_sep=1.0*u.arcsec):
    '''On sky cross match for ALL candidates within a certain radius.
    User can configure the matches with the help of the printed information for multiple candidates within the maximum separation.
    The closest is chosen by default.
    --------------------
    - Parameters: 
        - a_coord: astropy Skycoord.
        - b_coord: astropy Skycoord.
        - max_sep: maximum separation. Default is 1.0 arcsec.
    - Returns: 
        - matches: list of closest index in b_coord or np.nan. e.g.: a_coord[i] --> b_coord[matches[i]].
        If there's no match: np.nan.
    - Prints:
        When there are multiple matches within max_sep, print out their distances and parallax.
        The closest match is chosen by defaut. However, users can easily change the match with the help of printed information.
    '''
    print('Multiple Cross Match Results:')
    
    b_idx = np.arange(len(b_coord))
    matches = []
    
    for i, star in enumerate(a_coord):
        
        circle_match = b_idx[star.separation(b_coord) < max_sep]
        
        if len(circle_match) > 1:
            separation = star.separation(b_coord[circle_match])
            matches.append(circle_match[np.argmin(separation)])
            print('{} --> {}'.format(i, circle_match))
            print('Separations: {}\n'.format((separation**(1/2)).to(u.arcsec)))
        elif len(circle_match) == 1:
            matches.append(circle_match[0])
        else:
            matches.append(np.nan)
    
    return matches


def fit_mass(teff, teff_e):
    '''Fit mass, logg, etc. assuming a 2-Myr age.
    --------------------
    - Parameters:
        - teff: array-like effective temperature in K.
        - teff_e: array-like effective temperature error in K.
    - Returns:
        - result: pandas dataframe.
    '''
    teff = np.vstack((teff, teff_e)).transpose()
    
    result = pd.DataFrame.from_dict({
        **BHAC15_Fit.fit(teff),
        **MIST_Fit.fit(teff),
        **Feiden_Fit.fit(teff),
        **Palla_Fit.fit(teff),
        **DAntona_Fit.fit(teff)
    })
    
    return result


def plot_four_catalogs(catalogs, save_path, save=True, max_sep=1.*u.arcsec, marker_size=3, opacity=1):
    """Plot four catalogs:NIRSPEC sources, APOGEE, Proper Motion by Kim, and Gaia DR3.
    Any matches with Gaia or Kim's proper motion catalog will be circled out.

    Parameters
    ----------
    catalogs : list of pandas.DataFrame
        sources_epoch_combined, apogee, pm, gaia.
    save_path : str
        save path.
    save : bool, optional
        save or not, by default True
    max_sep : astropy.Quantity, optional
        maximum separation of matches, by default 1.*u.arcsec
    marker_size : int, optional
        marker size, by default 3
    opacity : int, optional
        opacity of markers, by default 1
    """
    C0 = '#1f77b4'
    C1 = '#ff7f0e'
    C3 = '#d62728'
    C4 = '#9467bd'
    C6 = '#e377c2'
    C7 = '#7f7f7f'
    C9 = '#17becf'
    
    sources_epoch_combined, apogee, pm, gaia = catalogs
    
    pm_coord = SkyCoord(ra=pm._RAJ2000*u.degree, dec=pm._DEJ2000*u.degree)
    
    matched_circles_idx = np.arange(len(sources_epoch_combined))[~sources_epoch_combined.ID_gaia.isna() | ~sources_epoch_combined.ID_kim.isna()]
    
    ply_shapes = dict()
    
    for i in matched_circles_idx:
        ply_shapes['shape_' + str(i)] = go.layout.Shape(
            type='circle',
            xref='x', yref='y',
            x0=sources_epoch_combined._RAJ2000[i] - (max_sep.to(u.degree)).value / np.cos(sources_epoch_combined._DEJ2000[i]*np.pi/180),
            y0=sources_epoch_combined._DEJ2000[i] - (max_sep.to(u.degree)).value,
            x1=sources_epoch_combined._RAJ2000[i] + (max_sep.to(u.degree)).value / np.cos(sources_epoch_combined._DEJ2000[i]*np.pi/180),
            y1=sources_epoch_combined._DEJ2000[i] + (max_sep.to(u.degree)).value,
            line_color='#2f2f2f',
            line_width=1,
            opacity=opacity,
            name='Matched Sources'
        )
       
    trapezium = SkyCoord("05h35m16.26s", "-05d23m16.4s")
    
    apogee_index = [apogee.index[apogee.ID_apogee == _][0] for _ in sources_epoch_combined.ID_apogee[~sources_epoch_combined.ID_apogee.isna()]]
    
    fig_data = [
        # Plot Trapezium
        go.Scatter(
            mode='markers',
            name='Trapezium',
            x=[trapezium.ra.degree],
            y=[trapezium.dec.degree],
            marker=dict(
                symbol='star',
                size=marker_size*4,
                color=C3
            ),
            legendrank=1
        ),
        
        # Plot Gaia Sources
        go.Scatter(
            mode='markers',
            name='Gaia Sources',
            x=gaia._RAJ2000, 
            y=gaia._DEJ2000, 
            opacity=opacity,
            marker=dict(
                size=marker_size,
                color=C7
            ),
            legendrank=5
        ),
        
        # Plot PM Sources
        go.Scatter(
            mode='markers',
            name='Proper Motion Sources',
            x=pm._RAJ2000[pm_coord.separation(trapezium) < 4*u.arcmin], 
            y=pm._DEJ2000[pm_coord.separation(trapezium) < 4*u.arcmin], 
            opacity=opacity/2,
            marker=dict(
                size=marker_size,
                color=C4
            ),
            legendrank=4
        ),
        
        # Plot APOGEE Sources
        go.Scatter(
            mode='markers',
            name='APOGEE Sources',
            x=apogee._RAJ2000[apogee_index], 
            y=apogee._DEJ2000[apogee_index], 
            opacity=1,
            marker=dict(
                size=marker_size,
                color=C1
            ),
            legendrank=3
        ),
        
        # Plot NIRSPEC Sources
        go.Scatter(
            mode='markers',
            name='NIRSPEC Sources',
            x=sources_epoch_combined._RAJ2000[~sources_epoch_combined.HC2000.isna()], 
            y=sources_epoch_combined._DEJ2000[~sources_epoch_combined.HC2000.isna()], 
            opacity=1,
            marker=dict(
                size=marker_size,
                color=C0
            ),
            legendrank=2
        )
    ]
    
    fig = go.Figure(fig_data)
    
    # Plot matched
    fig.update_layout(shapes=list(ply_shapes.values()))
    
    # Final configuration
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1
    )
    
    fig.update_layout(
        width=750,
        height=750,
        xaxis_title="Right Ascension (degree)",
        yaxis_title="Declination (degree)"
    )
    
    fig.show()
    if save:
        fig.write_html(save_path + "/synthetic skymap.html")



###############################################
######### Construct Synthetic Catalog #########
###############################################

def construct_synthetic_catalog(nirspec_path, chris_table_path, apogee_path, kounkel_path, pm_path, gaia_path, hillenbrand_path, tobin_path, save_path):
    '''Construct synthetic catalog combining nirspec, apogee, pm, gaia, and ONC Mass catalog by Hillenbrand 1997 (J/AJ/113/1733/ONC).
    --------------------
    Both a multi-epoch combined & not combined catalog are saved under the save_path folder.
    - Parameters:
        - nirspec_path: path of nirspec catalog csv.
        - chris_table_path: path of Chris's table.
        - apogee_path: path of apogee catalog csv.
        - kounkel_path: path of Kounkel et al. 2018.
        - pm_path: path of pm catalog csv.
        - gaia_path: path of gaia catalog csv.
        - hillenbrand_path: path of ONC mass catalog.
        - tobin_path: path of Tobin et al. 2009.
        - save_path: folder path to save the two catalogs.
    - Returns:
        - sources: pandas dataframe of the synthetic catalog.
        - sources_epoch_combined: pandas dataframe of the multi-epoch combined synthetic catalog.
    - Saves:
        - synthetic catalog.csv: synthetic catalog
        - synthetic catalog - epoch combined.csv: multi-epoch combined catalog.
        - synthetic skymap.html: interactive skymap of all catalogs.
    '''
    trapezium = SkyCoord("05h35m16.26s", "-05d23m16.4s")
    
    nirspec     = pd.read_csv(nirspec_path)
    chris_table = pd.read_csv(chris_table_path)
    apogee      = pd.read_csv(apogee_path)
    kounkel     = pd.read_csv(kounkel_path)
    pm          = pd.read_csv(pm_path)
    gaia        = pd.read_csv(gaia_path, dtype={'source_id': str})
    hillenbrand = pd.read_csv(hillenbrand_path)
    tobin       = pd.read_csv(tobin_path)
    
    
    # Add & Replace Trapezium Stars
    trapezium_stars = {}

    # A1-A3, B1-B5, C1-C2, D, EA-EB
    trapezium_stars['HC2000'] = ['336', '354', '309', '330', '344']
    trapezium_stars['theta_orionis'] = ['A', 'B', 'C', 'D', 'E']
    trapezium_stars['mass_literature'] = [26.6, 16.3, 44, 25, 5.604]
    trapezium_stars['mass_e_literature'] = [26.6*0.05, 16.3*0.05, 5*2**0.5, 25*0.05, 0.048*2**0.5]
    
    trapezium_stars = pd.DataFrame.from_dict(trapezium_stars)
    
    trapezium_stars.insert(1, '_RAJ2000', np.nan)
    trapezium_stars.insert(2, '_DEJ2000', np.nan)
    
    # Get RA DEC from nirspec catalog
    hc2000 = pd.read_csv('/home/l3wei/ONC/Catalogs/HC2000.csv', dtype={'[HC2000]': str})
    for i in range(len(trapezium_stars)):
        index = list(hc2000['[HC2000]']).index(trapezium_stars.loc[i, 'HC2000'])
        coord = SkyCoord(' '.join((hc2000['RAJ2000'][index], hc2000['DEJ2000'][index])), unit=(u.hourangle, u.deg))
        trapezium_stars.loc[i, '_RAJ2000'] = coord.ra.degree
        trapezium_stars.loc[i, '_DEJ2000'] = coord.dec.degree
    
    # Remove trapezium stars from nirspec
    nirspec_hc2000 = [_.split('_')[0] for _ in nirspec.HC2000]
    remove_list = []
    for i in range(len(trapezium_stars)):
        if trapezium_stars.loc[i, 'HC2000'] in nirspec_hc2000:
            remove_list.extend([index for index, element in enumerate(nirspec_hc2000) if element == trapezium_stars.loc[i, 'HC2000']])
    
    nirspec = nirspec.drop(remove_list).reset_index(drop=True)
            
    # Merge nirspec and trapezium stars
    nirspec = pd.concat([nirspec, trapezium_stars], ignore_index=True, sort=False)
    
    theta_orionis = nirspec.theta_orionis
    nirspec = nirspec.drop(columns='theta_orionis')
    nirspec.insert(1, 'theta_orionis', theta_orionis)
    
    nirspec = nirspec.rename(columns={
        'teff':     'teff_nirspec',
        'teff_e':   'teff_e_nirspec',
        'rv':       'rv_nirspec',
        'rv_e':     'rv_e_nirspec'
    })
    
    nirspec.insert(12, 'teff', np.nan)
    nirspec.insert(13, 'teff_e', np.nan)
    
    chris_table = chris_table.rename(columns={
        'teff': 'teff_chris',
        'teff_e': 'teff_e_chris',
        'vr': 'vr_chris',
        'vr_e': 'vr_e_chris',
        'vsini': 'vsini_chris',
        'vsini_e': 'vsini_e_chris',
        'veiling_param_O33': 'veiling_param_O33_chris'
    })
    
    # merge with chris table
    chris_table_binary_idx = np.where([_.endswith(('A', 'B')) for _ in chris_table.HC2000])[0]
    chris_table_binary_hc2000 = set(_.strip('A|B') for _ in chris_table.HC2000[chris_table_binary_idx])
    for hc2000 in chris_table_binary_hc2000:
        # if a target has both A and B, change both suffix to _A and _B.
        if ((hc2000 + 'A') in list(chris_table.HC2000)) and ((hc2000 + 'B') in list(chris_table.HC2000)):
            chris_table.loc[chris_table.HC2000==hc2000 + 'A', 'HC2000'] = hc2000 + '_A'
            chris_table.loc[chris_table.HC2000==hc2000 + 'B', 'HC2000'] = hc2000 + '_B'
        # else: remove the suffix
        else:
            chris_table.loc[chris_table.HC2000==hc2000 + 'A', 'HC2000'] = hc2000
            chris_table.loc[chris_table.HC2000==hc2000 + 'B', 'HC2000'] = hc2000
    
    nirspec = nirspec.merge(chris_table[[
        'HC2000', 
        'teff_chris', 'teff_e_chris', 
        'vr_chris', 'vr_e_chris', 
        'vsini_chris', 'vsini_e_chris', 
        'veiling_param_O33_chris'
    ]], how='left', on='HC2000')
    
    apogee = apogee.rename(columns={
        'APOGEE_ID': 'ID_apogee',
        'rv':   'rv_apogee',
        'rv_e': 'rv_e_apogee'
    })
    
    kounkel = kounkel.rename(columns={
        'RAJ2000':  '_RAJ2000',
        'DEJ2000':  '_DEJ2000',
        '_2MASS':   'ID_2MASS',
        'Gaia':     'ID_gaia_dr2',
        'Teff':     'teff_kounkel',
        'e_Teff':   'teff_e_kounkel',
        'logg':     'logg_kounkel',
        'e_logg':   'logg_e_kounkel',
        'RVmean':   'rv_mean_kounkel',
        'e_RVmean': 'rv_mean_e_kounkel'
    })
    
    pm = pm.rename(columns={
        'ID':       'ID_kim',
        'pmRA':     'pmRA_kim',
        'e_pmRA':   'pmRA_e_kim',
        'pmDE':     'pmDE_kim',
        'e_pmDE':   'pmDE_e_kim'
    })
    
    gaia = gaia.rename(columns={
        'source_id':        'ID_gaia',
        'ra':               '_RAJ2000',
        'dec':              '_DEJ2000',
        'parallax':         'plx',
        'parallax_error':   'plx_e',
        'parallax_over_error': 'plx_over_e',
        'pmra':             'pmRA_gaia',
        'pmra_error':       'pmRA_e_gaia',
        'pmdec':            'pmDE_gaia',
        'pmdec_error':      'pmDE_e_gaia',
        'phot_g_mean_mag':  'Gmag'
    })
    
    # Add Gmag uncertainty.
    # magnitude = -2.5*log10(flux) + zero point. (https://en.wikipedia.org/wiki/Apparent_magnitude#Calculations)
    # zero point: 25.6874 ± 0.0028. (https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photProc/cu5pho_ssec_photCal.html#SSS3.P2)
    gaia.insert(
        loc=gaia.columns.get_loc('Gmag') + 1, 
        column='Gmag_e', 
        value=((2.5/np.log(10)*gaia['phot_g_mean_flux_error']/gaia['phot_g_mean_flux'])**2 + 0.0027553202**2)**(1/2)
    )
    
    hillenbrand = hillenbrand.rename(columns={'RAJ2000': '_RAJ2000', 'DEJ2000': '_DEJ2000', 'ID': 'ID_hillenbrand', 'M': 'mass_Hillenbrand'})
    # Offset specified in https://iopscience.iop.org/article/10.1086/309309/pdf, Section 4.5.
    hillenbrand_coord = SkyCoord(ra=(hillenbrand._RAJ2000 + 1.5/3600)*u.deg, dec=(hillenbrand._DEJ2000 - 0.3/3600)*u.deg)
    hillenbrand._RAJ2000 = hillenbrand_coord.ra.deg
    hillenbrand._DEJ2000 = hillenbrand_coord.dec.deg
    hillenbrand = hillenbrand.loc[hillenbrand_coord.separation(trapezium) <= 4*u.arcmin].reset_index(drop=True)
    hillenbrand = hillenbrand.replace(r' ', np.nan)
    
    tobin = tobin.rename(columns={
        '_2MASS': 'ID_2MASS_tobin',
        'HRV':  'rv_tobin',
        'e_HRV':    'rv_e_tobin',
        'Vmag': 'Vmag_tobin',
        'V-I':  'V-I_tobin'
    })
    
    ###############################################
    ################# Cross Match #################
    ###############################################
    
    # cross match apogee
    print('length of apogee: {}'.format(len(apogee)))
    sources = match_and_merge(nirspec, apogee, mode='or', plot=False, suffix='_apogee', columns=[
        'ID_apogee', 
        'rv_apogee',       'rv_e_apogee', 
        'teff',     'teff_e', 
        'vsini',    'vsini_e', 
        'veiling', 
        'Kmag',     'Kmag_e',
        'Hmag', 'Hmag_e'
    ])
    
    # cross match kounkel
    sources = match_and_merge(sources, kounkel, mode='and', plot=False, suffix='_kounkel', columns=[
        'ID_2MASS', 'ID_gaia_dr2',
        'teff_kounkel', 'teff_e_kounkel',
        'logg_kounkel', 'logg_e_kounkel',
        'rv_mean_kounkel', 'rv_mean_e_kounkel'
    ])
    
    # cross match proper motion
    sources = match_and_merge(sources, pm, mode='and', plot=False, suffix='_kim', columns=[
        'ID_kim',
        'pmRA_kim', 'pmRA_e_kim',
        'pmDE_kim', 'pmDE_e_kim'
    ])
    
    sources_coord = SkyCoord(ra=sources._RAJ2000*u.degree, dec=sources._DEJ2000*u.degree)
    gaia_coord = SkyCoord(ra=gaia._RAJ2000*u.degree, dec=gaia._DEJ2000*u.degree)
        
    max_sep = 1 * u.arcsec
    matches = cross_match_gaia(sources_coord, gaia_coord, gaia.plx, max_sep=max_sep)
    
    # cross match gaia
    sources = match_and_merge(sources, gaia, mode='and', plot=False, suffix='_gaia', matches=matches, columns=[
        'ID_gaia',
        'plx', 'plx_e',
        'plx_over_e',
        'pmRA_gaia', 'pmRA_e_gaia',
        'pmDE_gaia', 'pmDE_e_gaia',
        'Gmag', 'Gmag_e', 'bp_rp',
        'astrometric_n_good_obs_al', 'astrometric_gof_al', 'astrometric_excess_noise', 'ruwe'
    ])
    
    # cross match hillenbrand
    sources = match_and_merge(sources, hillenbrand, mode='and', plot=True, suffix='_hillenbrand', max_sep=2*u.arcsec, label1='NIRSPEC+APOGEE', label2='Hillenbrand')
    
    # cross match tobin
    sources = match_and_merge(sources, tobin, mode='and', plot=False, suffix='_tobin', columns=[
        'ID_2MASS_tobin',
        'rv_tobin', 'rv_e_tobin', 'Vmag_tobin', 'V-I_tobin'
    ])
    

    ###############################################
    ############# Calculate Median RV #############
    ###############################################
    # sources_epoch_combined = merge_multiepoch(sources)
    
    # rvs = weighted_avrg_and_merge(
    #     sources_epoch_combined.rv_helio,
    #     sources_epoch_combined.rv_apogee,
    #     sources_epoch_combined.rv_e_nirspec,
    #     sources_epoch_combined.rv_e_apogee
    # )[0]
    
    # median_rv = np.nanmedian(rvs)
    # with open(save_path + 'median rv.txt', 'w') as file:
    #     file.write(str(median_rv))
        
    # sources['rv_corrected_nirspec'] = sources.rv_helio - median_rv
    # sources['rv_corrected_apogee'] = sources.rv_apogee - median_rv
    
    
    ###############################################
    ########### Merge Duplicate Columns ###########
    ###############################################
    
    
    sources.teff = sources.teff_nirspec.fillna(sources.teff_apogee)
    sources.teff_e = sources.teff_e_nirspec.fillna(sources.teff_e_apogee)
    
    sources.Kmag, sources.Kmag_e = weighted_avrg_and_merge(sources.Kmag, sources.Kmag_apogee, error1=sources.Kmag_e, error2=sources.Kmag_e_apogee)
    for key in ['Kmag_apogee', 'Kmag_e_apogee']:
        sources.pop(key)
    
    apogee_id = sources.ID_apogee
    sources.pop('ID_apogee')
    sources.insert(1, 'ID_apogee', apogee_id)
    
    sources_epoch_combined = merge_multiepoch(sources)
    
    ###############################################
    ################### Fit Mass ##################
    ###############################################
    sources = pd.concat([sources, fit_mass(sources.teff, sources.teff_e)], axis=1)
    sources_epoch_combined = pd.concat([sources_epoch_combined, fit_mass(sources_epoch_combined.teff, sources_epoch_combined.teff_e)], axis=1)
    
    # Delete Fitting results for trapezium stars
    columns = fit_mass(sources.teff, sources.teff_e).keys()
    indices = ~sources.theta_orionis.isna()
    sources.loc[indices, columns] = np.nan
    
    indices = ~sources_epoch_combined.theta_orionis.isna()
    sources_epoch_combined.loc[indices, columns] = np.nan
    
    
    ###############################################
    ################# Write to csv ################
    ###############################################
    sources.to_csv(save_path + 'synthetic catalog.csv', index=False)
    sources_epoch_combined.to_csv(save_path + 'synthetic catalog - epoch combined.csv', index=False)
    
    ###############################################
    ##################### Plot ####################
    ###############################################
    
    plot_four_catalogs([sources_epoch_combined, apogee, pm, gaia], save_path=save_path)
    return sources, sources_epoch_combined





if __name__=='__main__':
    nirspec_path        = '/home/l3wei/ONC/Catalogs/nirspec sources.csv'
    chris_table_path    = '/home/l3wei/ONC/Catalogs/Chris\'s Table.csv'
    apogee_path         = '/home/l3wei/ONC/Catalogs/apogee x 2mass.csv'
    kounkel_path        = '/home/l3wei/ONC/Catalogs/kounkel 2018.csv'
    pm_path             = '/home/l3wei/ONC/Catalogs/proper motion.csv'
    gaia_path           = '/home/l3wei/ONC/Catalogs/gaia dr3.csv'
    hillenbrand_path    = '/home/l3wei/ONC/Catalogs/hillenbrand mass.csv'
    tobin_path          = '/home/l3wei/ONC/Catalogs/tobin 2009.csv'
    save_path           = '/home/l3wei/ONC/Catalogs/'

    sources, sources_epoch_combined = construct_synthetic_catalog(nirspec_path, chris_table_path, apogee_path, kounkel_path, pm_path, gaia_path, hillenbrand_path, tobin_path, save_path)