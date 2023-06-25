import os
import copy
import collections
import numpy as np
import numpy.ma as ma
import pandas as pd
import plotly.graph_objects as go
import astropy.units as u
from astropy.io import ascii
from astropy.table import QTable, MaskedColumn, hstack, vstack
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
Vizier.ROW_LIMIT = -1
Gaia.ROW_LIMIT = -1
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

from starrynight.models.BHAC15 import BHAC15_Fit
from starrynight.models.Feiden import Feiden_Fit
from starrynight.models.MIST import MIST_Fit
from starrynight.models.Palla import Palla_Fit
from starrynight.models.DAntona import DAntona_Fit

user_path = os.path.expanduser('~')
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
    unit = None
    if 'astropy' in str(type(array)):
        unit = array.unit
        array = ma.MaskedArray(array.value)
        if error is not None:
            error = ma.MaskedArray(error.value)
    else:
        array = np.array([array]).flatten()
    
    if error is not None:
        if all(np.isnan(array)) and all(np.isnan(error)):
            return np.nan, np.nan
        else:
            avrg = ma.average(array, weights=1/error**2)
            avrg_e = ma.sqrt(ma.sum(error**2))/ma.sum(~np.isnan(error))
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
    unit = None
    if 'astropy' in str(type(array1)):
        unit = array1.unit
        array1 = ma.MaskedArray(array1.value)
        array2 = ma.MaskedArray(array2.value)
        if error1 is not None and error2 is not None:
            error1 = ma.MaskedArray(error1.value).filled(np.nan)
            error2 = ma.MaskedArray(error2.value).filled(np.nan)
    else:
        array1 = np.array([array1]).flatten()
        array2 = np.array([array2]).flatten()
        error1 = np.array([error1]).flatten()
        error2 = np.array([error2]).flatten()

    
    if error1 is not None and error2 is not None:
        N_stars = len(array1)
        avrg_e = ma.empty(N_stars)
        value_in_1 = np.logical_and(~array1.mask,  array2.mask)   # value in 1 ONLY
        value_in_2 = np.logical_and( array1.mask, ~array2.mask)   # value in 2 ONLY
        value_both = np.logical_and(~array1.mask, ~array2.mask)   # value in both
        value_none = np.logical_and( array1.mask,  array2.mask)   # value in none
        
        avrg = ma.average([array1, array2], axis=0, weights=[1/error1**2, 1/error2**2])
        avrg_e[value_in_1] = error1[value_in_1]
        avrg_e[value_in_2] = error2[value_in_2]
        avrg_e[value_both] = 1 / np.sqrt(1/error1[value_both]**2 + 1/error2[value_both]**2)
        avrg_e[value_none] = np.nan
        
        if unit is not None:
            avrg = MaskedColumn(avrg, mask=value_none, unit=unit)
            avrg_e = MaskedColumn(avrg_e, mask=value_none, unit=unit)
        return avrg, avrg_e
    
    else:
        avrg = ma.average([array1, array2], axis=0)
        if unit is not None:
            avrg = MaskedColumn(avrg, unit=unit)
        return avrg


def cross_match(coord1, coord2, max_sep=1*u.arcsec, plot=True, label_names = ['1', '2']):
    '''Cross match two coordinates.
    
    
    Parameters
    ----------
        coord1 : astropy SkyCoord
        coord2 : astropy SkyCoord
        max_sep : astropy quantity
            Maximum allowed separation, by default 1*u.arcsec.
        plot : boolean.
            Plot cross match results or not.
        label_names : list or None
            Legend labels, by default ['1', '2'].
    
    Returns
    -------
        idx, sep_constraint: coord1[sep_constraint] <--> coord2[idx[sep_constraint]]. See https://docs.astropy.org/en/stable/coordinates/matchsep.html#matching-catalogs.
    '''
    
    idx, d2d, d3d = coord1.match_to_catalog_sky(coord2)
    sep_constraint = d2d < max_sep
    
    if plot:
        
        fig_data = [
            go.Scatter(
                mode='markers',
                name=label_names[1],
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
                name=label_names[0],
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
    
    print('{} pairs are found when cross matching between table {} and table {} with a radius of {} {}.'.format(
        sum(sep_constraint), 
        label_names[0], 
        label_names[0], 
        max_sep.value, 
        max_sep.unit.to_string())
    )
    return idx, sep_constraint


def merge_on_coords(catalog1, catalog2, join_type='left', max_sep=1.*u.arcsec, coord_cols=['RAJ2000', 'DEJ2000'], matches=None, left_coord_cols=None, right_coord_cols=None, merge_coords=True, plot=True, table_names=['1', '2'], uniq_col_name='{col_name}_{table_name}'):
    """Merge tables on coordinates

    Parameters
    ----------
    catalog1 : astropy table
    catalog2 : astropy table
    join_type : str, optional
        Join type, one of 'inner', 'outer', 'left', or 'right', by default 'left'
    max_sep : astropy quantity, optional
        Maximum allowed separation when cross matching, by default 1.*u.arcsec
    coord_cols : list of str, optional
        Column names with RA and DEC, by default ['RAJ2000', 'DEJ2000']
    matches : array-like, optional
        catalog1[i] <--> catalog2[matches[i]]. If matches[i]=NaN, there is no match for catalog1[i]. If not provided, the function will automatically calculate based on keys or coordinates, by default None
    left_coord_cols : list of str, optional
        Column names with RA and DEC for the left table, by default None
    right_coord_cols : list of str, optional
        Column names with RA and DEC for the right table, by default None
    merge_coords : bool, optional
        Merge the RA and DEC columns or not, by default True
    plot : bool, optional
        Plot the cross match by plotly or not, by default True
    table_names : list of str, optional
        Two-element list of table names used when generating unique output column names, by default ['1', '2']. See https://docs.astropy.org/en/stable/api/astropy.table.join.html
    uniq_col_name : str, optional
        String generate a unique output column name in case of a conflict, by default '{col_name}_{table_name}'. See https://docs.astropy.org/en/stable/api/astropy.table.join.html
    
    Returns
    -------
    astropy table
        Astropy table merged on coordinates
    """
    
    # match the closest among sep < max_sep
    # idx, d2d, d3d = catalog1_coord.match_to_catalog_sky(catalog2_coord)
    # sep_constraint = d2d < max_sep
    if left_coord_cols is None and right_coord_cols is None:
        left_coord_cols = coord_cols
        right_coord_cols = coord_cols
    catalog1_coord = SkyCoord(ra=catalog1[left_coord_cols[0]], dec=catalog1[left_coord_cols[1]])
    catalog2_coord = SkyCoord(ra=catalog2[right_coord_cols[0]], dec=catalog2[right_coord_cols[1]])
    idx, sep_constraint = cross_match(catalog1_coord, catalog2_coord, max_sep, plot, label_names=table_names)
    matches = np.array([i if valid_match else np.nan for i, valid_match in zip(idx, sep_constraint)])
    
    result = merge(catalog1, catalog2, matches, join_type, table_names, uniq_col_name)
    
    # If merge_coords, remove duplicate RA and DEC columns
    left_coord_cols_new = [uniq_col_name.format(col_name=left_col, table_name=table_names[0]) if table_names[0]!='' and table_names[0] is not None else left_col for left_col in left_coord_cols]
    right_coord_cols_new = [uniq_col_name.format(col_name=right_col, table_name=table_names[1]) if table_names[1]!='' and table_names[1] is not None else right_col for right_col in right_coord_cols]

    if merge_coords:
        if any(result[left_coord_cols_new[0]].mask):
            if join_type=='right':
                left_coord_cols, right_coord_cols = right_coord_cols, left_coord_cols
                table_names = table_names[::-1]
            
            for left_col_new, right_col_new in zip(left_coord_cols_new, right_coord_cols_new):
                result[left_col_new][result[left_col_new].mask] = result[right_col_new][result[left_col_new].mask]
        result.rename_columns(left_coord_cols_new, left_coord_cols)
        result.remove_columns(right_coord_cols_new)
    return result
    

def merge_on_keys(catalog1, catalog2, keys, keys_left=None, keys_right=None, join_type='left', table_names=['1', '2'], uniq_col_name='{col_name}_{table_name}'):
    """Merge tables on keys

    Parameters
    ----------
    catalog1 : astropy table
    catalog2 : astropy table
    keys : str or list of str, optional
        Name(s) of column(s) used to match rows of left and right tables.
    keys_left : str or list of str, optional
        Left column(s) used to match rows instead of keys arg. This can be be a single left table column name or list of column names, by default None
    keys_left : str or list of str, optional
        Right column(s) used to match rows instead of keys arg. This can be be a single left table column name or list of column names, by default None
    join_type : str, optional
        Join type, one of 'inner', 'outer', 'left', or 'right', by default 'left'
    table_names : list of str, optional
        Two-element list of table names used when generating unique output column names, by default ['1', '2']. See https://docs.astropy.org/en/stable/api/astropy.table.join.html
    uniq_col_name : str, optional
        String generate a unique output column name in case of a conflict, by default '{col_name}_{table_name}'. See https://docs.astropy.org/en/stable/api/astropy.table.join.html
    
    Returns
    -------
    astropy table
        Astropy table merged on keys
    """
    if isinstance(keys, str):
        keys = [keys]
    if keys is not None or (keys_left is not None and keys_right is not None):
        keys_left = keys
        keys_right = keys
        matches = np.empty(len(catalog1))
        for i in range(len(catalog1)):
            matched = [True if list(catalog1[keys][i]) == list(catalog2[keys][j]) else False for j in range(len(catalog2))]
            if any(matched):
                matches[i] = np.where(matched)[0][0]
            else:
                matches[i] = np.nan
    
    return merge(catalog1, catalog2, matches, join_type, table_names, uniq_col_name)

    
def merge(catalog1, catalog2, matches, join_type='left', table_names=['1', '2'], uniq_col_name='{col_name}_{table_name}'):
    """Cross match and merge two catalogs.

    Parameters
    ----------
    catalog1 : astropy table
    catalog2 : astropy table
    matches : array-like, optional
        catalog1[i] <--> catalog2[matches[i]]. If matches[i]=NaN, there is no match for catalog1[i]. If not provided, the function will automatically calculate based on keys or coordinates. None by default.
    join_type : str
        Specifies the join type: inner, outter, left, or right, by default left.
    table_names : list of str, optional
        Two-element list of table names used when generating unique output column names, by default ['1', '2']. See https://docs.astropy.org/en/stable/api/astropy.table.join.html.
    uniq_col_names : str, optional
        String generate a unique output column name in case of a conflict. The default is '{col_name}_{table_name}'. https://docs.astropy.org/en/stable/api/astropy.table.join.html.
    
    Returns
    -------
    astropy table
        Merged astropy table

    Raises
    ------
    ValueError
        A value error will be raised if join_type is not one of 'inner', 'outter', 'left', or 'right'.
    """
    catalog1 = catalog1.copy()
    catalog2 = catalog2.copy()
    if join_type not in ['inner', 'outer', 'left', 'right']:
        raise ValueError(f"join_type must be 'inner', 'outer', 'left', or 'right', not '{join_type}'.")

    if join_type=='right':
        catalog1, catalog2 = catalog2, catalog1
        table_names = table_names[::-1]

    valid_match = ~np.isnan(matches)

    uniq_columns_catalog1 = [uniq_col_name.format(col_name=col, table_name=table_names[0]) if col in catalog2.keys() else col for col in catalog1.keys()] if table_names[0]!='' and table_names[0] is not None else catalog1.keys()
    uniq_columns_catalog2 = [uniq_col_name.format(col_name=col, table_name=table_names[1]) if col in catalog1.keys() else col for col in catalog2.keys()] if table_names[1]!='' and table_names[1] is not None else catalog2.keys()
    catalog1.rename_columns(catalog1.keys(), uniq_columns_catalog1)
    catalog2.rename_columns(catalog2.keys(), uniq_columns_catalog2)

    # outer join matched sources
    result = catalog1.copy()
    # hstack matched
    for col in uniq_columns_catalog2:
        # add empty column
        if np.issubdtype(catalog2[col].dtype, np.integer):
            result[col] = MaskedColumn([0]*len(result), dtype=catalog2[col].dtype, mask=[True]*len(result))
        else:
            result[col] = MaskedColumn([None]*len(result), dtype=catalog2[col].dtype, mask=[True]*len(result))
        # fill empty column with catalog2
        result[col][valid_match] = catalog2[col][matches[valid_match].astype(int)]

    if join_type=='inner':
        result = result[valid_match]

    # vstack unmatched
    if join_type=='outer':
        result = vstack((result, catalog2[list(set(range(len(catalog2))) - set(matches[valid_match]))]))
    
    for col in uniq_columns_catalog2:
        result[col] = MaskedColumn(result[col], unit=catalog2[col].unit)
    
    return result


def fillna(column1, column2):
    result = column1.copy()
    result[result.mask] = column2[result.mask]
    return result

def cross_match_gaia(coord, gaia_coord, plx, max_sep=1*u.arcsec):
    """On-sky cross match for ALL candidates within a certain radius.
    User can configure the matches with the help of the printed information for multiple candidates within the maximum allowed separation.
    The closest is chosen by default.
    
    Parameters
    ----------
        coord : astropy SkyCoord
            Coordinates to be matched
        gaia_coord : astropy SkyCoord
            Gaia coordinates
        plx : astropy quantity
            Gaia parallax
        max_sep : astropy quantity
            Maximum separation for cross-matching, by default 1*u.arcsec
    
    Returns
    -------
        matches : array
            array of closest index in b_coord or np.nan if there is no match. i.e.: a_coord[i] --> b_coord[matches[i]].
    
    Prints
    ------
        When there are multiple matches within max_sep, print out their distances and parallax.
        The closest match is chosen by defaut. However, users can easily change the match with the help of printed information.
    """
    
    gaia_idx = np.arange(0, len(gaia_coord))
    matches = np.empty(len(coord))
    
    print('Multiple Cross Match Results with Gaia:')
    
    for i, star in enumerate(coord):
        
        circle_match = gaia_idx[star.separation(gaia_coord) < max_sep]
        
        if len(circle_match) > 1:
            separation = star.separation(gaia_coord[circle_match])
            circle_match = circle_match[np.argsort(separation)]
            separation.sort()
            matches[i] = circle_match[0]
            print('{} --> {}'.format(i, circle_match))
            print('Distances from Earth:\t{} pc'.format(1000/np.array(plx[circle_match])))
            print('Mutual separations:\t{}\n'.format(separation.to(u.arcsec)))
        elif len(circle_match) == 1:
            matches[i] = circle_match[0]
        else:
            matches[i] = np.nan
    
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


def merge_multiepoch(sources, column_pairs=None):
    '''Merge multiepoch observation data. The latest row is kept.
    
    Parameters
    ----------
        sources : pandas dataframe or astropy table
            Table containing column 'HC2000' and mcmc fitting results.
        column_pairs : dictionary 
            Column pairs of the form value --> error. e.g. {'teff': 'e_teff'}. If not provided, search for all keyword pairs of the form 'KEYWORDS' --> 'e_KEYWORDS' by default.
    
    Returns
    -------
        sources_epoch_combined: combined pandas dataframe. Each object is unique.
    '''
    units = None
    if 'astropy.table' in str(type(sources)):
        units = [sources[key].unit for key in sources.keys()]
    sources = sources.to_pandas()
    
    if column_pairs is None:
        column_pairs = {}
        keywords = [keyword for keyword in sources.keys() if 'e_{}'.format(keyword) in sources.keys()]
        for keyword in keywords:
            column_pairs[keyword] = 'e_{}'.format(keyword)
    
    ID_HC2000 = [str(l) + '_' + str(r) if str(r)!='nan' else str(l) for l, r in zip(sources.HC2000[~sources.HC2000.isna()], sources.m_HC2000[~sources.HC2000.isna()])]
    counts = collections.Counter(ID_HC2000)
    multiepoch = {k: v for k, v in counts.items() if ((v > 1) & (str(k).lower()!='nan'))}   # multi epoch objects
    sources_epoch_combined = copy.deepcopy(sources)
    for object in multiepoch.keys():
        if '_' in object:
            indices = sources.index[sources.HC2000==int(object.split('_')[0]) & sources.m_HC2000==object.split('_')[1]].tolist()
        else:
            indices = sources.index[sources.HC2000==int(object)].tolist()
        for value_column, error_column in column_pairs.items():
            sources_epoch_combined.loc[indices[-1], value_column], sources_epoch_combined.loc[indices[-1], error_column] = weighted_avrg(
                sources.loc[indices, value_column], error=sources.loc[indices, error_column]
            )
        sources_epoch_combined.loc[indices[-1], 'rv_helio'] = weighted_avrg(
            sources.loc[indices, 'rv_helio'], error=sources.loc[indices, 'e_rv_nirspao']
        )[0]
        sources_epoch_combined = sources_epoch_combined.drop(indices[:-1])
    sources_epoch_combined = sources_epoch_combined.reset_index(drop=True)
    
    if units is not None:
        mapping = {k:u for k, u in zip(sources.keys(), units)}
        sources_epoch_combined = QTable.from_pandas(sources_epoch_combined, units=mapping)
    return sources_epoch_combined


def configurable_cross_match(a_coord, b_coord, max_sep=1.0*u.arcsec):
    """On sky cross match for ALL candidates within a certain radius.
    User can configure the matches with the help of the printed information for multiple candidates within the maximum separation.
    The closest is chosen by default.
    
    Parameters
    ----------
        a_coord: astropy Skycoord
        b_coord: astropy Skycoord
        max_sep: maximum separation, by default 1.0*arcsec.
    
    Returns
    -------
        matches: list
            List of closest index in b_coord from a_coord[i] or np.nan if there is no match. e.g.: a_coord[i] --> b_coord[matches[i]].
    
    Outputs
    -------
        When there are multiple matches within max_sep, print out their distances and parallax.
        The closest match is chosen by defaut. However, users can easily change the match with the help of printed information.
    """
    
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


def fit_mass(teff, e_teff):
    '''Fit mass, logg, etc. assuming a 2-Myr age.
    
    Parameter
    ---------
        teff: array-like
            Effective temperature in K
        teff_e: array-like effective temperature error in K.
    - Returns:
        - result: pandas dataframe.
    '''
    teff = teff.value
    e_teff = e_teff.value
    teff = np.vstack((teff, e_teff)).transpose()
    
    result = pd.DataFrame.from_dict({
        **BHAC15_Fit.fit(teff),
        **MIST_Fit.fit(teff),
        **Feiden_Fit.fit(teff),
        **Palla_Fit.fit(teff),
        **DAntona_Fit.fit(teff)
    })
    
    mapping = {key: u.solMass for key in result.keys()}
    result = QTable.from_pandas(result, units=mapping)
    return result


def plot_four_catalogs(catalogs, save_path, save=True, max_sep=1.*u.arcsec, marker_size=3, opacity=1):
    """Plot four catalogs:NIRSPAO sources, APOGEE, Proper Motion by Kim, and Gaia DR3.
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
    
    pm_coord = SkyCoord(ra=pm['RAJ2000'], dec=pm['DEJ2000'])
    
    matched_circles_idx = np.arange(len(sources_epoch_combined))[~sources_epoch_combined['Gaia DR3'].mask | ~sources_epoch_combined['ID_kim'].mask]
    
    ply_shapes = dict()
    
    for i in matched_circles_idx:
        ply_shapes['shape_' + str(i)] = go.layout.Shape(
            type='circle',
            xref='x', yref='y',
            x0=(sources_epoch_combined['RAJ2000'][i] - max_sep / np.cos(sources_epoch_combined['DEJ2000'][i])).to(u.deg).value,
            y0=(sources_epoch_combined['DEJ2000'][i] - max_sep).to(u.deg).value,
            x1=(sources_epoch_combined['RAJ2000'][i] + max_sep / np.cos(sources_epoch_combined['DEJ2000'][i])).to(u.deg).value,
            y1=(sources_epoch_combined['DEJ2000'][i] + max_sep).to(u.deg).value,
            line_color='#2f2f2f',
            line_width=1,
            opacity=opacity,
            name='Matched Sources'
        )
       
    trapezium = SkyCoord("05h35m16.26s", "-05d23m16.4s")
    
    apogee_index = [list(apogee['APOGEE']).index(_) for _ in sources_epoch_combined['APOGEE'][~sources_epoch_combined['APOGEE'].mask]]
    
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
            x=gaia['RAJ2000'].to(u.deg).value, 
            y=gaia['DEJ2000'].to(u.deg).value, 
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
            x=(pm['RAJ2000'][pm_coord.separation(trapezium) < 4*u.arcmin]).to(u.deg).value, 
            y=(pm['DEJ2000'][pm_coord.separation(trapezium) < 4*u.arcmin]).to(u.deg).value, 
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
            x=(apogee['RAJ2000'][apogee_index]).to(u.deg).value, 
            y=(apogee['DEJ2000'][apogee_index]).to(u.deg).value, 
            opacity=1,
            marker=dict(
                size=marker_size,
                color=C1
            ),
            legendrank=3
        ),
        
        # Plot NIRSPAO Sources
        go.Scatter(
            mode='markers',
            name='NIRSPAO Sources',
            x=sources_epoch_combined['RAJ2000'][~sources_epoch_combined['HC2000'].mask], 
            y=sources_epoch_combined['DEJ2000'][~sources_epoch_combined['HC2000'].mask], 
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
        fig.write_html(f"{user_path}/ONC/figures/synthetic skymap.html")



###############################################
######### Construct Synthetic Catalog #########
###############################################

def construct_synthetic_catalog(nirspao_path, save_path):
    '''Construct synthetic catalog combining nirspao, apogee, pm, gaia, and ONC Mass catalog by Hillenbrand 1997 (J/AJ/113/1733/ONC).
    --------------------
    Both a multi-epoch combined & not combined catalog are saved under the save_path folder.
    - Parameters:
        - nirspao_path: path of nirspao catalog ecsv.
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
    
    # Read tables
    nirspao     = QTable.read(nirspao_path)
    nirspao_old = Vizier.get_catalogs('J/ApJ/926/141/table3')[0]
    apogee      = Vizier.get_catalogs('J/ApJ/926/141/table5')[0]
    # kounkel     = Vizier.get_catalogs('J/AJ/156/84/table1')[0]
    kounkel     = Vizier(
        columns=['RAJ2000', 'DEJ2000', '2MASS', 'Gaia', 'Teff', 'e_Teff', 'logg', 'e_logg', 'RVmean', 'e_RVmean', 'Fbol', 'Av', 'Theta', 'SimbadName'],
        row_limit=-1
    ).get_catalogs('J/AJ/156/84/table1')[0]
    kim         = Vizier.get_catalogs('J/AJ/157/109/table3')[0]
    gaia        = Gaia.cone_search_async(trapezium, radius=u.Quantity(4.2, u.arcmin)).get_results()
    hc2000      = Vizier.get_catalogs('J/ApJ/540/236/table1')[0]
    hillenbrand = Vizier.get_catalogs('J/AJ/113/1733/ONC')[0]
    tobin       = Vizier.get_catalogs('J/ApJ/697/1103/table3')[0]
    
    # Cross match apogee with 2MASS to get Hmag and Kmag
    apogee_keys = apogee.keys()
    apogee_units = [apogee[_].unit for _ in apogee.keys()]
    apogee = XMatch.query(
        cat1=apogee, cat2='vizier:II/246/out', 
        max_distance=1 * u.arcsec, 
        colRA1='RAJ2000', colDec1='DEJ2000'
    )
    apogee.remove_columns(['angDist', '2MASS', 'RAJ2000_2', 'DEJ2000_2'])
    apogee.rename_columns(['RAJ2000_1', 'DEJ2000_1'], ['RAJ2000', 'DEJ2000'])
    for key, unit in zip(apogee_keys, apogee_units):
        apogee[key].unit = unit
    for key in ['Jmag', 'Hmag', 'Kmag', 'e_Jmag', 'e_Hmag', 'e_Kmag']:
        apogee[key].unit = u.mag
    
    # Rename
    hc2000.rename_columns(['__HC2000_'], ['HC2000'])
    
    nirspao_old.rename_columns(
        ['__HC2000_',   'm__HC2000_',   'RV', 'e_RV', 'Teff', 'e_Teff', 'Veiling'],
        ['HC2000',      'm_HC2000',     'rv', 'e_rv', 'teff', 'e_teff', 'veiling_param_O33']
    )
    # Convert empty string to masked element in nirspao_old
    nirspao_old['m_HC2000'] = MaskedColumn(nirspao_old['m_HC2000'], mask=[True if _=='' else False for _ in nirspao_old['m_HC2000']], dtype=nirspao['m_HC2000'].dtype)
        
    apogee.rename_columns(
        ['RV',          'e_RV',         'Teff',         'e_Teff',           'vsini',        'e_vsini',          'Veiling',              'Kmag',         'e_Kmag',           'Hmag',         'e_Hmag'],
        ['rv_apogee',   'e_rv_apogee',  'teff_apogee',  'e_teff_apogee',    'vsini_apogee', 'e_vsini_apogee',   'veiling_param_apogee', 'Kmag_apogee',  'e_Kmag_apogee',    'Hmag_apogee',  'e_Hmag_apogee']
    )
    
    kounkel.rename_columns(
        ['_2MASS',  'Gaia',     'Teff',         'e_Teff',           'logg',         'e_logg',           'RVmean',       'e_RVmean'],
        ['2MASS',   'Gaia DR2', 'teff_kounkel', 'e_teff_kounkel',   'logg_kounkel', 'e_logg_kounkel',   'rv_kounkel',   'e_rv_kounkel']
    )
    
    kim.rename_columns(
        ['ID',      'pmRA',     'e_pmRA',       'pmDE',     'e_pmDE'],
        ['ID_kim',  'pmRA_kim', 'e_pmRA_kim',   'pmDE_kim', 'e_pmDE_kim']
    )
    
    gaia.rename_columns(
        ['source_id',   'ra',       'dec',      'parallax', 'parallax_error',   'parallax_over_error',  'pmra',         'pmra_error',   'pmdec',        'pmdec_error', 'phot_g_mean_mag'],
        ['Gaia DR3',    'RAJ2000',  'DEJ2000',  'plx',      'e_plx',            'plx_over_e',           'pmRA_gaia',    'e_pmRA_gaia',  'pmDE_gaia',    'e_pmDE_gaia', 'Gmag']
    )
    gaia['Gaia DR3'] = MaskedColumn([str(_) for _ in gaia['Gaia DR3']])
    
    hillenbrand.rename_columns(
        ['ID', 'M'],
        ['ID_hillenbrand', 'mass_Hillenbrand']
    )
    
    tobin.rename_columns(
        ['_2MASS',      'HRV',      'e_HRV',        'Vmag',         'V-I'], 
        ['2MASS_tobin', 'rv_tobin', 'e_rv_tobin',   'Vmag_tobin',   'V-I_tobin']
    )
    
    
    # Add & Replace Trapezium Stars
    trapezium_stars = QTable()

    # A1-A3, B1-B5, C1-C2, D, EA-EB
    trapezium_stars['HC2000'] = [336, 354, 309, 330, 344]
    trapezium_stars['theta_orionis'] = ['A', 'B', 'C', 'D', 'E']
    trapezium_stars['mass_literature'] = np.array([26.6, 16.3, 44, 25, 5.604]) * u.solMass
    trapezium_stars['e_mass_literature'] = np.array([26.6*0.05, 16.3*0.05, 5*2**0.5, 25*0.05, 0.048*2**0.5]) * u.solMass
    
    # trapezium_stars = Table(trapezium_stars, units={'mass_literature': u.solMass, 'e_mass_literature': u.solMass})
    trapezium_stars.add_columns([np.empty(5)*u.deg, np.empty(5)*u.deg], indexes=[2, 2], names=['RAJ2000', 'DEJ2000'])
    
    # Get RA DEC from HC2000 catalog
    coords = SkyCoord([f'{ra} {dec}' for ra, dec in zip(hc2000['RAJ2000'], hc2000['DEJ2000'])], unit=(u.hourangle, u.deg))
    for i in range(len(trapezium_stars)):
        idx = list(hc2000['HC2000']).index(trapezium_stars['HC2000'][i])
        trapezium_stars['RAJ2000'][i] = coords[idx].ra.deg * u.deg
        trapezium_stars['DEJ2000'][i] = coords[idx].dec.deg * u.deg
    
    if save_path is not None:
        trapezium_stars.write(f'{save_path}/trapezium_stars.ecsv', overwrite=True)
    # Remove trapezium stars from nirspao? No!
    # nirspao.remove_rows([index for index, element in enumerate(nirspao['HC2000']) if element in trapezium_stars['HC2000']])
    
    # Merge nirspec and trapezium stars
    nirspao = vstack((nirspao, trapezium_stars))
    nirspao.rename_column('theta_orionis', 'theta_orionis-temp')
    nirspao.add_column(nirspao['theta_orionis-temp'], name='theta_orionis', index=2)
    nirspao.remove_column('theta_orionis-temp')
    
    for trapezium_HC2000 in trapezium_stars['HC2000']:
        if any(nirspao['theta_orionis'][nirspao['HC2000']==trapezium_HC2000].mask):
            nirspao.remove_rows((nirspao['HC2000']==trapezium_HC2000) & ~nirspao['theta_orionis'].mask)
            nirspao['theta_orionis'][nirspao['HC2000']==trapezium_HC2000] = trapezium_stars['theta_orionis'][trapezium_stars['HC2000']==trapezium_HC2000]
            nirspao['mass_literature'][nirspao['HC2000']==trapezium_HC2000] = trapezium_stars['mass_literature'][trapezium_stars['HC2000']==trapezium_HC2000]

    
    # Remove multiplicty index with only A or B
    for i in np.where(~nirspao_old['m_HC2000'].mask)[0]:
        if not all([_ in nirspao_old['m_HC2000'][nirspao_old['HC2000']==nirspao_old['HC2000'][i]] for _ in ['A', 'B']]):
            nirspao_old['m_HC2000'][i] = ma.masked    
    
    # Add Gmag uncertainty.
    # magnitude = -2.5*log10(flux) + zero point. (https://en.wikipedia.org/wiki/Apparent_magnitude#Calculations)
    # zero point: 25.6874 ± 0.0028. (https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photProc/cu5pho_ssec_photCal.html#SSS3.P2)
    gaia.add_column(
        MaskedColumn(((2.5/np.log(10)*gaia['phot_g_mean_flux_error']/gaia['phot_g_mean_flux'])**2 + 0.0027553202**2)**(1/2), unit=u.mag),
        index=gaia.colnames.index('Gmag') + 1,
        name='Gmag_e'
    )
    
    # Offset specified in https://iopscience.iop.org/article/10.1086/309309/pdf, Section 4.5.
    hillenbrand_coord = SkyCoord([f'{ra} {dec}' for ra, dec in zip(hillenbrand['RAJ2000'], hillenbrand['DEJ2000'])], unit=(u.hourangle, u.deg))
    hillenbrand['RAJ2000'] = hillenbrand_coord.ra + 1.5*u.arcsec
    hillenbrand['DEJ2000'] = hillenbrand_coord.dec - 0.3*u.arcsec
    hillenbrand_coord = SkyCoord(ra=hillenbrand['RAJ2000'], dec=hillenbrand['DEJ2000'])
    hillenbrand = hillenbrand[hillenbrand_coord.separation(trapezium) <= 4*u.arcmin]
    
    tobin_coord = SkyCoord([f'{ra} {dec}' for ra, dec in zip(tobin['RAJ2000'], tobin['DEJ2000'])], unit=(u.hourangle, u.deg))
    tobin['RAJ2000'] = tobin_coord.ra
    tobin['DEJ2000'] = tobin_coord.dec
    
    ###############################################
    ################# Cross Match #################
    ###############################################
    
    # match nirspao_old    
    sources = merge_on_keys(nirspao, nirspao_old[[
        'HC2000', 'm_HC2000', 
        'teff', 'e_teff', 
        'rv', 'e_rv', 
        'vsini', 'e_vsini', 
        'veiling_param_O33'
    ]], join_type='left', keys=['HC2000', 'm_HC2000'], table_names=['nirspao', 'chris'])
    sources.rename_columns(['HC2000_nirspao', 'm_HC2000_nirspao'], ['HC2000', 'm_HC2000'])
    sources.remove_columns(['HC2000_chris', 'm_HC2000_chris'])

    # cross match apogee
    print('length of apogee: {}'.format(len(apogee)))
    sources = merge_on_coords(sources, apogee[[
        'APOGEE',
        'RAJ2000', 'DEJ2000', 
        'rv_apogee', 'e_rv_apogee', 
        'teff_apogee', 'e_teff_apogee', 
        'vsini_apogee', 'e_vsini_apogee', 
        'veiling_param_apogee',
        'Kmag_apogee', 'e_Kmag_apogee', 
        'Hmag_apogee', 'e_Hmag_apogee'
    ]], join_type='outer', table_names=['nirspao', 'apogee'])
    
    # cross match kounkel
    sources = merge_on_coords(sources, kounkel[[
        '2MASS', 'Gaia DR2', 
        'RAJ2000', 'DEJ2000', 
        'teff_kounkel', 'e_teff_kounkel', 
        'logg_kounkel', 'e_logg_kounkel', 
        'rv_kounkel', 'e_rv_kounkel'
    ]], join_type='left', table_names=['nirspao+apogee', 'kounkel'])

    # cross match proper motion
    sources = merge_on_coords(sources, kim[[
        'ID_kim', 
        'RAJ2000', 'DEJ2000', 
        'pmRA_kim', 'e_pmRA_kim', 
        'pmDE_kim', 'e_pmDE_kim'
    ]], join_type='left', table_names=['nirspao+apogee', 'kim'])
    
    sources_coord = SkyCoord(ra=sources['RAJ2000'], dec=sources['DEJ2000'])
    gaia_coord = SkyCoord(ra=gaia['RAJ2000'], dec=gaia['DEJ2000'])
    
    max_sep = 1 * u.arcsec
    matches = cross_match_gaia(sources_coord, gaia_coord, gaia['plx'], max_sep=max_sep)
    
    # cross match gaia
    sources = merge_on_coords(sources, gaia[[
        'Gaia DR3',
        'RAJ2000', 'DEJ2000', 
        'plx', 'e_plx', 'plx_over_e',
        'pmRA_gaia', 'e_pmRA_gaia', 
        'pmDE_gaia', 'e_pmDE_gaia', 
        'Gmag', 'Gmag_e', 'bp_rp', 
        'astrometric_n_good_obs_al', 'astrometric_gof_al', 'astrometric_excess_noise', 'ruwe'
    ]], matches=matches, table_names=['nirspao+apogee', 'gaia'])
    
    # cross match hillenbrand
    sources = merge_on_coords(sources, hillenbrand[[
        'ID_hillenbrand', 'RAJ2000', 'DEJ2000','mass_Hillenbrand'
    ]], max_sep=2*u.arcsec, table_names=['nirspao+apogee', 'hillenbrand'])
    
    # cross match tobin
    sources = merge_on_coords(sources, tobin[[
        'RAJ2000', 'DEJ2000', 'rv_tobin', 'e_rv_tobin'
    ]], table_names=['nirspao+apogee', 'tobin'])
    
    
    ###############################################
    ########### Merge Duplicate Columns ###########
    ###############################################
    
    sources.add_column(fillna(sources['teff_nirspao'], sources['teff_apogee']), index=10, name='teff')
    sources.add_column(fillna(sources['e_teff_nirspao'], sources['e_teff_apogee']), index=11, name='e_teff')
    
    sources['Kmag'], sources['e_Kmag'] = weighted_avrg_and_merge(
        sources['Kmag'], 
        sources['Kmag_apogee'], 
        error1=sources['e_Kmag'], 
        error2=sources['e_Kmag_apogee']
    )
    sources.remove_columns(['Kmag_apogee', 'e_Kmag_apogee'])
    
    sources.rename_column('APOGEE', 'APOGEE-temp')
    sources.add_column(sources['APOGEE-temp'], index=3, name='APOGEE')
    sources.remove_column('APOGEE-temp')
    
    sources_epoch_combined = merge_multiepoch(sources)
        

    ###############################################
    ################### Fit Mass ##################
    ###############################################
    sources = hstack((sources, fit_mass(sources['teff'], sources['e_teff'])))
    sources_epoch_combined = hstack((sources_epoch_combined, fit_mass(sources_epoch_combined['teff'], sources_epoch_combined['e_teff'])))
    
    # Delete Fitting results for trapezium stars
    columns = fit_mass(sources['teff'], sources['e_teff']).keys()
    indices = ~sources['theta_orionis'].mask
    sources[columns][indices] = np.nan
    
    indices = ~sources_epoch_combined['theta_orionis'].mask
    sources_epoch_combined[columns][indices] = np.nan
    
    # plot
    plot_four_catalogs([sources_epoch_combined, apogee, kim, gaia], save_path=save_path)

    ###############################################
    ################# Write to csv ################
    ###############################################
    sources.write(f'{save_path}/synthetic catalog.ecsv', overwrite=True)
    sources.write(f'{save_path}/synthetic catalog - new.csv', overwrite=True)
    sources_epoch_combined.write(f'{save_path}/synthetic catalog - epoch combined.ecsv', overwrite=True)
    sources_epoch_combined.write(f'{save_path}/synthetic catalog - epoch combined - new.csv', overwrite=True)
    
    return sources, sources_epoch_combined





if __name__=='__main__':
    nirspao_path        = f'{user_path}/ONC/starrynight/catalogs/nirspao sources.ecsv'
    save_path           = f'{user_path}/ONC/starrynight/catalogs'

    sources, sources_epoch_combined = construct_synthetic_catalog(nirspao_path, save_path)