# Code Description

| File |  Description  |
|---|---|
| **[StarCluster.py](StarCluster.py)** | **The primary analysis code, published version of analysis in the style of object-oriented programing** |
| [analysis_old.py](analysis_old.py) |  Legacy: Previous version of analysis  |
| [close_binary.py](close_binary.py)  | Plot the vrel vs mass using the simulated binary mass with the observed mass as the primary |
| [corner_plot.py](corner_plot.py) | Helper function for corner plot |
| [fit_line.py](fit_line.py) | Fit the slope k and intercept b of the form y = kx + b to data with xerr and yerr |
| [fit_offset.py](fit_offset.py) | Fit a line y=x+b to data with xerr and yerr |
| [fit_vdisp pandas.py](fit_vdisp_pandas.py) | Legacy: Fit velocity dispersion for all three directions: ra, dec, and rv, using pandas dataframe |
| [fit_vdisp.py](fit_vdisp.py) | Currently used: Fit velocity dispersion for all three directions: ra, dec, and rv, using astropy QTable |
| [flat_specs.py](flat_specs.py) | Exploring the sources with flat spectra |
| [KFold.py](KFold.py) | K-fold cross validation of the vrel-mass slope |
| [linear_regression.py](linear_regression.py) | Weighted linear regression |
