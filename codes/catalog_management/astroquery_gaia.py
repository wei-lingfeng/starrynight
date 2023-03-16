# import astropy.units as u
# from astropy.coordinates import SkyCoord
# from astroquery.gaia import Gaia

# trapezium = SkyCoord("05h35m16.26s", "-05d23m16.4s")
# trapezium = trapezium.transform_to(frame='icrs')

# Gaia.ROW_LIMIT = 50  # Ensure the default row limit.
# radius = u.Quantity(4.2, u.arcmin)
# j = Gaia.cone_search_async(trapezium, radius)
# r = j.get_results()
# r.pprint()