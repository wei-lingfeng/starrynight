import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.esa.jwst import Jwst

Jwst.login(user='l3wei@ucsd.edu', password='Louiswlf1126')
Jwst.set_token(token='9de258a66f7240509a838ac57baaf649')

trapezium = SkyCoord("05h35m16.26s", "-05d23m16.4s", distance=1000/2.59226*u.pc)
radius = u.Quantity(5.0, u.deg)
j = Jwst.cone_search(coordinate=trapezium, radius=radius, async_job=True)
result = j.get_results()

# product_list = Jwst.get_product_list(observation_id=result['observationid'][0], product_type='science')
results = Jwst.get_obs_products(observation_id=result['observationid'][0], product_type='science') 