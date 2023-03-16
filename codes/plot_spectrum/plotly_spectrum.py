import os
import copy
import smart
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

apogee_id = '2M05351259-0523440'
instrument = 'apogee'
object_path = '/home/l3wei/ONC/Data/APOGEE/{}/'.format(apogee_id)

def plot_specs(specs):
    fig_data = []
    for spec in specs:
        fig_data.append(go.Scatter(x=spec.wave, y=spec.flux, mode='lines+markers', name='Visit {}'.format(visit), line=dict(width=1), marker=dict(size=1.5)))
    
    fig = go.Figure()
    fig.add_traces(fig_data)
    fig.update_layout(width=1000, height=500, xaxis = dict(tickformat='000'))
    return fig

n_visit = len([_ for _ in os.listdir(object_path + 'specs/') if os.path.isfile(object_path + 'specs/' + _) and _.startswith('apVisit')])
specs = []

for visit in range(1, n_visit + 1):
    data_path  = object_path + 'specs/' + 'apVisit-' + apogee_id + '-{}.fits'.format(visit)
    spec = smart.Spectrum(name=apogee_id, path=data_path, instrument=instrument, applymask=True, datatype='apvisit', applytell=True)    
    # Normalize
    spec.noise = spec.noise / np.median(spec.flux)
    spec.flux  = spec.flux  / np.median(spec.flux)
    specs.append(copy.deepcopy(spec))



# fig = plot_specs(specs=specs_cut)
# fig.write_html('APOGEE Spectrum.html')
# fig.show()