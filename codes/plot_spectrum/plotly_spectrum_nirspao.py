import os
import copy
import smart
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

sci_frames = [40, 41, 42, 43]
order = 33
prefix = '/home/l3wei/ONC/data/NIRSPAO/2022jan20/reduced/'

def plot_specs(specs):
    fig_data = []
    for i, spec in enumerate(specs):
        fig_data.append(go.Scatter(x=np.arange(len(spec.wave)), y=spec.flux, mode='lines+markers', name=f'Frame {i+1}', line=dict(width=1), marker=dict(size=3)))
    
    fig = go.Figure()
    fig.add_traces(fig_data)
    fig.update_layout(width=1000, height=500, xaxis = dict(tickformat='000'))
    return fig

specs = []
for sci_frame in sci_frames:
    sci_name = f'nspec220120_{str(sci_frame).zfill(4)}'
    spec = smart.Spectrum(name=sci_name, order=order, path=f'{prefix}nsdrp_out/fits/all')
    # Normalize
    spec.noise  = spec.noise / np.median(spec.flux) 
    spec.flux   = spec.flux / np.median(spec.flux) 
    specs.append(spec)

fig = plot_specs(specs=specs)
fig.update_layout(xaxis_title='Pixel', yaxis_title='Normalized Flux')
fig.show()