from helpers.metrics import compute_metric
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


def plot_dist(data, corr, **kwargs):
    # Change the format of the references
    ref = data.references
    ref.sort_values(by = ['age_'], inplace = True)
    ref['age_str'] = ref.age_.astype(str)
    ref['race_str'] = ref.apply(lambda row : corr['race_'][row['race_']], axis = 1)
    ref['gender_str'] = ref.apply(lambda row : corr['gender_'][row['gender_']], axis = 1)
    
    # Initialization of the plot
    fig = make_subplots(rows = 2, cols = 3)
    
    # Figure 1
    trace_1 = px.histogram(ref, x = 'age_str', color = 'gender_str', text_auto = ".3r")
    trace_2 = px.histogram(ref, x = 'race_str', color = 'gender_str', text_auto = ".3r").update_traces(showlegend=False)
    trace_3 = px.histogram(ref, x = 'gender_str', color = 'gender_str', text_auto = ".3r").update_traces(showlegend=False)
    trace_4 = px.density_heatmap(ref[ref.gender_ == 0], x = 'race_str', y = 'age_str', text_auto = '.3r', color_continuous_scale = px.colors.sequential.PuBuGn)
    trace_5 = px.density_heatmap(ref[ref.gender_ == 1], x = 'race_str', y = 'age_str', text_auto = '.3r',color_continuous_scale=px.colors.sequential.Inferno)#, color_continuous_midpoint = px.colors.sequential.PuBuGn)
    # Update the figure
    for t in range(len(trace_1['data'])): fig.add_trace(trace_1['data'][t], row = 1, col = 1)
    for t in range(len(trace_2['data'])): fig.add_trace(trace_2['data'][t], row = 1, col = 2)
    for t in range(len(trace_3['data'])): fig.add_trace(trace_3['data'][t], row = 1, col = 3)
    for t in range(len(trace_4['data'])): fig.add_trace(trace_4['data'][t], row = 2, col = 1)
    for t in range(len(trace_5['data'])): fig.add_trace(trace_5['data'][t], row = 2, col = 2)
    fig.update_layout(height = 1000, width = 1200)
    fig.update_traces(bingroup=None)
    fig.update_coloraxes(showscale=False)
    fig.show()