from helpers.metrics import test_compute_metrics
from plotly.subplots import make_subplots
import plotly.express as px


def plot_results_simple(response_df, args):
    combs = [[0, 1], [0, 3],
            [1, 1], [1, 3]]
    metrics_2 = [0, 0, 0, 0]
    metric_2 = 'TPR - Recall'
    for idx in range(4):
        c = combs[idx]
        sub_df = response_df[(response_df.gender_ == c[0]) & (response_df.race_ == c[1])]
        metrics_2[idx] = test_compute_metrics(sub_df, args.protected_attributes)[metric_2].iloc[0]
    metrics_1 = [0, 0, 0, 0]
    metric_1 = 'TNR'
    for idx in range(4):
        c = combs[idx]
        sub_df = response_df[(response_df.gender_ == c[0]) & (response_df.race_ == c[1])]
        metrics_1[idx] = test_compute_metrics(sub_df, args.protected_attributes)[metric_1].iloc[0]
        
    # Plot
    fig = make_subplots(rows = 1, cols = 2)
    
    trace_1 = px.imshow([metrics_1[:2], metrics_1[2:]],
                    labels = dict(x = 'Race', y = 'Gender', color = metric_1),
                    title = metric_1,
                    x = ['Asian', 'Black'],
                    y = ['Female', 'Male'], text_auto = '.3r', color_continuous_scale = px.colors.sequential.Viridis)
    
    trace_2 = px.imshow([metrics_2[:2], metrics_2[2:]],
                    labels = dict(x = 'Race', y = 'Gender', color = metric_2),
                    title = metric_2,
                    x = ['Asian', 'Black'],
                    y = ['Female', 'Male'], text_auto = '.3r', color_continuous_scale = px.colors.sequential.Viridis)
    
    # Update the figure
    for t in range(len(trace_1['data'])): fig.add_trace(trace_1['data'][t], row = 1, col = 1)
    for t in range(len(trace_2['data'])): fig.add_trace(trace_2['data'][t], row = 1, col = 2)
    fig.show()
    
    