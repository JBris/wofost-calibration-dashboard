from dash import dcc, html, Input, Output, register_page, callback, no_update, State, dash_table
import pandas as pd
import dash_bootstrap_components as dbc
from dataproviders import parameters, agromanagement, weather
import copy
from pcse.models import Wofost72_WLP_FD
import numpy as np
import plotly.graph_objects as go
import sklearn.metrics as skmetrics

register_page(__name__, path="/what-if", name="What-If Analysis")

state = {}

def get_obs_data():
    true_params = {}
    true_params["SPAN"] = 33

    p = copy.deepcopy(parameters)
    for par, distr in true_params.items():
        p.set_override(par, distr)
    ground_truth = Wofost72_WLP_FD(p, weather, agromanagement)
    ground_truth.run_till_terminate()

    df = pd.DataFrame(ground_truth.get_output())
    obs = df["LAI"].values

    return df, obs

df, obs = get_obs_data()

layout = dbc.Row([
    dbc.Col([
        dbc.Card([
            html.H4("Scenario A: Parameters", style={'textAlign': 'center', "padding": "0.5em"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Life span of leaves growing at 35°C mean", style={"padding-left": "2em"}),
                    dbc.Tooltip("The mean for life span of leaves growing at 35°C", target="input-span-a-mean"),
                    dcc.Input(
                        id="input-span-a-mean",
                        type="number",
                        value=31,
                        placeholder="SPAN Mean",
                        style={"width": "65%", "margin-left": "2em"}
                    ),
                ]),
                dbc.Col([
                    html.Label("Life span of leaves growing at 35°C standard deviation", style={"padding-left": "2em"}),
                    dbc.Tooltip("The standard deviation for life span of leaves growing at 35°C", target="input-span-a-sd"),
                    dcc.Input(
                        id="input-span-a-sd",
                        type="number",
                        value=3,
                        placeholder="SPAN Standard Deviation",
                        style={"width": "65%", "margin-left": "2em"}
                    ),
                ])
            ]),
            
        ], style={"padding-bottom": "2em"}),

        dbc.Card([
            html.H4("Scenario B: Parameters", style={'textAlign': 'center', "padding": "0.5em"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Life span of leaves growing at 35°C mean", style={"padding-left": "2em"}),
                    dbc.Tooltip("The mean for life span of leaves growing at 35°C", target="input-span-b-mean"),
                    dcc.Input(
                        id="input-span-b-mean",
                        type="number",
                        value=19,
                        placeholder="SPAN Mean",
                        style={"width": "65%", "margin-left": "2em"}
                    ),
                ]),
                dbc.Col([
                    html.Label("Life span of leaves growing at 35°C standard deviation", style={"padding-left": "2em"}),
                    dbc.Tooltip("The standard deviation for life span of leaves growing at 35°C", target="input-span-b-sd"),
                    dcc.Input(
                        id="input-span-b-sd",
                        type="number",
                        value=3,
                        placeholder="SPAN Standard Deviation",
                        style={"width": "65%", "margin-left": "2em"}
                    ),
                ])
            ]),
        ], style={"padding-bottom": "2em"}),

        dbc.Card([
            html.H4("What-If Analysis", style={'textAlign': 'center', "padding": "0.5em"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Ensemble size", style={"padding-left": "2em"}),
                    dbc.Tooltip("The number of ensemble members", target="input-ensemble-size"),
                    dcc.Input(
                        id="input-ensemble-size",
                        type="number",
                        value=5,
                        placeholder="Ensemble Size",
                        style={"margin-left": "2em"}
                    ),
                ]),
                  dbc.Col([
                    html.Label("Alpha", style={"padding-left": "2em"}),
                    dbc.Tooltip("The alpha-level for the prediction intervals", target="input-alpha"),
                    dcc.Input(
                        id="input-alpha",
                        type="number",
                        value=0.95,
                        placeholder="Alpha",
                        style={"margin-left": "2em", "margin-bottom": "2em"}
                    ),
                ]),
            ]),
 
            dbc.Row([
                dbc.Col(
                [
                    dbc.Tooltip("Run the what-if analysis procedure", target="btn-run"),
                    dbc.Button("Run", id="btn-run", color="primary", className="me-1", style={"width": "30%", "margin-left": "3em", "padding-up": "2em", "margin-bottom": "2em"}),
                ]
                ),
                dbc.Col(
                    [
                        dbc.Tooltip("Reset the state of the ensemble", target="btn-reset"),
                        dbc.Button("Reset", id="btn-reset", color="primary", className="me-1", style={"width": "30%", "margin-left": "3em", "padding-up": "2em", "margin-bottom": "2em"}),
                    ]
                )
            ]),
            dbc.Row(
                dash_table.DataTable(
                    id='what-if-metrics-table',
                    columns=[
                        {"name": "scenario", "id": "scenario", "editable": True}, 
                        {"name": "metric", "id": "metric", "editable": True}, 
                        {"name": "value", "id": "value", "editable": True}
                    ],
                    data=[],
                    filter_action="native",    
                    sort_action="native",      
                    row_selectable="multi",   
                    selected_rows=[],         
                    page_action="native",     
                    page_size=10,              
                )
            ),
        ], style={"padding-bottom": "2em"}),
    ]),
    dbc.Col([
        dbc.Row(dcc.Graph(id='lai-scatter-a-plot')),
        dbc.Row(dcc.Graph(id='lai-scatter-b-plot')),
        dbc.Row(dcc.Graph(id='lai-pi-scatter-a-plot')),
        dbc.Row(dcc.Graph(id='lai-pi-scatter-b-plot')),
        dbc.Row(dcc.Graph(id='lai-pred-obs-scatter-a-plot')),
        dbc.Row(dcc.Graph(id='lai-pred-obs-scatter-b-plot')),
        dbc.Row(dcc.Graph(id='lai-residual-scatter-a-plot')),
        dbc.Row(dcc.Graph(id='lai-residual-scatter-b-plot')),
        dbc.Row(dcc.Graph(id='span-hist-a-plot')),
        dbc.Row(dcc.Graph(id='span-hist-b-plot')),
    ]),
])

def get_quantiles(Y_IES_ert, alpha):
    lower_q = (1 - alpha) / 2   
    upper_q = 1 - lower_q     
    stacked_Y_IES = np.stack(Y_IES_ert)
    median = np.median(stacked_Y_IES.T, axis=0)
    lower = np.quantile(stacked_Y_IES.T, q=lower_q, axis=0)
    upper = np.quantile(stacked_Y_IES.T, q=upper_q, axis=0)

    return median, lower, upper, lower_q, upper_q

@callback(
    Output('lai-scatter-a-plot', 'figure', allow_duplicate=True),
    Output('lai-scatter-b-plot', 'figure', allow_duplicate=True),
    Output('lai-pi-scatter-a-plot', 'figure', allow_duplicate=True),
    Output('lai-pi-scatter-b-plot', 'figure', allow_duplicate=True),
    Output('lai-pred-obs-scatter-a-plot', 'figure', allow_duplicate=True),
    Output('lai-pred-obs-scatter-b-plot', 'figure', allow_duplicate=True),
    Output('lai-residual-scatter-a-plot', 'figure', allow_duplicate=True),
    Output('lai-residual-scatter-b-plot', 'figure', allow_duplicate=True),
    Output('span-hist-a-plot', 'figure', allow_duplicate=True),
    Output('span-hist-b-plot', 'figure', allow_duplicate=True),
    Output('what-if-metrics-table', 'data', allow_duplicate=True),
    Input("btn-run", "n_clicks"),
    State("input-span-a-mean", "value"),
    State("input-span-a-sd", "value"),
    State("input-span-b-mean", "value"),
    State("input-span-b-sd", "value"),
    State("input-ensemble-size", "value"),
    State("input-alpha", "value"),
    prevent_initial_call=True
)
def run_ensemble(
    n, span_a_mean, span_a_sd, span_b_mean, span_b_sd, 
    ensemble_size, alpha
):
    if n == 0:
        return no_update
    
    override_parameters_a = {}
    override_parameters_a["SPAN"] = np.random.normal(span_a_mean, span_a_sd ,(ensemble_size))
    override_parameters_b = {}
    override_parameters_b["SPAN"] = np.random.normal(span_b_mean, span_b_sd ,(ensemble_size))

    ensemble_a = []
    for i in range(ensemble_size):
        p = copy.deepcopy(parameters)
        for par, distr in override_parameters_a.items():
            p.set_override(par, distr[i])
        member = Wofost72_WLP_FD(p, weather, agromanagement)
        member.run_till_terminate()
        ensemble_a.append(member)
    results = [pd.DataFrame(member.get_output()).set_index("day") for member in ensemble_a]
    LAI_a = np.array([
        result["LAI"].values
        for result in results
    ]).T

    ensemble_b = []
    for i in range(ensemble_size):
        p = copy.deepcopy(parameters)
        for par, distr in override_parameters_b.items():
            p.set_override(par, distr[i])
        member = Wofost72_WLP_FD(p, weather, agromanagement)
        member.run_till_terminate()
        ensemble_b.append(member)
    results = [pd.DataFrame(member.get_output()).set_index("day") for member in ensemble_b]
    LAI_b = np.array([
        result["LAI"].values
        for result in results
    ]).T

    fig_span_a = go.Figure()    
    fig_span_a.add_trace(go.Histogram(x=override_parameters_a["SPAN"], name = "Prior SPAN"))        
    fig_span_a.update_layout(xaxis=dict(title="Scenario A: SPAN"), showlegend=True)

    fig_span_b = go.Figure()    
    fig_span_b.add_trace(go.Histogram(x=override_parameters_b["SPAN"], name = "Prior SPAN"))        
    fig_span_b.update_layout(xaxis=dict(title="Scenario B: SPAN"), showlegend=True)

    fig_lai_a = go.Figure()
    fig_lai_a.add_trace(go.Scatter(x=df["day"], y=df["LAI"], name="Observed LAI"))        
    fig_lai_a.update_layout(xaxis=dict(title="Scenario A: LAI"), showlegend=True)
    for i in range(ensemble_size):
        fig_lai_a.add_trace(go.Scatter(x=df["day"], y=LAI_a.T[i], showlegend=False)) 

    fig_lai_b = go.Figure()
    fig_lai_b.add_trace(go.Scatter(x=df["day"], y=df["LAI"], name="Observed LAI"))        
    fig_lai_b.update_layout(xaxis=dict(title="Scenario B: LAI"), showlegend=True)
    for i in range(ensemble_size):
        fig_lai_b.add_trace(go.Scatter(x=df["day"], y=LAI_b.T[i], showlegend=False)) 

    median, lower, upper, lower_q, upper_q = get_quantiles(LAI_a, alpha)
    fig_pi_lai_a = go.Figure()
    fig_pi_lai_a.add_trace(go.Scatter(x=df["day"], y=df["LAI"], name="Observed LAI")) 
    fig_pi_lai_a.add_trace(go.Scatter(x=df["day"], y=median, name="Ensemble median LAI"))  
    fig_pi_lai_a.add_trace(go.Scatter(x=df["day"], y=lower, name=f"Lower quantile LAI ({round(lower_q, 2)})"))  
    fig_pi_lai_a.add_trace(go.Scatter(x=df["day"], y=upper, name=f"Upper quantile LAI ({round(upper_q, 2)})"))        
    fig_pi_lai_a.update_layout(xaxis=dict(title=f"Scenario A: Quantile LAI (alpha={alpha})"), showlegend=True)

    fig_pred_obs_a = go.Figure()
    fig_pred_obs_a.add_trace(go.Scatter(x=median, y=df["LAI"], name="Predicted vs Observed LAI", mode='markers')) 
    fig_pred_obs_a.update_layout(xaxis=dict(title="Scenario A: Ensemble Median LAI"), yaxis=dict(title="Observed LAI"), showlegend=True)

    residuals = df["LAI"] - median
    fig_residuals_a = go.Figure()
    fig_residuals_a.add_trace(go.Scatter(x=median, y=residuals, name="Predicted LAI", mode='markers'))        
    fig_residuals_a.update_layout(xaxis=dict(title="Scenario A: Residuals"), showlegend=True)

    metrics = []
    metric_list = ["mean_squared_error", "mean_absolute_error", "r2_score", "mean_pinball_loss", "mean_absolute_percentage_error"]
    for metric in metric_list:
        metric_func = getattr(skmetrics, metric)
        metric_value = metric_func(median, df["LAI"])
        metrics.append({"scenario": "A", "metric": metric, "value": metric_value})

    median, lower, upper, lower_q, upper_q = get_quantiles(LAI_b, alpha)
    fig_pi_lai_b = go.Figure()
    fig_pi_lai_b.add_trace(go.Scatter(x=df["day"], y=df["LAI"], name="Observed LAI")) 
    fig_pi_lai_b.add_trace(go.Scatter(x=df["day"], y=median, name="Ensemble median LAI"))  
    fig_pi_lai_b.add_trace(go.Scatter(x=df["day"], y=lower, name=f"Lower quantile LAI ({round(lower_q, 2)})"))  
    fig_pi_lai_b.add_trace(go.Scatter(x=df["day"], y=upper, name=f"Upper quantile LAI ({round(upper_q, 2)})"))        
    fig_pi_lai_b.update_layout(xaxis=dict(title=f"Scenario B: Quantile LAI (alpha={alpha})"), showlegend=True)

    fig_pred_obs_b = go.Figure()
    fig_pred_obs_b.add_trace(go.Scatter(x=median, y=df["LAI"], name="Predicted vs Observed LAI", mode='markers')) 
    fig_pred_obs_b.update_layout(xaxis=dict(title="Scenario B: Ensemble Median LAI"), yaxis=dict(title="Observed LAI"), showlegend=True)

    residuals = df["LAI"] - median
    fig_residuals_b = go.Figure()
    fig_residuals_b.add_trace(go.Scatter(x=median, y=residuals, name="Predicted LAI", mode='markers'))        
    fig_residuals_b.update_layout(xaxis=dict(title="Scenario B: Residuals"), showlegend=True)

    for metric in metric_list:
        metric_func = getattr(skmetrics, metric)
        metric_value = metric_func(median, df["LAI"])
        metrics.append({"scenario": "B", "metric": metric, "value": metric_value})

    return fig_lai_a, fig_lai_b, fig_pi_lai_a, fig_pi_lai_b, fig_pred_obs_a, fig_pred_obs_b, fig_residuals_a, fig_residuals_b, fig_span_a, fig_span_b, metrics

@callback(
    Output('lai-scatter-a-plot', 'figure', allow_duplicate=True),
    Output('lai-scatter-b-plot', 'figure', allow_duplicate=True),
    Output('lai-pi-scatter-a-plot', 'figure', allow_duplicate=True),
    Output('lai-pi-scatter-b-plot', 'figure', allow_duplicate=True),
    Output('lai-pred-obs-scatter-a-plot', 'figure', allow_duplicate=True),
    Output('lai-pred-obs-scatter-b-plot', 'figure', allow_duplicate=True),
    Output('lai-residual-scatter-a-plot', 'figure', allow_duplicate=True),
    Output('lai-residual-scatter-b-plot', 'figure', allow_duplicate=True),
    Output('span-hist-a-plot', 'figure', allow_duplicate=True),
    Output('span-hist-b-plot', 'figure', allow_duplicate=True),
    Output('what-if-metrics-table', 'data', allow_duplicate=True),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True
)
def reset_ensemble(n):
    if n == 0:
        return no_update
    
    lai_a, lai_b, lai_pi_a, lai_pi_b = go.Figure(), go.Figure(), go.Figure(), go.Figure()
    span_a, span_b, pred_obs_a, pred_obs_b = go.Figure(), go.Figure(), go.Figure(), go.Figure()
    residual_a, residual_b = go.Figure(), go.Figure()
    return lai_a, lai_b, lai_pi_a, lai_pi_b, pred_obs_a, pred_obs_b, residual_a, residual_b, span_a, span_b, []
