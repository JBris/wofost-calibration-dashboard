from dash import dcc, html, Input, Output, register_page, callback, no_update, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from dataproviders import parameters, agromanagement, weather
import copy
from pcse.models import Wofost72_WLP_FD
import iterative_ensemble_smoother as ies
import numpy as np
import plotly.graph_objects as go
from iterative_ensemble_smoother.utils import steplength_exponential

register_page(__name__, path="/", name="Forecasting")

state = {}

def get_obs_data():
    true_params = {}
    true_params["TDWI"] = 160
    true_params["WAV"] = 5
    true_params["SPAN"] = 33
    true_params["SMFCF"] = .33

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
            html.H4("Parameter Distributions", style={'textAlign': 'center', "padding": "0.5em"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Initial total crop dry weight mean", style={"padding-left": "2em"}),
                    dbc.Tooltip("The mean for initial total crop dry weight", target="input-tdwi-mean"),
                    dcc.Input(
                        id="input-tdwi-mean",
                        type="number",
                        value=150,
                        placeholder="TDWI Mean",
                        style={"width": "65%", "margin-left": "2em"}
                    ),
                ]),
                dbc.Col([
                    html.Label("Initial total crop dry weight standard deviation", style={"padding-left": "2em"}),
                    dbc.Tooltip("The standard deviation for initial total crop dry weight", target="input-tdwi-sd"),
                    dcc.Input(
                        id="input-tdwi-sd",
                        type="number",
                        value=50,
                        placeholder="TDWI Standard Deviation",
                        style={"width": "65%", "margin-left": "2em"}
                    ),
                ])
            ]),

            dbc.Row([
                dbc.Col([
                    html.Label("Water availability factor mean", style={"padding-left": "2em"}),
                    dbc.Tooltip("The mean for water availability factor", target="input-wav-mean"),
                    dcc.Input(
                        id="input-wav-mean",
                        type="number",
                        value=4.5,
                        placeholder="WAV Mean",
                        style={"width": "65%", "margin-left": "2em"}
                    ),
                ]),
                dbc.Col([
                    html.Label("Water availability factor standard deviation", style={"padding-left": "2em"}),
                    dbc.Tooltip("The standard deviation for water availability factor", target="input-wav-sd"),
                    dcc.Input(
                        id="input-wav-sd",
                        type="number",
                        value=1.5,
                        placeholder="WAV Standard Deviation",
                        style={"width": "65%", "margin-left": "2em"}
                    ),
                ])
            ]),

            dbc.Row([
                dbc.Col([
                    html.Label("Life span of leaves growing at 35°C mean", style={"padding-left": "2em"}),
                    dbc.Tooltip("The mean for life span of leaves growing at 35°C", target="input-span-mean"),
                    dcc.Input(
                        id="input-span-mean",
                        type="number",
                        value=31,
                        placeholder="SPAN Mean",
                        style={"width": "65%", "margin-left": "2em"}
                    ),
                ]),
                dbc.Col([
                    html.Label("Life span of leaves growing at 35°C standard deviation", style={"padding-left": "2em"}),
                    dbc.Tooltip("The standard deviation for life span of leaves growing at 35°C", target="input-span-sd"),
                    dcc.Input(
                        id="input-span-sd",
                        type="number",
                        value=3,
                        placeholder="SPAN Standard Deviation",
                        style={"width": "65%", "margin-left": "2em"}
                    ),
                ])
            ]),


            dbc.Row([
                dbc.Col([
                html.Label("Soil Moisture Content at Field Capacity mean", style={"padding-left": "2em"}),
                dbc.Tooltip("The mean for Soil Moisture Content at Field Capacity", target="input-smfcf-mean"),
                dcc.Input(
                    id="input-smfcf-mean",
                    type="number",
                    value=0.31,
                    placeholder="SMFCF Mean",
                    style={"margin-left": "2em"}
                ),
                ]),
                dbc.Col([
                    html.Label("Soil Moisture Content at Field Capacity standard deviation", style={"padding-left": "2em"}),
                    dbc.Tooltip("The standard deviation for Soil Moisture Content at Field Capacity", target="input-smfcf-sd"),
                    dcc.Input(
                        id="input-smfcf-sd",
                        type="number",
                        value=0.03,
                        placeholder="SMFCF Standard Deviation",
                        style={"margin-left": "2em"}
                    ),
                ])
            ], 
            style={"padding-bottom": "2em"}),
        ]),
        dbc.Card([
            html.H4("Calibration Procedure", style={'textAlign': 'center', "padding": "0.5em"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Ensemble size", style={"padding-left": "2em"}),
                    dbc.Tooltip("The number of ensemble members", target="input-ensemble-size"),
                    dcc.Input(
                        id="input-ensemble-size",
                        type="number",
                        value=10,
                        placeholder="Ensemble Size",
                        style={"margin-left": "2em"}
                    ),
                ]),
                dbc.Col([
                    html.Label("Number of iterations", style={"padding-left": "2em"}),
                    dbc.Tooltip("The number of history matching iterations", target="input-number-iterations"),
                    dcc.Input(
                        id="input-number-iterations",
                        type="number",
                        value=1,
                        placeholder="Number of Iterations",
                        style={"margin-left": "2em", "margin-bottom": "2em"}
                    ),
                ])   
            ]),

            dbc.Row([
                dbc.Col(
                [
                    dbc.Tooltip("Run the history matching procedure", target="btn-run"),
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
        ]),
    ]),
    dbc.Col([
        dbc.Row(dcc.Graph(id='lai-scatter-plot')),
        dbc.Row(dcc.Graph(id='tdwi-hist-plot')),
        dbc.Row(dcc.Graph(id='wav-hist-plot')),
        dbc.Row(dcc.Graph(id='span-hist-plot')),
        dbc.Row(dcc.Graph(id='smfcf-hist-plot')),

    ]),
])

def run_smoother(Y_IES_ert, n_ies_iter, smoother_ies, override_parameters):
    for i in range(n_ies_iter):
        step_length = steplength_exponential(i + 1)
        X_IES_ert = smoother_ies.sies_iteration(Y_IES_ert, step_length=step_length)

        IES_ensemble = []
        for j in range(X_IES_ert.shape[1]):
            p = copy.deepcopy(parameters)
            for k, par in enumerate(override_parameters.keys()):
                p.set_override(par, X_IES_ert[k, j])
            IES_member = Wofost72_WLP_FD(p, weather, agromanagement)
            IES_member.run_till_terminate()
            IES_ensemble.append(IES_member)

        Y_IES_ert_dfs = [pd.DataFrame(member.get_output()).set_index("day") for member in IES_ensemble]
        Y_IES_ert = np.array([
            Y_IES_ert_df["LAI"].values
            for Y_IES_ert_df in Y_IES_ert_dfs
        ]).T

    return X_IES_ert, Y_IES_ert

def plot_distributions(override_parameters):
    fig_tdwi = go.Figure()  
    fig_tdwi.add_trace(go.Histogram(x=override_parameters["TDWI"], name = "Prior TDWI"))           
    fig_tdwi.update_layout(xaxis=dict(title="TDWI"), showlegend=True)

    fig_wav = go.Figure()      
    fig_wav.add_trace(go.Histogram(x=override_parameters["WAV"], name = "Prior WAV"))              
    fig_wav.update_layout(xaxis=dict(title="WAV"), showlegend=True)

    fig_span = go.Figure()    
    fig_span.add_trace(go.Histogram(x=override_parameters["SPAN"], name = "Prior SPAN"))        
    fig_span.update_layout(xaxis=dict(title="SPAN"), showlegend=True)

    fig_smfcf = go.Figure()     
    fig_smfcf.add_trace(go.Histogram(x=override_parameters["SMFCF"], name = "Prior SMFCF"))        
    fig_smfcf.update_layout(xaxis=dict(title="SMFCF"), showlegend=True)

    return fig_tdwi, fig_wav, fig_span, fig_smfcf

@callback(
    Output('tdwi-hist-plot', 'figure', allow_duplicate=True),
    Output('wav-hist-plot', 'figure', allow_duplicate=True),
    Output('span-hist-plot', 'figure', allow_duplicate=True),
    Output('smfcf-hist-plot', 'figure', allow_duplicate=True),
    Output('lai-scatter-plot', 'figure', allow_duplicate=True),
    Output('store-ensemble-data', 'data', allow_duplicate=True),
    Input("btn-run", "n_clicks"),
    State("input-tdwi-mean", "value"),
    State("input-tdwi-sd", "value"),
    State("input-wav-mean", "value"),
    State("input-wav-sd", "value"),
    State("input-span-mean", "value"),
    State("input-span-sd", "value"),
    State("input-smfcf-mean", "value"),
    State("input-smfcf-sd", "value"),
    State("input-ensemble-size", "value"),
    State("input-number-iterations", "value"),
    State("store-ensemble-data", "data"),
    prevent_initial_call=True
)
def run_ensemble(
    n, tdwi_mean, tdwi_sd, wav_mean, wav_sd, 
    span_mean, span_sd, smfcf_mean, smfcf_sd,  
    ensemble_size, num_iterations, ensemble_data
):
    if isinstance(ensemble_data, list):
        ensemble_data = {"num_iterations": 0}
    current_iteration = ensemble_data["num_iterations"]
    
    if current_iteration == 0:
        override_parameters = {}
        override_parameters["TDWI"] = np.random.normal(tdwi_mean, tdwi_sd, (ensemble_size))
        override_parameters["WAV"] = np.random.normal(wav_mean, wav_sd, (ensemble_size))
        override_parameters["SPAN"] = np.random.normal(span_mean, span_sd ,(ensemble_size))
        override_parameters["SMFCF"] = np.random.normal(smfcf_mean, smfcf_sd ,(ensemble_size))
        X_IES_ert = np.array([
            distr
            for distr in override_parameters.values()
        ])
        state["override_parameters"] = override_parameters
        state["smoother_ies"] = ies.SIES(
            parameters = X_IES_ert, covariance = np.eye(obs.shape[0]), observations = obs
        )

        fig_tdwi, fig_wav, fig_span, fig_smfcf = plot_distributions(override_parameters)

        ensemble = []
        for i in range(ensemble_size):
            p = copy.deepcopy(parameters)
            for par, distr in override_parameters.items():
                p.set_override(par, distr[i])
            member = Wofost72_WLP_FD(p, weather, agromanagement)
            member.run_till_terminate()
            ensemble.append(member)

        results = [pd.DataFrame(member.get_output()).set_index("day") for member in ensemble]
        Y_IES_ert = np.array([
            result["LAI"].values
            for result in results
        ]).T

        state["X_IES_ert"], state["Y_IES_ert"] = run_smoother(
            Y_IES_ert, num_iterations, state["smoother_ies"], override_parameters
        )
        ensemble_data["num_iterations"] += num_iterations
        num_iterations = ensemble_data["num_iterations"]

        fig_tdwi.add_trace(go.Histogram(x=state["X_IES_ert"][0, :], name=f"Posterior TDWI (Iteration {num_iterations})"))   
        fig_wav.add_trace(go.Histogram(x=state["X_IES_ert"][1, :], name=f"Posterior WAV (Iteration {num_iterations})"))    
        fig_span.add_trace(go.Histogram(x=state["X_IES_ert"][2, :], name=f"Posterior SPAN (Iteration {num_iterations})"))    
        fig_smfcf.add_trace(go.Histogram(x=state["X_IES_ert"][3, :], name=f"Posterior SMFCF (Iteration {num_iterations})"))   

        fig_lai = go.Figure()
        fig_lai.add_trace(go.Scatter(x=df["day"], y=df["LAI"], name="Observed LAI"))        
        fig_lai.update_layout(xaxis=dict(title="LAI"), showlegend=True)

        for i in range(ensemble_size):
            fig_lai.add_trace(go.Scatter(x=df["day"], y=state["Y_IES_ert"].T[i], showlegend=False))  

        return fig_tdwi, fig_wav, fig_span, fig_smfcf, fig_lai, ensemble_data
    elif current_iteration > 0:
        override_parameters = state["override_parameters"]
        fig_tdwi, fig_wav, fig_span, fig_smfcf = plot_distributions(override_parameters)
        state["X_IES_ert"], state["Y_IES_ert"] = run_smoother(
            state["Y_IES_ert"], num_iterations, state["smoother_ies"], override_parameters
        )
        ensemble_data["num_iterations"] += num_iterations
        num_iterations = ensemble_data["num_iterations"]

        fig_tdwi.add_trace(go.Histogram(x=state["X_IES_ert"][0, :], name=f"Posterior TDWI (Iteration {num_iterations})"))   
        fig_wav.add_trace(go.Histogram(x=state["X_IES_ert"][1, :], name=f"Posterior WAV (Iteration {num_iterations})"))    
        fig_span.add_trace(go.Histogram(x=state["X_IES_ert"][2, :], name=f"Posterior SPAN (Iteration {num_iterations})"))    
        fig_smfcf.add_trace(go.Histogram(x=state["X_IES_ert"][3, :], name=f"Posterior SMFCF (Iteration {num_iterations})"))   

        fig_lai = go.Figure()
        fig_lai.add_trace(go.Scatter(x=df["day"], y=df["LAI"], name="Observed LAI"))        
        fig_lai.update_layout(xaxis=dict(title="LAI"), showlegend=True)

        for i in range(ensemble_size):
            fig_lai.add_trace(go.Scatter(x=df["day"], y=state["Y_IES_ert"].T[i], showlegend=False))  

        return fig_tdwi, fig_wav, fig_span, fig_smfcf, fig_lai, ensemble_data
    
    return no_update

@callback(
    Output('tdwi-hist-plot', 'figure', allow_duplicate=True),
    Output('wav-hist-plot', 'figure', allow_duplicate=True),
    Output('span-hist-plot', 'figure', allow_duplicate=True),
    Output('smfcf-hist-plot', 'figure', allow_duplicate=True),
    Output('lai-scatter-plot', 'figure', allow_duplicate=True),
    Output('store-ensemble-data', 'data', allow_duplicate=True),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True
)
def reset_ensemble(n):
    if n == 0:
        return no_update
    
    fig_tdwi, fig_wav, fig_span, fig_smfcf = go.Figure(), go.Figure(), go.Figure(), go.Figure()
    fig_lai = go.Figure()

    ensemble_data = {"num_iterations": 0}
    return fig_tdwi, fig_wav, fig_span, fig_smfcf, fig_lai, ensemble_data
