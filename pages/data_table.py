from dash import html, register_page, dash_table
import pandas as pd
from dataproviders import parameters, agromanagement, weather
import copy
from pcse.models import Wofost72_WLP_FD

register_page(__name__, path="/data", name="Data Table")

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

    return df

df = get_obs_data()

layout = html.Div([
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i, "editable": True} for i in df.columns],
        data=df.to_dict('records'),
        filter_action="native",    
        sort_action="native",      
        row_selectable="multi",   
        selected_rows=[],         
        page_action="native",     
        page_size=20,              
        style_table={
            'overflowX': 'auto',    
            'maxWidth': '100%',    
        },
        style_cell={
            'minWidth': '100px',   
            'width': '100px',       
            'maxWidth': '200px',    
            'whiteSpace': 'normal',
        }
    )
])