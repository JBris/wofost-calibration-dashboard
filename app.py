import dash
from dash import Dash, html, dcc, page_registry, page_container
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

load_figure_template("MINTY")
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.MINTY],
    use_pages=True,
    title="WOFOST Calibration",
)


def define_ui(app: Dash) -> None:
    nav_links = [
        dbc.NavLink(page["name"], href=page["path"], active="exact")
        for page in page_registry.values()
    ]

    app_navbar = dbc.Row(
        dbc.Nav(
            nav_links,
            vertical=False,
            pills=True,
            style={"padding-left": "1em", "padding-top": "0.5em"},
        )
    )
    
    app.layout = html.Div(
        [
            dcc.Store(id="store-ensemble-data", storage_type="local", data=[]),
            dcc.Loading(
                id="loading_page_content",
                children=[
                    html.Div(
                        dbc.Row(
                            [
                                dbc.Card(
                                    app_navbar,
                                    style={
                                        "padding-bottom": "0.5em",
                                        "borderRadius": "0",
                                    },
                                ),
                                page_container,
                            ],
                            style={"width": "100%"}
                        ),
                        id="app-contents",
                    ),
                ],
                color="primary",
                fullscreen=False,
                delay_show=500,
                delay_hide=500,
                type="circle",
            ),
        ],
        id="app-wrapper",
        style={
            # "position": "fixed",
            "top": "0",
        }
    )

server = app.server

define_ui(app)

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8000, debug=True)