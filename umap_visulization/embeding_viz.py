from dash import Dash, html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


"""File to Visualize"""
DATA = r"embedding_data_clean_v1.csv"

"""Missing math.IT and math.MP"""
CATAGORIES = [
    "math.AC (Commutative Algebra)",
    "math.AG (Algebraic Geometry)",
    "math.AP (Analysis of PDEs)",
    "math.AT (Algebraic Topology)",
    "math.CA (Classical Analysis and ODEs)",
    "math.CO (Combinatorics)",
    "math.CT (Category Theory)",
    "math.CV (Complex Variables)",
    "math.DG (Differential Geometry)",
    "math.DS (Dynamical Systems)",
    "math.FA (Functional Analysis)",
    "math.GM (General Mathematics)",
    "math.GN (General Topology)",
    "math.GR (Group Theory)",
    "math.GT (Geometric Topology)",
    "math.HO (History and Overview)",
    "math.KT (K-Theory and Homology)",
    "math.LO (Logic)",
    "math.MG (Metric Geometry)math.NA (Numerical Analysis)",
    "math.NT (Number Theory)",
    "math.OA (Operator Algebras)",
    "math.OC (Optimization and Control)",
    "math.PR (Probability)",
    "math.QA (Quantum Algebra)",
    "math.RA (Rings and Algebras)",
    "math.RT (Representation Theory)",
    "math.SG (Symplectic Geometry)",
    "math.SP (Spectral Theory)",
    "math.ST (Statistics Theory)"
]

TYPES = [
    "All",
    "Theorem",
    "Definition",
    "Lemma",
    "Proposition",
    "Corollary",
    "Conjecture",
    "Claim",
]

def get_filtered_df(meta_df: pd.DataFrame, select_cate, select_type):
    cate = [c[:7] for c in select_cate]
    filtered_df = meta_df[meta_df['categories'].isin(cate)]
    if select_type and select_type != 'All':
        filtered_df = filtered_df[filtered_df['type'] == select_type]
    filtered_df.drop(columns='text', inplace=True)
    return filtered_df

def scatter_plot(filtered_df):
    return px.scatter(filtered_df, x='x', y='y', color='categories', custom_data=[filtered_df.index])

meta_df = pd.read_csv(DATA)
print("df loaded")

app = Dash("embeding space", external_stylesheets=[dbc.themes.BOOTSTRAP])

category_select = html.Div(
    [
        dbc.Label("Select Catagories"),
        dcc.Dropdown(id="categories", options=CATAGORIES, multi=True, value=[])
    ]
)

type_select = html.Div(
    [
        dbc.Label("Select Statement Types"),
        dcc.Dropdown(id="types", options=TYPES, value='All')
    ]
)

scatter = html.Div(
    [
        dbc.Label("Scatter plot"),
        dcc.Graph(id="scatter_plot", figure=None)
    ]
)

texts = html.Div(
    [
        dbc.Label("Select statement"),
        dbc.Textarea(id="text", value="")
    ]
)

controls = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(category_select, md=8),
                dbc.Col(type_select, md=4)
            ]
        )
    ]
)

graphs = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(scatter, lg=8),
                dbc.Col(texts, lg=4)
            ]
        ),
        dbc.Label("*Tips: Start with select categories, click on points to show more details", 
                color='Gray', style={"background": "rgba(192, 255, 255, 0.5)"})
    ]
)


about = html.Div(
    html.P("#TODO")
)

body = html.Div(
[
    html.H1("Embedding Space", style={"text-align": "center"}),
    html.H2("Scatter Plot"),
    html.Hr(),
    controls,
    graphs,
    html.H2("About"),
    html.Hr(),
    about
], 
className="bd-content ps-lg-4", 
style={
    "width": "90%",
    "background": "rgba(255, 255, 255, 0.3)",  # semi-transparent
    "backdropFilter": "blur(4px)",
    "border": "1px solid rgba(255, 255, 255, 0.3)",
    "borderRadius": "10px",
    "padding": "20px",
    "box-shadow": "0 4px 30px rgba(0, 0, 0, 0.1)",
    }
)

app.layout=html.Div(
    [
        body,
    ],
    # centering the body
    style={
        "display": "flex",
        "justify-content":"center",

    }
)

@callback(
    Output("scatter_plot", "figure"),

    Input("categories", "value"),
    Input("types", "value"),
    
    State("scatter_plot", "figure")
)
def update_scatter(select_cate, select_type, fig):
    global meta_df
    if select_cate:
        return scatter_plot(get_filtered_df(meta_df, select_cate, select_type))
    else:
        return fig


@callback(
    Output("text", "value"),

    Input("scatter_plot", "clickData"),

    State("scatter_plot", "figure"),
    prevent_initial_call=True
)
def display_text(clickData, fig):
    if fig:
        if clickData:
            # print(clickData)
            # print()
            # print(fig['data'])
            # print()
            index: int = fig['data'][clickData['points'][0]['curveNumber']]['customdata']['_inputArray'][clickData['points'][0]['pointIndex']]['0']
            row = meta_df.iloc[index]
            info =  f"paper id: {row['paper_id']}\ntype: {row['type']}\ncategory: {row['categories']}\n\n{row['text']}"
            return info
        

app.run(debug=True)