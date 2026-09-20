from dash import Dash, html, dcc, callback, Input, Output, State, Patch, no_update
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# CSV File to visualize
# DATA = r"C:\E\works\CC\4\BlockA\RAG\data\plot_data\tsne\theorems_Qwen3-Embedding-0.6B_new_ALL_50k_all_tsne.csv"
# DATA = r"C:\E\works\CC\4\BlockA\RAG\data\plot_data\tsne\theorems_Qwen3-Embedding-0.6B_ALL_50k_only6_tsne_projection_clean.csv"
DATA = r"C:\E\works\CC\4\BlockA\RAG\data\plot_data\tsne\theorems_Qwen3-Embedding-0.6B_new_base_all_tsne.csv"

CATEGORIES = [
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
    "math.MG (Metric Geometry)",
    "math.NA (Numerical Analysis)",
    "math.NT (Number Theory)",
    "math.OA (Operator Algebras)",
    "math.OC (Optimization and Control)",
    "math.PR (Probability)",
    "math.QA (Quantum Algebra)",
    "math.RA (Rings and Algebras)",
    "math.RT (Representation Theory)",
    "math.SG (Symplectic Geometry)",
    "math.SP (Spectral Theory)",
    "math.ST (Statistics Theory)",
]

# Use one global category order and color mapping across every filter result.
CATEGORY_CODES = [label.split(" ", 1)[0] for label in CATEGORIES]
CATEGORY_COLORS = {
    code: color
    for code, color in zip(
        CATEGORY_CODES,
        px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24,
    )
}

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

meta_df = pd.read_csv(DATA)

# Preserve a stable identifier before any filtering or reindexing occurs.
meta_df["point_id"] = np.arange(len(meta_df), dtype=np.int32)
meta_df.set_index("point_id", drop=False, inplace=True)

print("df loaded")

def build_point_id_lookup(meta_df: pd.DataFrame):
    point_id_lookup = {}

    # Cache compact server-side ID arrays once so click callbacks do not need to
    # rescan the full DataFrame or send point IDs to the browser with the figure.
    for category, category_df in meta_df.groupby(
        "categories",
        sort=False,
        observed=True,
    ):
        point_id_lookup[(category, "All")] = category_df[
            "point_id"
        ].to_numpy(dtype=np.int32, copy=True)

        for statement_type, type_df in category_df.groupby(
            "type",
            sort=False,
            observed=True,
        ):
            # Copy only the int32 IDs so temporary grouped DataFrames can be released.
            point_id_lookup[(category, statement_type)] = type_df[
                "point_id"
            ].to_numpy(dtype=np.int32, copy=True)

    return point_id_lookup


POINT_IDS_BY_FILTER = build_point_id_lookup(meta_df)

print("point ids lookup built")

max_x = ((meta_df["x"].max() // 5) + 1) * 5
max_y = ((meta_df["y"].max() // 5) + 1) * 5
min_x = (meta_df["x"].min() // 5) * 5
min_y = (meta_df["y"].min() // 5) * 5


def get_category_codes(select_categories):
    selected = {
        category.split(" ", 1)[0]
        for category in (select_categories or [])
    }

    # Follow the global order instead of the dropdown selection or row order.
    return [code for code in CATEGORY_CODES if code in selected]


def get_filtered_df(meta_df: pd.DataFrame, category_codes, select_type):
    mask = meta_df["categories"].isin(category_codes)

    if select_type and select_type != "All":
        mask &= meta_df["type"].eq(select_type)

    # Carry only the columns required to construct the WebGL traces.
    return meta_df.loc[
        mask,
        ["x", "y", "categories"],
    ].copy()


def apply_common_layout(figure):
    figure.update_xaxes(range=[min_x, max_x])
    figure.update_yaxes(range=[min_y, max_y])
    figure.update_layout(uirevision="embedding-space")
    return figure


def empty_scatter_plot():
    return apply_common_layout(go.Figure())


def create_highlight_trace():
    return go.Scattergl(
        x=[],
        y=[],
        mode="markers",
        name="Selected point",
        showlegend=False,
        hoverinfo="skip",
        marker=dict(
            size=12,
            color="rgba(0, 0, 0, 0)",
            line=dict(color="black", width=2),
        ),
    )


def scatter_plot(filtered_df, selected_codes):
    # Group only once. Re-filtering 3.2 million rows for every category would be
    # substantially more expensive than constructing the Scattergl traces.
    grouped = {
        category: group
        for category, group in filtered_df.groupby(
            "categories",
            sort=False,
            observed=True,
        )
    }

    traces = []

    # Create traces in the global category order so their visual stacking remains
    # stable after applying a different statement-type filter.
    for category in selected_codes:
        category_df = grouped.get(category)

        if category_df is None:
            x_values = np.empty(0, dtype=np.float32)
            y_values = np.empty(0, dtype=np.float32)
        else:
            # Pass NumPy arrays directly to avoid creating millions of Python objects.
            x_values = category_df["x"].to_numpy(copy=False)
            y_values = category_df["y"].to_numpy(copy=False)

        traces.append(
            go.Scattergl(
                x=x_values,
                y=y_values,
                mode="markers",
                name=category,
                uid=f"category-{category}",
                marker=dict(color=CATEGORY_COLORS[category]),
            )
        )

    # Add the highlight last so it is rendered above every category trace.
    highlight_trace_index = len(traces)
    traces.append(create_highlight_trace())

    return apply_common_layout(go.Figure(data=traces)), highlight_trace_index

app = Dash("embedding space", external_stylesheets=[dbc.themes.BOOTSTRAP])

category_select = html.Div(
    [
        dbc.Label("Select Categories"),  # type: ignore
        dcc.Dropdown(id="categories", options=CATEGORIES, multi=True, value=[]),
    ]
)

type_select = html.Div(
    [
        dbc.Label("Select Statement Types"),  # type: ignore
        dcc.Dropdown(id="types", options=TYPES, value="All"),
    ]
)

scatter = html.Div(
    [
        dbc.Label("Scatter plot"),  # type: ignore
        dcc.Graph(
            id="scatter_plot",
            figure=empty_scatter_plot(),
            style={"height": "80%"},
        ),
    ]
)

texts = html.Div(
    [
        dbc.Label("Select statement"),  # type: ignore
        dbc.Textarea(id="text", value=""),  # type: ignore
    ]
)

controls = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(category_select, md=8),  # type: ignore
                dbc.Col(type_select, md=4),  # type: ignore
            ]
        )  # type: ignore
    ]
)

graphs = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(scatter, lg=6),  # type: ignore
                dbc.Col(texts, lg=6),  # type: ignore
            ]
        ),  # type: ignore
        dbc.Label(
            "*Tips: Start with select categories, click on points to show more details",
            color="Gray",
            style={"background": "rgba(192, 255, 255, 0.5)"},
        ),  # type: ignore
    ]
)

about = html.Div(html.P("#TODO"))

body = html.Div(
    [
        # Store only one small trace index instead of passing the full figure as State.
        dcc.Store(id="highlight_trace_index", data=None),
        html.H1("Embedding Space", style={"text-align": "center"}),
        html.H2("Scatter Plot"),
        html.Hr(),
        controls,
        graphs,
        html.H2("About"),
        html.Hr(),
        about,
    ],
    className="bd-content ps-lg-4",
    style={
        "width": "90%",
        "background": "rgba(255, 255, 255, 0.3)",
        "backdropFilter": "blur(4px)",
        "border": "1px solid rgba(255, 255, 255, 0.3)",
        "borderRadius": "10px",
        "padding": "20px",
        "box-shadow": "0 4px 30px rgba(0, 0, 0, 0.1)",
    },
)

app.layout = html.Div(
    [body],
    style={
        "display": "flex",
        "justify-content": "center",
    },
)


@callback(
    Output("scatter_plot", "figure"),
    Output("highlight_trace_index", "data"),
    Input("categories", "value"),
    Input("types", "value"),
)
def update_scatter(select_categories, select_type):
    selected_codes = get_category_codes(select_categories)

    if not selected_codes:
        return empty_scatter_plot(), None

    filtered_df = get_filtered_df(meta_df, selected_codes, select_type)
    return scatter_plot(filtered_df, selected_codes)


@callback(
    Output("text", "value"),
    Output("scatter_plot", "figure", allow_duplicate=True),
    Input("scatter_plot", "clickData"),
    State("highlight_trace_index", "data"),
    State("categories", "value"),
    State("types", "value"),
    prevent_initial_call=True,
)
def display_text(
    click_data,
    highlight_trace_index,
    select_categories,
    select_type,
):
    if (
        not click_data
        or not click_data.get("points")
        or highlight_trace_index is None
    ):
        return no_update, no_update

    point = click_data["points"][0]
    curve_number = point.get("curveNumber")
    point_number = point.get("pointNumber")

    if curve_number is None or point_number is None:
        return no_update, no_update

    curve_number = int(curve_number)
    point_number = int(point_number)
    selected_codes = get_category_codes(select_categories)

    # curveNumber identifies the fixed-order category trace; the final trace is
    # reserved for highlighting and must never be used for point lookup.
    if curve_number < 0 or curve_number >= len(selected_codes):
        return no_update, no_update

    category = selected_codes[curve_number]
    filter_type = select_type if select_type and select_type != "All" else "All"
    point_ids = POINT_IDS_BY_FILTER.get((category, filter_type))

    # pointNumber indexes the same original-row order used to build each trace.
    if point_ids is None or point_number < 0 or point_number >= len(point_ids):
        return no_update, no_update

    point_id = int(point_ids[point_number])
    row = meta_df.loc[point_id]
    info = (
        f"paper id: {row['paper_id']}\n"
        f"type: {row['type']}\n"
        f"category: {row['categories']}\n\n"
        f"{row['text']}"
    )

    # Patch only the final one-point trace; the base point cloud is not retransmitted.
    patched_figure = Patch()
    patched_figure["data"][highlight_trace_index]["x"] = [point["x"]]
    patched_figure["data"][highlight_trace_index]["y"] = [point["y"]]

    return info, patched_figure


app.run(debug=True)
