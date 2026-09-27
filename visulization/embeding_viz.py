from dash import Dash, html, dcc, callback, Input, Output, State, Patch, no_update
import dash_bootstrap_components as dbc

from functools import lru_cache
import re
import shutil
import subprocess

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

MAX_SELECTED_CATEGORIES = 5
PANDOC_CACHE_SIZE = 2048
PANDOC_TIMEOUT_SECONDS = 3

# Resolve Pandoc once at startup instead of searching PATH after every click.
PANDOC_PATH = shutil.which("pandoc")

# Keep input cleanup deliberately small. Pandoc handles LaTeX structure; these
# patterns only remove empty math fragments that its Markdown writer preserves.
EMPTY_DISPLAY_MATH_PATTERN = re.compile(
    r"(?<!\\)(?<!\$)\$\$\s*\$\$(?!\$)",
    flags=re.DOTALL,
)
EMPTY_INLINE_MATH_PATTERN = re.compile(
    r"(?<!\\)(?<!\$)\$(?!\$)\s*\$(?!\$)",
    flags=re.DOTALL,
)
PANDOC_DISPLAY_MATH_PATTERN = re.compile(
    r"(?<!\\)\$\$(?P<math>.*?)(?<!\\)\$\$",
    flags=re.DOTALL,
)
PANDOC_INLINE_MATH_PATTERN = re.compile(
    r"(?<!\\)(?<!\$)\$(?!\$)(?P<math>.*?)(?<!\\)\$(?!\$)",
    flags=re.DOTALL,
)
MATHJAX_TEXT_MACRO_PATTERN = re.compile(
    r"\\(?:text(?:normal|up|rm|it|bf|sf|tt)?|emph)\s*\{"
)
MATHJAX_TEXTMACROS_REQUIRE = r"\require{textmacros}"
MATHJAX_ENSUREMATH_PATTERN = re.compile(
    r"(?<!\\)\\ensuremath(?![A-Za-z@])"
)

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

if PANDOC_PATH is None:
    print("warning: Pandoc was not found; LaTeX previews will use raw source")

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


def get_category_options(select_categories):
    selected = set(select_categories or [])
    limit_reached = len(selected) >= MAX_SELECTED_CATEGORIES

    # Keep selected categories removable while preventing any additional choice.
    return [
        {
            "label": category,
            "value": category,
            "disabled": limit_reached and category not in selected,
        }
        for category in CATEGORIES
    ]


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


def clean_latex_for_pandoc(raw_text):
    """Remove only empty math spans before passing the fragment to Pandoc."""
    cleaned = EMPTY_DISPLAY_MATH_PATTERN.sub("", raw_text)
    cleaned = EMPTY_INLINE_MATH_PATTERN.sub("", cleaned)
    return cleaned.strip()


def prepare_math_for_mathjax(math_source):
    """Normalize LaTeX wrappers and load MathJax text-mode macros as needed."""
    # The source is already inside MathJax delimiters, so \ensuremath is an
    # identity wrapper. Removing only the command keeps its braced argument and
    # grouping unchanged without requiring unsupported LaTeX document macros.
    math_source = MATHJAX_ENSUREMATH_PATTERN.sub("", math_source.strip())

    # Dash loads MathJax 3's compact tex-svg component. Explicitly request the
    # textmacros extension so nested forms such as \textsf{\textit{D}} and
    # \textnormal{\textrm{R}} keep both formatting levels.
    if (
        MATHJAX_TEXT_MACRO_PATTERN.search(math_source)
        and MATHJAX_TEXTMACROS_REQUIRE not in math_source
    ):
        return f"{MATHJAX_TEXTMACROS_REQUIRE} {math_source}"

    return math_source


@lru_cache(maxsize=PANDOC_CACHE_SIZE)
def convert_latex_to_markdown(raw_text):
    """Convert one LaTeX fragment to MathJax-compatible CommonMark."""
    cleaned_source = clean_latex_for_pandoc(raw_text)

    if not cleaned_source:
        return "", None

    if PANDOC_PATH is None:
        return (
            cleaned_source,
            "Pandoc is not installed, so this preview is showing the original LaTeX.",
        )

    command = [
        PANDOC_PATH,
        "--sandbox",
        "--from=latex+raw_tex",
        "--to=commonmark_x-raw_html-fenced_divs",
        "--wrap=none",
    ]

    try:
        result = subprocess.run(
            command,
            input=cleaned_source,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=PANDOC_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (
            cleaned_source,
            "Pandoc timed out, so this preview is showing the original LaTeX.",
        )
    except OSError as error:
        return (
            cleaned_source,
            f"Pandoc could not be started ({error}), so the original LaTeX is shown.",
        )

    if result.returncode != 0:
        error_message = result.stderr.strip() or "unknown Pandoc error"
        return (
            cleaned_source,
            f"Pandoc conversion failed ({error_message}); the original LaTeX is shown.",
        )

    # Pandoc preserves unknown text macros as raw-LaTeX code spans. Remove its
    # writer-specific attribute while keeping the unsupported command visible.
    markdown = result.stdout.replace("{=latex}", "").strip()
    markdown = PANDOC_DISPLAY_MATH_PATTERN.sub(
        lambda match: (
            f"\n\n$$\n{prepare_math_for_mathjax(match.group('math'))}\n$$\n\n"
        ),
        markdown,
    )
    markdown = PANDOC_INLINE_MATH_PATTERN.sub(
        lambda match: f"${prepare_math_for_mathjax(match.group('math'))}$",
        markdown,
    )
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return markdown, None


app = Dash("embedding space", external_stylesheets=[dbc.themes.BOOTSTRAP])

category_select = html.Div(
    [
        dbc.Label("Select Categories (up to 5)"),  # type: ignore
        dcc.Dropdown(
            id="categories",
            options=get_category_options([]),
            multi=True,
            value=[],
        ),
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

texts = dbc.Card(
    [
        dbc.CardHeader("Selected statement"),
        dbc.CardBody(
            [
                html.Div(
                    "Click a point to inspect its statement.",
                    id="statement_metadata",
                    className="text-muted mb-3",
                ),
                dbc.Alert(
                    id="latex_warning",
                    color="warning",
                    is_open=False,
                    className="py-2",
                ),
                dcc.Loading(
                    dcc.Markdown(
                        id="statement_latex",
                        children="",
                        mathjax=True,
                        link_target="_blank",
                        style={
                            "overflowX": "auto",
                            "minHeight": "8rem",
                            "padding": "0.75rem",
                            "backgroundColor": "rgba(248, 249, 250, 0.75)",
                            "borderRadius": "0.375rem",
                        },
                    ),
                    type="default",
                ),
                html.Details(
                    [
                        html.Summary("Raw LaTeX", className="fw-semibold"),
                        html.Pre(
                            id="statement_raw",
                            children="",
                            style={
                                "whiteSpace": "pre-wrap",
                                "overflowWrap": "anywhere",
                                "marginTop": "0.75rem",
                                "padding": "0.75rem",
                                "backgroundColor": "#f8f9fa",
                                "border": "1px solid #dee2e6",
                                "borderRadius": "0.375rem",
                            },
                        ),
                    ],
                    className="mt-3",
                ),
            ]
        ),
    ],
    className="h-100",
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
    Output("categories", "options"),
    Output("categories", "value"),
    Input("categories", "value"),
)
def enforce_category_limit(select_categories):
    # Deduplicate defensively and reject any selection beyond the configured limit.
    unique_categories = list(dict.fromkeys(select_categories or []))
    limited_categories = unique_categories[:MAX_SELECTED_CATEGORIES]
    options = get_category_options(limited_categories)

    if unique_categories != limited_categories:
        return options, limited_categories

    return options, no_update


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
    Output("statement_metadata", "children"),
    Output("statement_latex", "children"),
    Output("latex_warning", "children"),
    Output("latex_warning", "is_open"),
    Output("statement_raw", "children"),
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
        return (no_update,) * 6

    point = click_data["points"][0]
    curve_number = point.get("curveNumber")
    point_number = point.get("pointNumber")

    if curve_number is None or point_number is None:
        return (no_update,) * 6

    curve_number = int(curve_number)
    point_number = int(point_number)
    selected_codes = get_category_codes(select_categories)

    # curveNumber identifies the fixed-order category trace; the final trace is
    # reserved for highlighting and must never be used for point lookup.
    if curve_number < 0 or curve_number >= len(selected_codes):
        return (no_update,) * 6

    category = selected_codes[curve_number]
    filter_type = select_type if select_type and select_type != "All" else "All"
    point_ids = POINT_IDS_BY_FILTER.get((category, filter_type))

    # pointNumber indexes the same original-row order used to build each trace.
    if point_ids is None or point_number < 0 or point_number >= len(point_ids):
        return (no_update,) * 6

    point_id = int(point_ids[point_number])
    row = meta_df.loc[point_id]
    raw_text = "" if pd.isna(row["text"]) else str(row["text"]) # type: ignore
    markdown_preview, conversion_warning = convert_latex_to_markdown(raw_text)

    metadata = html.Div(
        [
            html.Div([html.Strong("Paper ID: "), str(row["paper_id"])]),
            html.Div([html.Strong("Type: "), str(row["type"])]),
            html.Div([html.Strong("Category: "), str(row["categories"])]),
        ]
    )

    # Patch only the final one-point trace; the base point cloud is not retransmitted.
    patched_figure = Patch()
    patched_figure["data"][highlight_trace_index]["x"] = [point["x"]]
    patched_figure["data"][highlight_trace_index]["y"] = [point["y"]]

    return (
        metadata,
        markdown_preview,
        conversion_warning or "",
        conversion_warning is not None,
        raw_text,
        patched_figure,
    )


app.run(debug=True)
