from dash import (
    Dash,
    html,
    dcc,
    callback,
    Input,
    Output,
    State,
    Patch,
    no_update,
    ctx,
)
import dash_bootstrap_components as dbc

import colorsys
from functools import lru_cache
import re
import shutil
import subprocess
import uuid

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# CSV File to visualize
DATA = r"C:\E\works\CC\4\BlockA\RAG\data\plot_data\tsne\theorems_Qwen3-Embedding-0.6B_new_ALL_50k_all_tsne.csv"
# DATA = r"C:\E\works\CC\4\BlockA\RAG\data\plot_data\tsne\theorems_Qwen3-Embedding-0.6B_ALL_50k_only6_tsne_projection_clean.csv"
# DATA = r"C:\E\works\CC\4\BlockA\RAG\data\plot_data\tsne\theorems_Qwen3-Embedding-0.6B_new_base_all_tsne.csv"

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
MAX_QUERY_PAPERS = 10
MAX_QUERY_HIGHLIGHT_POINTS = 50_000
MAX_QUERY_HISTORY = 20
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

# Paper colors are assigned only for the active query. This candidate pool is
# deliberately independent of paper IDs, since future queries may group points
# by citation distance instead of assigning a persistent color to each paper.
QUERY_COLOR_CANDIDATES = [
    "#{:02X}{:02X}{:02X}".format(
        *(
            round(channel * 255)
            for channel in colorsys.hsv_to_rgb(hue / 24, 0.90, 0.95)
        )
    )
    for hue in range(24)
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

# Read paper IDs as strings so exact identifiers and any leading zeros are kept.
meta_df = pd.read_csv(DATA, dtype={"paper_id": "string"})

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


def parse_paper_ids(raw_value):
    """Parse comma-, semicolon-, or whitespace-separated paper IDs."""
    paper_ids = re.split(r"[,;\s]+", raw_value or "")
    return list(dict.fromkeys(value.strip() for value in paper_ids if value.strip()))


def normalize_history_categories(select_categories):
    """Keep only valid dropdown values in the same stable global order."""
    selected = set(select_categories or [])
    return [
        category
        for category in CATEGORIES
        if category in selected
    ][:MAX_SELECTED_CATEGORIES]


def make_filter_history_record(
    select_categories,
    select_type,
    paper_ids,
    keyword,
):
    """Create a compact, JSON-serializable record for one successful query."""
    return {
        "id": uuid.uuid4().hex,
        "kind": "filter",
        "categories": normalize_history_categories(select_categories),
        "statement_type": select_type if select_type in TYPES else "All",
        "paper_ids": list(paper_ids),
        "keyword": keyword,
    }


def make_point_history_record(
    point_id,
    row,
    select_categories,
    select_type,
):
    """Create a compact record for one point inspected directly on the plot."""
    return {
        "id": uuid.uuid4().hex,
        "kind": "point",
        "point_id": int(point_id),
        "paper_id": "" if pd.isna(row["paper_id"]) else str(row["paper_id"]),
        "category": "" if pd.isna(row["categories"]) else str(row["categories"]),
        "statement_type": select_type if select_type in TYPES else "All",
        "point_statement_type": "" if pd.isna(row["type"]) else str(row["type"]),
        "categories": normalize_history_categories(select_categories),
    }


def history_record_key(record):
    """Return the semantic identity used to deduplicate history records."""
    if not isinstance(record, dict):
        return None

    kind = record.get("kind")
    categories = tuple(record.get("categories") or [])
    statement_type = record.get("statement_type", "All")

    if kind == "filter":
        return (
            kind,
            categories,
            statement_type,
            tuple(record.get("paper_ids") or []),
            record.get("keyword", ""),
        )

    if kind == "point":
        return (
            kind,
            categories,
            statement_type,
            record.get("point_id"),
        )

    return None


def add_history_record(history, new_record):
    """Prepend a record, moving an existing duplicate to the front."""
    new_key = history_record_key(new_record)
    existing_history = history if isinstance(history, list) else []
    if new_key is None:
        return existing_history[:MAX_QUERY_HISTORY]

    remaining = [
        record
        for record in existing_history
        if history_record_key(record) is not None
        and history_record_key(record) != new_key
    ]
    return [new_record, *remaining][:MAX_QUERY_HISTORY]


def history_record_label(record):
    """Build a concise dropdown label without including statement text."""
    if record.get("kind") == "filter":
        parts = []
        paper_ids = record.get("paper_ids") or []
        keyword = record.get("keyword", "")

        if paper_ids:
            parts.append("Papers: " + ", ".join(str(value) for value in paper_ids))
        if keyword:
            parts.append(f'Keyword: "{keyword}"')

        return "[Filter] " + " · ".join(parts)

    if record.get("kind") == "point":
        return (
            f"[Point] {record.get('paper_id', '')} · "
            f"{record.get('point_statement_type', '')} · "
            f"{record.get('category', '')}"
        )

    return "[Unknown history record]"


def color_to_rgb(color):
    """Convert the Plotly color formats used here to an RGB NumPy vector."""
    if color.startswith("#") and len(color) == 7:
        return np.array(
            [int(color[index:index + 2], 16) for index in (1, 3, 5)],
            dtype=np.float32,
        )

    match = re.fullmatch(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+).*\)", color)
    if match:
        return np.array([float(value) for value in match.groups()], dtype=np.float32)

    raise ValueError(f"Unsupported color format: {color}")


def select_query_colors(selected_codes, number_of_colors):
    """Choose query-local colors far from visible categories and each other."""
    available_colors = QUERY_COLOR_CANDIDATES.copy()
    reference_colors = [
        color_to_rgb(CATEGORY_COLORS[code])
        for code in selected_codes
    ]

    # Including the white page background discourages low-contrast highlights.
    reference_colors.append(color_to_rgb("#FFFFFF"))
    selected_colors = []

    for _ in range(min(number_of_colors, len(available_colors))):
        best_color = max(
            available_colors,
            key=lambda candidate: min(
                np.sum((color_to_rgb(candidate) - reference) ** 2)
                for reference in reference_colors
            ),
        )
        selected_colors.append(best_color)
        reference_colors.append(color_to_rgb(best_color))
        available_colors.remove(best_color)

    return selected_colors


def find_paper_points_in_view(selected_codes, select_type, paper_ids):
    """Return matching point IDs grouped by paper within the current plot view."""
    filter_type = select_type if select_type and select_type != "All" else "All"
    visible_groups = []

    # Reuse the same compact ID arrays that define the points in each trace.
    for category in selected_codes:
        point_ids = POINT_IDS_BY_FILTER.get((category, filter_type))
        if point_ids is None or len(point_ids) == 0:
            continue

        paper_values = meta_df.loc[point_ids, "paper_id"].to_numpy(
            dtype=object,
            na_value=None,# type: ignore
        ) 
        visible_groups.append((point_ids, paper_values))

    matches_by_paper = {}
    for paper_id in paper_ids:
        matching_parts = [
            point_ids[paper_values == paper_id]
            for point_ids, paper_values in visible_groups
        ]
        matching_parts = [part for part in matching_parts if len(part) > 0]

        if matching_parts:
            matches_by_paper[paper_id] = np.concatenate(matching_parts)
        else:
            matches_by_paper[paper_id] = np.empty(0, dtype=np.int32)

    return matches_by_paper


def filter_point_ids_by_keyword(point_ids, keyword):
    """Keep point IDs whose statement contains a literal keyword."""
    if len(point_ids) == 0:
        return np.empty(0, dtype=np.int32)

    text_values = meta_df.loc[point_ids, "text"]
    matching_mask = text_values.str.contains(
        keyword,
        case=False,
        regex=False,
        na=False,
    ).to_numpy(dtype=bool)
    return point_ids[matching_mask]


def find_keyword_points_in_view(selected_codes, select_type, keyword):
    """Return keyword matches from only the currently visible filters."""
    filter_type = select_type if select_type and select_type != "All" else "All"
    matching_parts = []

    for category in selected_codes:
        point_ids = POINT_IDS_BY_FILTER.get((category, filter_type))
        if point_ids is None:
            continue

        matching_point_ids = filter_point_ids_by_keyword(point_ids, keyword)
        if len(matching_point_ids) > 0:
            matching_parts.append(matching_point_ids)

    if not matching_parts:
        return np.empty(0, dtype=np.int32)

    return np.concatenate(matching_parts)


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
    figure.update_layout(
        uirevision="embedding-space",
        # Keep legend growth vertical so longer query labels cannot squeeze the
        # scatter plot's horizontal drawing area.
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=1,
            yanchor="bottom",
        ),
        # Leave just enough room for the horizontal legend. The previous
        # 80-pixel top margin plus an additional legend offset created a large
        # empty band above the plotting area.
        margin=dict(r=20, t=45, b=40),
    )
    return figure


def empty_scatter_plot():
    return apply_common_layout(go.Figure())


def create_paper_query_trace(slot):
    return go.Scattergl(
        x=[],
        y=[],
        customdata=[],
        mode="markers",
        name=f"Paper query {slot + 1}",
        uid=f"paper-query-{slot}",
        showlegend=False,
        marker=dict(
            size=11,
            color="#000000",
            opacity=1,
            line=dict(color="white", width=1.5),
        ),
        hovertemplate="%{fullData.name}<extra></extra>",
    )


def create_clicked_point_trace():
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

    # Reserve a bounded number of query traces so paper searches can update only
    # small overlays instead of retransmitting the multi-million-point base plot.
    query_trace_start = len(traces)
    traces.extend(
        create_paper_query_trace(slot)
        for slot in range(MAX_QUERY_PAPERS)
    )

    # Keep the clicked-point outline last so it remains above every query group.
    clicked_point_trace_index = len(traces)
    traces.append(create_clicked_point_trace())

    trace_indices = {
        "query_start": query_trace_start,
        "clicked_point": clicked_point_trace_index,
    }

    return apply_common_layout(go.Figure(data=traces)), trace_indices


def build_query_highlight_patch(
    selected_codes,
    select_type,
    paper_ids,
    keyword,
    trace_indices,
):
    """Build one bounded overlay update for a search or history replay."""
    missing_papers = []
    query_groups = []

    if paper_ids:
        matches_by_paper = find_paper_points_in_view(
            selected_codes,
            select_type,
            paper_ids,
        )

        # When both filters are present, apply the keyword only to the much
        # smaller per-paper results instead of rescanning the full view.
        if keyword:
            matches_by_paper = {
                paper_id: filter_point_ids_by_keyword(point_ids, keyword)
                for paper_id, point_ids in matches_by_paper.items()
            }

        matched_papers = [
            paper_id
            for paper_id in paper_ids
            if len(matches_by_paper[paper_id]) > 0
        ]
        missing_papers = [
            paper_id
            for paper_id in paper_ids
            if len(matches_by_paper[paper_id]) == 0
        ]
        query_groups = [
            (f"Paper {paper_id}", matches_by_paper[paper_id])
            for paper_id in matched_papers
        ]
    else:
        keyword_point_ids = find_keyword_points_in_view(
            selected_codes,
            select_type,
            keyword,
        )
        if len(keyword_point_ids) > 0:
            # The full term already appears in the input and status message;
            # keep the legend compact so it remains useful in narrow layouts.
            query_groups = [("Keyword match", keyword_point_ids)]

    total_matches = sum(len(point_ids) for _, point_ids in query_groups)
    too_many_matches = total_matches > MAX_QUERY_HIGHLIGHT_POINTS
    rendered_groups = [] if too_many_matches else query_groups
    query_colors = select_query_colors(selected_codes, len(rendered_groups))
    query_trace_start = int(trace_indices["query_start"])
    patched_figure = Patch()

    # Plotly does not expose a trace click-priority setting. While query results
    # are active, exclude underlying category traces from hit testing so the
    # highlighted points receive hover and click interactions.
    category_hoverinfo = "skip" if rendered_groups else "all"
    for trace_index in range(query_trace_start):
        patched_figure["data"][trace_index]["hoverinfo"] = category_hoverinfo

    # Clear every slot first so a shorter query cannot leave stale groups.
    for slot in range(MAX_QUERY_PAPERS):
        trace_index = query_trace_start + slot
        patched_figure["data"][trace_index]["x"] = []
        patched_figure["data"][trace_index]["y"] = []
        patched_figure["data"][trace_index]["customdata"] = []
        patched_figure["data"][trace_index]["showlegend"] = False

    for slot, ((trace_name, point_ids), color) in enumerate(
        zip(rendered_groups, query_colors)
    ):
        trace_index = query_trace_start + slot
        patched_figure["data"][trace_index]["x"] = meta_df.loc[
            point_ids,
            "x",
        ].tolist()
        patched_figure["data"][trace_index]["y"] = meta_df.loc[
            point_ids,
            "y",
        ].tolist()
        # Query overlays are small enough to carry point IDs for click lookup.
        patched_figure["data"][trace_index]["customdata"] = point_ids.tolist()
        patched_figure["data"][trace_index]["name"] = trace_name
        patched_figure["data"][trace_index]["marker"]["color"] = color
        patched_figure["data"][trace_index]["showlegend"] = True

    if too_many_matches:
        status = (
            f"The query matched {total_matches:,} statements, exceeding the "
            f"{MAX_QUERY_HIGHLIGHT_POINTS:,}-point highlight limit. Narrow "
            "the categories, statement type, paper IDs, or keyword."
        )
        status_color = "warning"
    elif not query_groups:
        if paper_ids and keyword:
            status = (
                "No statements from the requested paper IDs contain the "
                "keyword in the current view."
            )
        elif paper_ids:
            status = (
                "No statements from the requested paper IDs are in the "
                "current view."
            )
        else:
            status = (
                f'No statements containing "{keyword}" are in the current view.'
            )
        status_color = "warning"
    else:
        if paper_ids:
            status = (
                f"Highlighted {total_matches:,} statements from "
                f"{len(query_groups)} paper(s) in the current view."
            )
            if keyword:
                status += f' Keyword: "{keyword}".'
        else:
            status = (
                f'Highlighted {total_matches:,} statements containing "{keyword}" '
                "in the current view."
            )
        status_color = "success"

    if missing_papers and not too_many_matches:
        status += " No matches: " + ", ".join(missing_papers) + "."
        if query_groups:
            status_color = "info"

    return (
        patched_figure,
        status,
        status_color,
        bool(rendered_groups),
    )


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


def build_statement_content(point_id):
    """Load and render the small statement payload for one stable point ID."""
    row = meta_df.loc[point_id]
    raw_text = "" if pd.isna(row["text"]) else str(row["text"])
    markdown_preview, conversion_warning = convert_latex_to_markdown(raw_text)
    metadata = html.Div(
        [
            html.Div([html.Strong("Paper ID: "), str(row["paper_id"])]),
            html.Div([html.Strong("Type: "), str(row["type"])]),
            html.Div([html.Strong("Category: "), str(row["categories"])]),
        ]
    )
    return row, metadata, markdown_preview, conversion_warning, raw_text


app = Dash("embedding space", external_stylesheets=[dbc.themes.BOOTSTRAP])

category_select = dbc.Card(
    [
        dbc.CardHeader("Select Categories (up to 5)"),
        dbc.CardBody(
            dcc.Dropdown(
                id="categories",
                options=get_category_options([]),
                multi=True,
                value=[],
            )
        ),
    ],
    className="h-100",
)

type_select = dbc.Card(
    [
        dbc.CardHeader("Select Statement Types"),
        dbc.CardBody(
            dcc.Dropdown(id="types", options=TYPES, value="All")
        ),
    ],
    className="h-100",
)

paper_search_panel = dbc.Card(
    [
        dbc.CardHeader("Search and Highlight"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(
                                    f"Paper IDs (optional, up to {MAX_QUERY_PAPERS})"
                                ),
                                dbc.Input(
                                    id="paper_ids",
                                    type="text",
                                    placeholder="Separate multiple IDs with commas",
                                ),
                            ],
                            xs=12,
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Keyword (optional)"),
                                dbc.Input(
                                    id="keyword",
                                    type="text",
                                    placeholder="Case-insensitive literal keyword",
                                ),
                            ],
                            xs=12,
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Query History"),
                                dcc.Dropdown(
                                    id="query_history",
                                    options=[],
                                    placeholder="No query history yet",
                                    clearable=False,
                                    disabled=True,
                                ),
                            ],
                            xs=12,
                            md=4,
                        ),
                    ],
                    className="g-3",
                ),
                html.Div(
                    [
                        dbc.Button(
                            "Search",
                            id="paper_search_button",
                            color="primary",
                            n_clicks=0,
                            style={"flexShrink": 0},
                        ),
                        dbc.Button(
                            "Clear Highlights",
                            id="clear_paper_search_button",
                            color="secondary",
                            outline=True,
                            n_clicks=0,
                            style={"flexShrink": 0},
                        ),
                        dbc.Alert(
                            id="paper_search_status",
                            is_open=False,
                            className="mb-0 py-2 flex-grow-1",
                        ),
                    ],
                    className="d-flex align-items-center gap-2",
                ),
            ]
        ),
    ],
    className="mt-3",
)

scatter = dbc.Card(
    [
        dbc.CardHeader("Scatter plot"),
        dbc.CardBody(
            html.Div(
                dcc.Graph(
                    id="scatter_plot",
                    figure=empty_scatter_plot(),
                    responsive=True,
                    style={"width": "100%", "height": "100%"},
                ),
                # Keep the graph responsive without allowing the statement card
                # on the right to stretch or compress its aspect ratio.
                style={"width": "100%", "aspectRatio": "3 / 2"},
            ),
            className="p-0",
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
    ]
)

controls = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(category_select, md=8),  # type: ignore
                dbc.Col(type_select, md=4),  # type: ignore
            ],
            className="g-3",
        ),  # type: ignore
        paper_search_panel,
    ]
)

graphs = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(scatter, lg=6),  # type: ignore
                dbc.Col(texts, lg=6),  # type: ignore
            ],
            className="g-3",
            align="start",
        ),  # type: ignore
    ],
    className="mt-3",
)

about = html.Div(
    [
        dbc.Card(
            [
                dbc.CardHeader("Exploring the Embedding Space"),
                dbc.CardBody(
                    [
                        html.P(
                            "Each point represents one theorem-like mathematical "
                            "statement extracted from a research paper, such as a "
                            "theorem, definition, lemma, proposition, corollary, "
                            "conjecture, or claim. The plotted statements are encoded "
                            "with an all-domain NaturalPRISM embedding model fine-tuned "
                            "on examples spanning mathematical subject areas. Point "
                            "metadata identifies the statement type, source paper, and "
                            "mathematical category.",
                            className="mb-3",
                        ),
                        html.P(
                            "The model is trained to retrieve a relevant mathematical "
                            "statement given a partial proof. Nearby statements are "
                            "therefore best understood as occupying similar retrieval "
                            "neighborhoods: the model may associate them with the same "
                            "or similar kinds of partial proofs. This suggests that the "
                            "model treats them as related for this retrieval task. It is "
                            "an indirect signal, since training directly connects partial "
                            "proofs with target statements rather than explicitly "
                            "training pairs of statements to be close.",
                            className="mb-3",
                        ),
                        html.P(
                            "Each statement is represented by a high-dimensional vector, "
                            "which t-SNE projects onto two dimensions for interactive "
                            "exploration. The projection can distort distances and is "
                            "most useful for exploring local neighborhoods. The axes "
                            "have no direct mathematical meaning, and proximity does "
                            "not establish that two statements are equivalent or that "
                            "any particular partial proof would retrieve both equally "
                            "well.",
                            className="mb-0",
                        ),
                    ]
                ),
            ],
            className="mb-3",
        ),
        dbc.Card(
            [
                dbc.CardHeader("About NaturalPRISM"),
                dbc.CardBody(
                    html.P(
                        "NaturalPRISM studies how to retrieve mathematical premises "
                        "needed to continue a proof written in natural language. We "
                        "collected LaTeX papers from 235,314 arXiv mathematics papers "
                        "published between April 2007 and the end of 2018, spanning 32 "
                        "subject categories. After cleaning and filtering, we paired "
                        "partial proofs with the theorem-like statements they reference, "
                        "producing 1,779,598 examples. These examples were used to "
                        "fine-tune embedding models for mathematical retrieval. See the "
                        "paper for details on the dataset, training, and evaluation.",
                        className="mb-0",
                    )
                ),
            ],
            className="mb-3",
        ),
        dbc.Card(
            [
                dbc.CardHeader("Related Links"),
                dbc.CardBody(
                    html.Ul(
                        [
                            html.Li(
                                html.A(
                                    "Read the paper",
                                    href="PAPER_LINK_PLACEHOLDER",
                                    target="_blank",
                                    rel="noopener noreferrer",
                                )
                            ),
                            html.Li(
                                html.A(
                                    "Project GitHub repository",
                                    href="GITHUB_LINK_PLACEHOLDER",
                                    target="_blank",
                                    rel="noopener noreferrer",
                                )
                            ),
                        ],
                        className="mb-0",
                    )
                ),
            ]
        ),
    ],
    className="mt-3",
)

body = html.Div(
    [
        # Store only small trace offsets instead of passing the full figure as State.
        dcc.Store(id="highlight_trace_indices", data=None),
        # Persist only compact query descriptors in the browser. Point arrays and
        # Plotly figures remain server-side or are reconstructed on demand.
        dcc.Store(id="query_history_store", storage_type="local", data=[]),
        dcc.Store(id="filter_history_event", data=None),
        dcc.Store(id="point_history_event", data=None),
        dcc.Store(id="history_replay_request", data=None),
        html.H1("Embedding Space", style={"text-align": "center"}),
        html.Div(
            [
                html.H2("Scatter Plot", className="mb-0"),
                html.Span(
                    [
                        "How to use ",
                        html.Span(
                            "?",
                            className=(
                                "border rounded-circle d-inline-flex "
                                "align-items-center justify-content-center"
                            ),
                            style={
                                "width": "1.25rem",
                                "height": "1.25rem",
                                "fontSize": "0.8rem",
                                "fontWeight": "bold",
                            },
                        ),
                    ],
                    id="embedding_help_target",
                    tabIndex=0,
                    role="button",
                    className="text-muted small d-inline-flex align-items-center gap-1",
                    style={"cursor": "help"},
                ),
                dbc.Tooltip(
                    "Choose up to five categories and a statement type to filter "
                    "the plot. Click a point to inspect its statement and metadata. "
                    "To find statements from particular papers, enter one or more "
                    "paper IDs; you may also add a keyword. If both are supplied, "
                    "results must match both. Searches use the selected categories "
                    "and statement type. Use Query History to revisit recent "
                    "searches or points.",
                    target="embedding_help_target",
                    placement="right",
                    style={"maxWidth": "360px", "textAlign": "left"},
                ),
            ],
            className="d-flex align-items-center gap-2",
        ),
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
    Output("query_history_store", "data"),
    Input("filter_history_event", "data"),
    Input("point_history_event", "data"),
    State("query_history_store", "data"),
    prevent_initial_call=True,
)
def update_query_history(filter_event, point_event, history):
    """Merge both history sources into one bounded, deduplicated local list."""
    if ctx.triggered_id == "filter_history_event":
        new_record = filter_event
    elif ctx.triggered_id == "point_history_event":
        new_record = point_event
    else:
        return no_update

    if not isinstance(new_record, dict):
        return no_update

    return add_history_record(history, new_record)


@callback(
    Output("query_history", "options"),
    Output("query_history", "disabled"),
    Output("query_history", "placeholder"),
    Input("query_history_store", "data"),
)
def update_query_history_options(history):
    existing_history = history if isinstance(history, list) else []
    valid_records = [
        record
        for record in existing_history
        if isinstance(record, dict)
        and record.get("id")
        and history_record_key(record) is not None
    ][:MAX_QUERY_HISTORY]
    options = [
        {
            "label": history_record_label(record),
            "value": record["id"],
        }
        for record in valid_records
    ]

    if options:
        return options, False, "Select a previous query"

    return [], True, "No query history yet"


@callback(
    Output("categories", "value", allow_duplicate=True),
    Output("types", "value"),
    Output("paper_ids", "value"),
    Output("keyword", "value"),
    Output("history_replay_request", "data"),
    Output("query_history", "value"),
    Input("query_history", "value"),
    State("query_history_store", "data"),
    prevent_initial_call=True,
)
def prepare_history_replay(record_id, history):
    if not record_id:
        return (no_update,) * 6

    record = next(
        (
            item
            for item in (history or [])
            if isinstance(item, dict) and item.get("id") == record_id
        ),
        None,
    )
    if record is None or history_record_key(record) is None:
        return no_update, no_update, no_update, no_update, no_update, None

    categories = normalize_history_categories(record.get("categories"))
    statement_type = record.get("statement_type", "All")
    if statement_type not in TYPES:
        statement_type = "All"

    replay_request = dict(record)
    replay_request["replay_id"] = uuid.uuid4().hex

    if record.get("kind") == "filter":
        paper_value = ", ".join(
            str(value)
            for value in (record.get("paper_ids") or [])
        )
        keyword_value = record.get("keyword", "")
    else:
        paper_value = ""
        keyword_value = ""

    # Reset the dropdown after dispatch so the same record can be selected again.
    return (
        categories,
        statement_type,
        paper_value,
        keyword_value,
        replay_request,
        None,
    )


@callback(
    Output("scatter_plot", "figure"),
    Output("highlight_trace_indices", "data"),
    Output("paper_search_status", "children"),
    Output("paper_search_status", "color"),
    Output("paper_search_status", "is_open"),
    Output("filter_history_event", "data"),
    Input("categories", "value"),
    Input("types", "value"),
    Input("paper_search_button", "n_clicks"),
    Input("clear_paper_search_button", "n_clicks"),
    Input("history_replay_request", "data"),
    State("highlight_trace_indices", "data"),
    State("paper_ids", "value"),
    State("keyword", "value"),
)
def update_scatter_and_search(
    select_categories,
    select_type,
    _search_clicks,
    _clear_clicks,
    history_replay_request,
    trace_indices,
    raw_paper_ids,
    raw_keyword,
):
    selected_codes = get_category_codes(select_categories)
    triggered_id = ctx.triggered_id
    replay_triggered = (
        "history_replay_request.data" in ctx.triggered_prop_ids
    )

    if replay_triggered and history_replay_request:
        if not selected_codes:
            return (
                empty_scatter_plot(),
                None,
                "The saved history record has no valid categories.",
                "danger",
                True,
                no_update,
            )

        filtered_df = get_filtered_df(meta_df, selected_codes, select_type)
        figure, new_trace_indices = scatter_plot(filtered_df, selected_codes)
        new_trace_indices["history_replay_id"] = history_replay_request.get(
            "replay_id"
        )
        return figure, new_trace_indices, "", "info", False, no_update

    if triggered_id == "clear_paper_search_button":
        if not trace_indices:
            return no_update, no_update, "", "info", False, no_update

        patched_figure = Patch()
        query_trace_start = int(trace_indices["query_start"])

        # Restore normal category-point interaction when query mode ends.
        for trace_index in range(query_trace_start):
            patched_figure["data"][trace_index]["hoverinfo"] = "all"

        for slot in range(MAX_QUERY_PAPERS):
            trace_index = query_trace_start + slot
            patched_figure["data"][trace_index]["x"] = []
            patched_figure["data"][trace_index]["y"] = []
            patched_figure["data"][trace_index]["customdata"] = []
            patched_figure["data"][trace_index]["showlegend"] = False

        return patched_figure, no_update, "", "info", False, no_update

    if triggered_id == "paper_search_button":
        paper_ids = parse_paper_ids(raw_paper_ids)
        keyword = (raw_keyword or "").strip()

        if not selected_codes:
            return (
                no_update,
                no_update,
                "Select at least one category before searching.",
                "warning",
                True,
                no_update,
            )

        if not paper_ids and not keyword:
            return (
                no_update,
                no_update,
                "Enter at least one paper ID or a keyword.",
                "warning",
                True,
                no_update,
            )

        if len(paper_ids) > MAX_QUERY_PAPERS:
            return (
                no_update,
                no_update,
                f"Enter no more than {MAX_QUERY_PAPERS} paper IDs per query.",
                "warning",
                True,
                no_update,
            )

        if not trace_indices:
            return (
                no_update,
                no_update,
                "The scatter plot is not ready for query highlighting.",
                "danger",
                True,
                no_update,
            )

        patched_figure, status, status_color, succeeded = (
            build_query_highlight_patch(
                selected_codes,
                select_type,
                paper_ids,
                keyword,
                trace_indices,
            )
        )
        history_event = (
            make_filter_history_record(
                select_categories,
                select_type,
                paper_ids,
                keyword,
            )
            if succeeded
            else no_update
        )
        return (
            patched_figure,
            no_update,
            status,
            status_color,
            True,
            history_event,
        )

    if not selected_codes:
        return empty_scatter_plot(), None, "", "info", False, no_update

    filtered_df = get_filtered_df(meta_df, selected_codes, select_type)
    figure, new_trace_indices = scatter_plot(filtered_df, selected_codes)
    return figure, new_trace_indices, "", "info", False, no_update


@callback(
    Output("statement_metadata", "children"),
    Output("statement_latex", "children"),
    Output("latex_warning", "children"),
    Output("latex_warning", "is_open"),
    Output("statement_raw", "children"),
    Output("scatter_plot", "figure", allow_duplicate=True),
    Output("point_history_event", "data"),
    Input("scatter_plot", "clickData"),
    State("highlight_trace_indices", "data"),
    State("categories", "value"),
    State("types", "value"),
    prevent_initial_call=True,
)
def display_text(
    click_data,
    trace_indices,
    select_categories,
    select_type,
):
    if (
        not click_data
        or not click_data.get("points")
        or not trace_indices
    ):
        return (no_update,) * 7

    point = click_data["points"][0]
    curve_number = point.get("curveNumber")
    point_number = point.get("pointNumber")

    if curve_number is None or point_number is None:
        return (no_update,) * 7

    curve_number = int(curve_number)
    point_number = int(point_number)
    selected_codes = get_category_codes(select_categories)
    query_trace_start = int(trace_indices["query_start"])
    clicked_point_trace_index = int(trace_indices["clicked_point"])

    if 0 <= curve_number < len(selected_codes):
        category = selected_codes[curve_number]
        filter_type = select_type if select_type and select_type != "All" else "All"
        point_ids = POINT_IDS_BY_FILTER.get((category, filter_type))

        # pointNumber indexes the original-row order used to build each trace.
        if point_ids is None or point_number < 0 or point_number >= len(point_ids):
            return (no_update,) * 7

        point_id = int(point_ids[point_number])
    elif query_trace_start <= curve_number < clicked_point_trace_index:
        # Query traces carry IDs only for their small highlighted subsets.
        point_id = point.get("customdata")
        if isinstance(point_id, (list, tuple)):
            point_id = point_id[0] if point_id else None

        if point_id is None:
            return (no_update,) * 7

        point_id = int(point_id)
    else:
        return (no_update,) * 7

    row, metadata, markdown_preview, conversion_warning, raw_text = (
        build_statement_content(point_id)
    )

    # Patch only the final one-point trace; the base point cloud is not retransmitted.
    patched_figure = Patch()
    patched_figure["data"][clicked_point_trace_index]["x"] = [row["x"]]
    patched_figure["data"][clicked_point_trace_index]["y"] = [row["y"]]
    history_event = make_point_history_record(
        point_id,
        row,
        select_categories,
        select_type,
    )

    return (
        metadata,
        markdown_preview,
        conversion_warning or "",
        conversion_warning is not None,
        raw_text,
        patched_figure,
        history_event,
    )


@callback(
    Output("scatter_plot", "figure", allow_duplicate=True),
    Output("paper_search_status", "children", allow_duplicate=True),
    Output("paper_search_status", "color", allow_duplicate=True),
    Output("paper_search_status", "is_open", allow_duplicate=True),
    Output("statement_metadata", "children", allow_duplicate=True),
    Output("statement_latex", "children", allow_duplicate=True),
    Output("latex_warning", "children", allow_duplicate=True),
    Output("latex_warning", "is_open", allow_duplicate=True),
    Output("statement_raw", "children", allow_duplicate=True),
    Input("highlight_trace_indices", "data"),
    Input("history_replay_request", "data"),
    prevent_initial_call=True,
)
def apply_history_replay(trace_indices, replay_request):
    """Apply a saved action only after its restored base plot is ready."""
    if not trace_indices or not replay_request:
        return (no_update,) * 9

    replay_id = replay_request.get("replay_id")
    if not replay_id or trace_indices.get("history_replay_id") != replay_id:
        return (no_update,) * 9

    selected_categories = normalize_history_categories(
        replay_request.get("categories")
    )
    selected_codes = get_category_codes(selected_categories)
    select_type = replay_request.get("statement_type", "All")
    if select_type not in TYPES:
        select_type = "All"

    if replay_request.get("kind") == "filter":
        paper_ids = [
            str(value)
            for value in (replay_request.get("paper_ids") or [])
            if str(value)
        ][:MAX_QUERY_PAPERS]
        keyword = str(replay_request.get("keyword", "")).strip()

        if not selected_codes or (not paper_ids and not keyword):
            return (
                no_update,
                "The saved filter record is no longer valid.",
                "danger",
                True,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        patched_figure, status, status_color, _ = (
            build_query_highlight_patch(
                selected_codes,
                select_type,
                paper_ids,
                keyword,
                trace_indices,
            )
        )
        return (
            patched_figure,
            status,
            status_color,
            True,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    if replay_request.get("kind") != "point":
        return (no_update,) * 9

    try:
        point_id = int(replay_request.get("point_id"))
    except (TypeError, ValueError):
        return (no_update,) * 9

    if point_id not in meta_df.index:
        return (
            no_update,
            "The selected point no longer exists in the current dataset.",
            "warning",
            True,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    row, metadata, markdown_preview, conversion_warning, raw_text = (
        build_statement_content(point_id)
    )
    row_category = str(row["categories"])
    row_type = str(row["type"])
    point_is_visible = (
        row_category in selected_codes
        and (select_type == "All" or row_type == select_type)
    )
    if not point_is_visible:
        return (
            no_update,
            "The saved point is not visible under its restored filters.",
            "warning",
            True,
            metadata,
            markdown_preview,
            conversion_warning or "",
            conversion_warning is not None,
            raw_text,
        )

    patched_figure = Patch()
    clicked_point_trace_index = int(trace_indices["clicked_point"])
    patched_figure["data"][clicked_point_trace_index]["x"] = [row["x"]]
    patched_figure["data"][clicked_point_trace_index]["y"] = [row["y"]]
    return (
        patched_figure,
        "",
        "info",
        False,
        metadata,
        markdown_preview,
        conversion_warning or "",
        conversion_warning is not None,
        raw_text,
    )


app.run(debug=True)
