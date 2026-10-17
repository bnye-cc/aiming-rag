<!-- TODO: Add documentation for the projection-generation code in a separate section here. -->

# Embedding Space Visualization

This Dash application provides an interactive view of mathematical statements in a two-dimensional embedding projection. It lets users filter statements, search for records, inspect individual points, and view the corresponding statement text.

## Requirements

Install the Python packages used by the app:

```bash
python -m pip install dash dash-bootstrap-components pandas numpy plotly
```

Pandoc is recommended for converting LaTeX statement fragments to Markdown. The app displays the converted text with Dash Markdown and MathJax. If Pandoc is unavailable, the original LaTeX is shown in the statement preview.

## Data file

The app reads a CSV file. Set the `DATA` constant near the top of `embeding_viz.py` to its location. The CSV must contain the following columns:

| Column | Description |
| --- | --- |
| `x`, `y` | Two-dimensional coordinates for each statement. |
| `text` | Mathematical statement text or LaTeX fragment. |
| `paper_id` | Identifier of the source paper. |
| `type` | Statement type, such as Theorem, Definition, Lemma, Proposition, Corollary, Conjecture, or Claim. |
| `categories` | Mathematical subject category used by the category filter. |

## Run the app

From the directory containing `embeding_viz.py`, run:

```bash
python embeding_viz.py
```

Open the local address printed by Dash in your browser, normally `http://127.0.0.1:8050`.

## Using the interface

1. **Choose categories.** Select one or more mathematical subject categories. Up to five can be selected at once. The plot appears after at least one category is selected.
2. **Choose statement types.** Select a statement type, or leave it set to `All`.
3. **Inspect a statement.** Click a point to see its metadata and statement text. Expand **Raw LaTeX** to view the original source.
4. **Search and highlight.** Enter up to ten comma-separated paper IDs, a keyword, or both, then click **Search**. Keyword matching is case-insensitive. If both fields are filled, results must match a requested paper ID and contain the keyword. Searches apply to the currently selected categories and statement type. Click **Clear Highlights** to remove the search overlay.
5. **Restore a recent action.** Use **Query History** to revisit a filter search or selected point. Up to 20 recent entries are stored in that browser's local storage.

Each search can highlight at most 50,000 points. If a search exceeds this limit, narrow the selected categories, statement type, paper IDs, or keyword.

## Reading the scatter plot

Each point represents one theorem-like mathematical statement. Its position is a two-dimensional projection of the statement's embedding vector. The axes have no direct mathematical meaning, and the projection can distort distances, particularly across larger regions of the plot.

The embedding model was fine-tuned to retrieve statements relevant to a partial proof. Nearby statements are therefore best interpreted as statements the model places in similar retrieval neighborhoods: it may associate them with the same or similar partial proofs. This is an indirect, task-specific similarity; proximity does not establish that two statements are mathematically equivalent or that every partial proof would retrieve both equally well.

## Implementation

- Dash and dash-bootstrap-components provide the web interface.
- Plotly `Scattergl` traces render the point cloud using WebGL. Dash partial figure updates are used for point selection and search highlights.
- Pandoc converts statement fragments to Markdown, and MathJax renders mathematical notation in the browser.
- Query history is stored in browser local storage; point-ID lookup arrays are kept on the server.


