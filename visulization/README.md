# Visualization

This folder contains a script for preparing two-dimensional coordinates from mathematical-statement embeddings and a Dash application for exploring the resulting points. The projection script writes a CSV; the visualization app reads that CSV and lets users filter and inspect its statements.

## Requirements

Install the Python packages used by both scripts:

```bash
python -m pip install numpy pandas tqdm scikit-learn openTSNE dash dash-bootstrap-components plotly
```

Pandoc is recommended for converting LaTeX statement fragments to Markdown. The app renders the converted content with Dash Markdown and MathJax. If Pandoc is unavailable, the original LaTeX is shown in the statement preview.

## Prepare the projection data

`tsne_proj.py` reads an embedding matrix and its matching metadata, looks up each paper's arXiv categories, and optionally filters records by category. It normalizes the vectors, optionally reduces their dimensions with PCA, and uses openTSNE with cosine distance to calculate two-dimensional coordinates. The output CSV contains the selected records with `x` and `y` coordinate columns.

Set `ARXIV_SNAPSHOT` near the top of `tsne_proj.py` to the local arXiv metadata snapshot. The script accepts the prefix shared by the embedding matrix and metadata, an output CSV path, and an optional category whitelist. For example:

```bash
python tsne_proj.py \
  --prefix /path/to/embedding_run \
  --out_csv /path/to/visualization_data.csv \
  --whitelist math.NT math.CO math.AG
```

Omit `--whitelist` to include records with any matched mathematical category. With the current `PRIMARY_ONLY = True` setting, filtering uses each paper's primary arXiv category, which is also the category written to the CSV. Projection parameters such as perplexity, PCA dimension, iteration count, and random seed are configured near the top of the script.

## Run the visualization app

In `embeding_viz.py`, set `DATA` to the projection CSV you want to explore, then run:

```bash
python embeding_viz.py
```

Open the local address printed by Dash in your browser, normally `http://127.0.0.1:8050`.

The CSV needs these columns:

| Column | Description |
| --- | --- |
| `x`, `y` | Two-dimensional coordinates for each statement. |
| `text` | Mathematical statement text or LaTeX fragment. |
| `paper_id` | Identifier of the source paper. |
| `type` | Statement type, such as Theorem, Definition, Lemma, Proposition, Corollary, Conjecture, or Claim. |
| `categories` | Mathematical subject category used by the category filter. |

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

Before running the two steps, check that the output CSV path supplied to `tsne_proj.py` matches the `DATA` path configured in `embeding_viz.py`.
