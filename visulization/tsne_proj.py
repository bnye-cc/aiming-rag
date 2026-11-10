import argparse
import json
import re
import numpy as np
from tqdm import tqdm
import pandas as pd

# --- Configuration ---
WRITE_JSONL = False
PRIMARY_ONLY = True
ARXIV_SNAPSHOT = 'arxiv-metadata-oai-snapshot.json'

PERPLEXITY    = 150.0
EARLY_EXAG    = 12.0
LEARNING_RATE = 'auto'
N_ITER        = 1000
INIT          = 'pca'
PCA_DIMS      = 50
USE_OPENTSNE  = True 

SEED = 42
COORD_DECIMALS = 5
DROP_ENV    = True

def normalize_paper_id(pid):
    if pid is None:
        return None
    pid = re.sub(r'v\d+$', '', str(pid).strip())
    if re.match(r'^\d{4}_\d{4,5}$', pid):
        return pid.replace('_', '.')
    m = re.match(r'^([a-z-]+)_([A-Za-z]+)_(\d{7})$', pid)
    if m:
        return f"{m.group(1)}.{m.group(2)}/{m.group(3)}"
    m = re.match(r'^([a-z-]+)_(\d{7})$', pid)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return pid

def build_tag_lookup(needed_ids):
    needed = set(i for i in needed_ids if i)
    lookup = {}
    print(f"Streaming {ARXIV_SNAPSHOT} for {len(needed):,} unique paper ids...")
    with open(ARXIV_SNAPSHOT, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Scanning arXiv snapshot"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get('id')
            if pid in needed:
                lookup[pid] = rec.get('categories') or ''
                if len(lookup) == len(needed):
                    break
    print(f"Found categories for {len(lookup):,} / {len(needed):,} ids")
    return lookup

def run_tsne(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = (vectors / norms).astype(np.float32)

    if PCA_DIMS and PCA_DIMS < vectors.shape[1]:
        from sklearn.decomposition import PCA
        print(f"PCA {vectors.shape[1]} -> {PCA_DIMS} ...")
        vectors = PCA(n_components=PCA_DIMS, random_state=SEED).fit_transform(vectors)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    print(f"Running openTSNE on {len(vectors):,} x {vectors.shape[1]} (perplexity={PERPLEXITY}) ...")
    from openTSNE import TSNE as OpenTSNE
    reducer = OpenTSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        metric='cosine',
        initialization=INIT,
        learning_rate=LEARNING_RATE,
        early_exaggeration=EARLY_EXAG,
        n_iter=N_ITER,
        random_state=SEED,
        n_jobs=-1,  # Kept -1 since it's working fine for you!
        verbose=True,
    )
    return np.asarray(reducer.fit(vectors))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="Path prefix (without _embeddings.npy)")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--whitelist", nargs='*', default=[], help="Math categories to keep")
    args = parser.parse_args()

    EMBEDDINGS = f"{args.prefix}_embeddings.npy"
    METADATA   = f"{args.prefix}_metadata.json"
    OUTPUT_CSV = args.out_csv
    WHITELIST  = args.whitelist

    print(f"Loading embeddings: {EMBEDDINGS}")
    vectors = np.load(EMBEDDINGS)
    print(f"Loading metadata:   {METADATA}")
    with open(METADATA, 'r') as f:
        metadata = json.load(f)

    paper_ids = [normalize_paper_id(m.get('paper_id')) for m in metadata]
    lookup = build_tag_lookup(paper_ids)

    wl = set(WHITELIST)
    keep_idx, cats_keep = [], []
    for i, pid in enumerate(paper_ids):
        cats = lookup.get(pid)
        if wl:
            if not cats:
                continue
            tags = cats.split()
            if PRIMARY_ONLY:
                if tags[0] not in wl:
                    continue
            elif not (set(tags) & wl):
                continue
        keep_idx.append(i)
        cats_keep.append(cats)

    if not keep_idx:
        raise SystemExit("Nothing to project (whitelist matched no theorems).")

    vectors = vectors[keep_idx]
    meta_keep = [metadata[i] for i in keep_idx]
    print(f"{len(keep_idx):,} theorems selected for projection" + (f" (whitelist {WHITELIST})" if wl else " (all)"))

    coords = run_tsne(vectors)

    records = []
    for m, cats, xy in zip(meta_keep, cats_keep, coords):
        rec = dict(m)
        if DROP_ENV:
            rec.pop('env', None)
        rec['categories'] = cats
        rec['projection_coords'] = [round(float(xy[0]), COORD_DECIMALS),
                                    round(float(xy[1]), COORD_DECIMALS)]
        records.append(rec)

    print(f"Building dataframe ({len(records):,} rows) ...")
    df = pd.DataFrame(records)
    df['x'] = df['projection_coords'].apply(lambda v: v[0])
    df['y'] = df['projection_coords'].apply(lambda v: v[1])
    df.drop(columns='projection_coords', inplace=True)

    df['categories'] = df['categories'].apply(lambda c: str(c).split(' ')[0])
    df = df[df['categories'].apply(lambda c: c.startswith('math.'))]

    df.to_csv(OUTPUT_CSV, encoding='utf-8', index=False)
    print(f"Done. {len(df):,} records written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
