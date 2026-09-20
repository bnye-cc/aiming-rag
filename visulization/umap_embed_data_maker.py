#This file makes UMAP projection data from embedding spaces, with optional filtering to only project some math tags
import json
import re
import numpy as np
from tqdm import tqdm


EMBEDDINGS = '/home/aiming/embedspaces/theorems_Qwen3-Embedding-0.6B_base_embeddings.npy'#theorems_Qwen3-Embedding-0.6B_AG_50k_embeddings.npy
METADATA   = '/home/aiming/embedspaces/theorems_Qwen3-Embedding-0.6B_base_metadata.json'
OUTPUT     = '/home/aiming/plotdata/theorems_Qwen3-Embedding-0.6B_base_only6_projection.jsonl'

WHITELIST  = ['math.NT','math.CO','math.AG','math.GT','math.PR','math.FA']          # e.g. ['math.CO', 'math.AG']; [] projects ALL theorems
SUBSAMPLE  = None        # cap the number of projected points; None = all

ARXIV_SNAPSHOT = '/home/aiming/arxiv-metadata-oai-snapshot.json'

N_NEIGHBORS = 15
MIN_DIST    = 0.1
SEED        = 42
COORD_DECIMALS = 5       # rounding coords
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
    """Stream the snapshot once; map needed paper_id -> full categories string."""
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


def main():
    print(f"Loading embeddings: {EMBEDDINGS}")
    vectors = np.load(EMBEDDINGS)
    print(f"Loading metadata:   {METADATA}")
    with open(METADATA, 'r') as f:
        metadata = json.load(f)
    assert len(vectors) == len(metadata), \
        f"Embedding/metadata length mismatch: {len(vectors)} vs {len(metadata)}"

    paper_ids = [normalize_paper_id(m.get('paper_id')) for m in metadata]
    lookup = build_tag_lookup(paper_ids)

    #
    wl = set(WHITELIST)
    keep_idx, cats_keep = [], []
    for i, pid in enumerate(paper_ids):
        cats = lookup.get(pid)          
        if wl:
            if not cats or not (set(cats.split()) & wl):
                continue
        keep_idx.append(i)
        cats_keep.append(cats)

    if not keep_idx:
        raise SystemExit("Nothing to project (whitelist matched no theorems).")

    vectors = vectors[keep_idx]
    meta_keep = [metadata[i] for i in keep_idx]
    print(f"{len(keep_idx):,} theorems selected for projection"
          + (f" (whitelist {WHITELIST})" if wl else " (all)"))

    
    if SUBSAMPLE and SUBSAMPLE < len(vectors):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(vectors), size=SUBSAMPLE, replace=False)
        idx.sort()
        vectors = vectors[idx]
        meta_keep = [meta_keep[j] for j in idx]
        cats_keep = [cats_keep[j] for j in idx]
        print(f"Subsampled to {len(vectors):,} points")

    # UMAP
    import umap
    print(f"Running UMAP on {len(vectors):,} x {vectors.shape[1]} ...")
    reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST,
                        metric='cosine', init='random', verbose=True)
    coords = reducer.fit_transform(vectors)

    # write JSONL
    print(f"Writing {OUTPUT} ...")
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for m, cats, xy in zip(meta_keep, cats_keep, coords):
            rec = dict(m)
            if DROP_ENV:
                rec.pop('env', None)
            rec['categories'] = cats          # None if the id wasn't in the snapshot
            rec['projection_coords'] = [round(float(xy[0]), COORD_DECIMALS),
                                        round(float(xy[1]), COORD_DECIMALS)]
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"Done. {len(meta_keep):,} records written to {OUTPUT}")


if __name__ == "__main__":
    main()
