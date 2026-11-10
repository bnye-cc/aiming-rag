"""
hydra_eval.py -- routed multi-embedder retrieval eval on FlagEmbedding custom datasets.

Pipeline, per eval set (any dir with test_queries.jsonl / test_qrels.jsonl / corpus.jsonl):

  1. ROUTE   the Qwen3.5-4B classifier predicts a category for every query
             -> {workdir}/routing.jsonl   (qid, category, confidence, full probs)
  2. BUCKET  queries are split by predicted category into 6 bucket dataset dirs,
             each with the bucket's queries, the FULL corpus (symlinked), and
             qrels filtered to the bucket's qids
  3. SEARCH  python -m FlagEmbedding.evaluation.custom runs per non-empty bucket
             with that category's specialist embedder -- stock CLI, so all of
             FlagEmbedding's multi-GPU embedding/search machinery is reused as-is.
             Corpus embeddings cache per (dataset, category) and survive reruns.
  4. MERGE   per-bucket search results (qid -> {docid: score}) are unioned
             (buckets are disjoint by qid) and scored jointly against the FULL
             qrels with pytrec_eval. This is the headline end-to-end number.
             The per-bucket markdown files FlagEmbedding writes are the
             CONDITIONAL numbers (retrieval quality given that routing).

Misrouting note: the corpus is shared, so a misrouted query still searches the
full corpus -- just through the wrong specialist's embedding space. It degrades
to that specialist's cross-area performance, it does not go to zero.

Usage:
    python hydra_eval.py \
        --dataset_dir ./datadictBALALL \
        --workdir ./hydra_runs/BALALL \
        --classifier qwen35_4b_category6/best_model \
        --embedder_CO /path/qwen3-0.6b-math-CO_50k \
        --embedder_AG /path/qwen3-0.6b-math-AG_50k \
        --embedder_NT /path/qwen3-0.6b-math-NT_50k \
        --embedder_GT /path/qwen3-0.6b-math-GT_50k \
        --embedder_FA /path/qwen3-0.6b-math-FA_50k \
        --embedder_PR /path/qwen3-0.6b-math-PR_50k \
        --devices cuda:0 cuda:1 cuda:2 cuda:3

    Stages are resumable: --skip_routing reuses routing.jsonl, and buckets whose
    search results already exist are skipped unless --force_search.

Requires: the training env (fla kernels for the classifier), FlagEmbedding,
pytrec_eval (a FlagEmbedding dependency, so already present).
"""

import os
import sys
import json
import glob
import argparse
import subprocess
from collections import Counter, defaultdict

CATEGORIES = ["math.CO", "math.AG", "math.NT", "math.GT", "math.FA", "math.PR"]
SHORT = {c: c.split(".")[1] for c in CATEGORIES}          # math.CO -> CO


# ---------------------------------------------------------------------------
# stage 1: routing
# ---------------------------------------------------------------------------
def run_routing(args, queries):
    """Classify every query; write routing.jsonl. Imports torch lazily so the
    merge-only path doesn't need a GPU."""
    routing_path = os.path.join(args.workdir, "routing.jsonl")
    if args.skip_routing and os.path.exists(routing_path):
        print(f"[route] reusing {routing_path}")
        return load_routing(routing_path)

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.classify_py)))
    from classify import load_classifier, classify   # reuse the inference tool

    model, tokenizer, categories, max_length = load_classifier(
        args.classifier, use_4bit=not args.classifier_no_4bit
    )
    assert categories == CATEGORIES, (
        f"classifier checkpoint category order {categories} != expected {CATEGORIES}"
    )

    texts = [q["text"] for q in queries]
    print(f"[route] classifying {len(texts):,} queries ...")
    preds = classify(texts, model, tokenizer, categories, max_length,
                     batch_size=args.classifier_batch_size)

    with open(routing_path, "w") as f:
        for q, p in zip(queries, preds):
            f.write(json.dumps({
                "qid": q["id"],
                "category": p["category"],
                "confidence": p["confidence"],
                "probs": p["probs"],
            }) + "\n")

    # free the 4B before the 0.6B embedders start
    del model
    import torch
    torch.cuda.empty_cache()
    print(f"[route] wrote {routing_path}")
    return load_routing(routing_path)


def load_routing(path):
    routing = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            routing[str(r["qid"])] = r["category"]
    return routing


# ---------------------------------------------------------------------------
# stage 2: bucket dataset dirs
# ---------------------------------------------------------------------------
def make_buckets(args, queries, qrels_rows, routing):
    buckets = defaultdict(list)
    for q in queries:
        buckets[routing[str(q["id"])]].append(q)

    qrels_by_qid = defaultdict(list)
    for r in qrels_rows:
        qrels_by_qid[str(r["qid"])].append(r)

    bucket_dirs = {}
    for cat, qs in buckets.items():
        short = SHORT[cat]
        bdir = os.path.join(args.workdir, f"bucket_{short}")
        os.makedirs(bdir, exist_ok=True)

        with open(os.path.join(bdir, "test_queries.jsonl"), "w") as f:
            for q in qs:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        with open(os.path.join(bdir, "test_qrels.jsonl"), "w") as f:
            for q in qs:
                for r in qrels_by_qid.get(str(q["id"]), []):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # full corpus, shared: symlink, don't copy
        corpus_link = os.path.join(bdir, "corpus.jsonl")
        corpus_src = os.path.abspath(os.path.join(args.dataset_dir, "corpus.jsonl"))
        if os.path.islink(corpus_link) or os.path.exists(corpus_link):
            os.remove(corpus_link)
        os.symlink(corpus_src, corpus_link)

        bucket_dirs[cat] = bdir
        print(f"[bucket] {cat}: {len(qs):,} queries -> {bdir}")
    return bucket_dirs


# ---------------------------------------------------------------------------
# stage 3: per-bucket FlagEmbedding runs
# ---------------------------------------------------------------------------
def run_bucket_evals(args, bucket_dirs, embedders):
    for cat, bdir in sorted(bucket_dirs.items()):
        short = SHORT[cat]
        out_dir = os.path.join(bdir, "search_results")
        if not args.force_search and glob.glob(os.path.join(out_dir, "**", "*.json"),
                                               recursive=True):
            print(f"[search] {cat}: results exist, skipping (use --force_search to redo)")
            continue

        # corpus embedding cache: per (dataset, category-embedder), OUTSIDE the
        # bucket dir so it survives re-routing with an updated classifier
        embd_cache = os.path.join(args.workdir, "corpus_embd_cache", short)

        cmd = [
            sys.executable, "-m", "FlagEmbedding.evaluation.custom",
            "--eval_name", f"hydra_{args.run_name}_{short}",
            "--dataset_dir", bdir,
            "--splits", "test",
            "--corpus_embd_save_dir", embd_cache,
            "--output_dir", out_dir,
            "--search_top_k", str(args.search_top_k),
            "--eval_output_method", "markdown",
            "--eval_output_path", os.path.join(bdir, f"eval_results_{short}.md"),
            "--eval_metrics", *args.eval_metrics,
            "--embedder_name_or_path", embedders[cat],
            "--devices", *args.devices,
            "--embedder_batch_size", str(args.embedder_batch_size),
            "--embedder_model_class", args.embedder_model_class,
        ]
        print(f"[search] {cat}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# stage 4: merge + joint metrics
# ---------------------------------------------------------------------------
def _extract_rankings(data, bucket_qids):
    """Return the {qid: {docid: score}} dict from `data`, whether it IS that
    dict or wraps it one level down (e.g. under a 'search_results' key)."""
    def looks_like_rankings(d):
        if not isinstance(d, dict) or not d:
            return False
        keys = list(d)[:50]
        return sum(1 for k in keys if str(k) in bucket_qids) >= len(keys) * 0.9

    if looks_like_rankings(data):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if looks_like_rankings(v):
                return v
    return None


def find_search_results(bucket_dir):
    with open(os.path.join(bucket_dir, "test_queries.jsonl")) as f:
        bucket_qids = {str(json.loads(ln)["id"]) for ln in f if ln.strip()}

    hits = []
    for path in glob.glob(os.path.join(bucket_dir, "search_results", "**", "*.json"),
                          recursive=True):
        parts = os.path.normpath(path).split(os.sep)
        if "EVAL" in parts or os.path.basename(path) == "eval_results.json":
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        rankings = _extract_rankings(data, bucket_qids)
        if rankings:
            hits.append((path, rankings))
    return hits


def merge_and_score(args, bucket_dirs, qrels_rows, routing, n_queries):
    merged = {}
    for cat, bdir in sorted(bucket_dirs.items()):
        hits = find_search_results(bdir)
        if not hits:
            print(f"[merge] WARNING: no search results found under {bdir} -- "
                  f"bucket {cat} contributes nothing. Joint metrics will treat "
                  f"its queries as complete misses.")
            continue
        if len(hits) > 1:
            print(f"[merge] NOTE: {len(hits)} result files in {bdir}, using the "
                  f"largest; paths: {[p for p, _ in hits]}")
        path, data = max(hits, key=lambda pd: len(pd[1]))
        overlap = set(data) & set(merged)
        assert not overlap, f"qid overlap across buckets (impossible if routing is a function): {list(overlap)[:5]}"
        merged.update(data)
        sample = next(iter(data))
        print(f"[merge] {cat}: sample qid {sample!r} -> {len(data[sample])} docs")
        print(f"[merge] {cat}: {len(data):,} queries from {path}")

    # pytrec_eval: qrels and run must be {qid: {docid: int/float}}
    import pytrec_eval
    qrels = defaultdict(dict)
    for r in qrels_rows:
        qrels[str(r["qid"])][str(r["docid"])] = int(r["relevance"])

    # metric name translation: FlagEmbedding style -> pytrec_eval style
    measures = set()
    for m in args.eval_metrics:
        name, _, k = m.rpartition("_at_")
        measures.add(f"{name}_cut.{k}" if name == "ndcg" else f"{name}.{k}")

    evaluator = pytrec_eval.RelevanceEvaluator(dict(qrels), measures)
    per_query = evaluator.evaluate(merged)

    # queries with no search results (empty bucket / failed run) score 0, and
    # must be counted -- silently dropping them would inflate the joint number
    missing = set(qrels) - set(merged)
    results = {}
    for m in args.eval_metrics:
        name, _, k = m.rpartition("_at_")
        pk = f"{name}_cut_{k}" if name == "ndcg" else f"{name}_{k}"
        total = sum(q.get(pk, 0.0) for q in per_query.values())
        results[m] = total / max(len(qrels), 1)   # denominator includes missing

    # ---- report ----
    dist = Counter(routing.values())
    lines = []
    lines.append(f"# Hydra eval: {args.run_name}")
    lines.append(f"dataset: {args.dataset_dir}   queries: {n_queries:,}")
    lines.append("")
    lines.append("## Routing distribution")
    for c in CATEGORIES:
        lines.append(f"  {c:8s} {dist.get(c, 0):>7,}  ({dist.get(c, 0) / max(n_queries,1):.1%})")
    lines.append("")
    lines.append("## Joint end-to-end metrics (all queries, routed)")
    for m, v in results.items():
        lines.append(f"  {m:15s} {v * 100:.2f}")
    if missing:
        lines.append(f"  [!] {len(missing):,} queries in qrels had no search results "
                     f"(scored 0): first few {sorted(missing)[:5]}")
    lines.append("")
    lines.append("Per-bucket CONDITIONAL results (given routing) are in "
                 "bucket_*/eval_results_*.md")

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(args.workdir, "hydra_results.md"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.workdir, "hydra_results.json"), "w") as f:
        json.dump({"metrics": results,
                   "routing_distribution": {c: dist.get(c, 0) for c in CATEGORIES},
                   "n_queries": n_queries,
                   "n_missing": len(missing)}, f, indent=2)
    print(f"\n[done] {os.path.join(args.workdir, 'hydra_results.md')}")


# ---------------------------------------------------------------------------
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True,
                    help="dir with test_queries.jsonl / test_qrels.jsonl / corpus.jsonl")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--classifier", default="qwen35_4b_category6/best_model")
    ap.add_argument("--classify_py", default="classify.py",
                    help="path to the classify.py inference tool")
    ap.add_argument("--classifier_batch_size", type=int, default=8)
    ap.add_argument("--classifier_no_4bit", action="store_true")
    for c in CATEGORIES:
        ap.add_argument(f"--embedder_{SHORT[c]}", required=True,
                        help=f"path to the {c} specialist embedder")
    ap.add_argument("--devices", nargs="+", default=["cuda:0"])
    ap.add_argument("--search_top_k", type=int, default=1000)
    ap.add_argument("--embedder_batch_size", type=int, default=32)
    ap.add_argument("--embedder_model_class", default="decoder-only-base")
    ap.add_argument("--eval_metrics", nargs="+", default=["ndcg_at_10", "recall_at_100"])
    ap.add_argument("--skip_routing", action="store_true")
    ap.add_argument("--force_search", action="store_true")
    args = ap.parse_args()

    if args.run_name is None:
        args.run_name = os.path.basename(os.path.normpath(args.dataset_dir))
    os.makedirs(args.workdir, exist_ok=True)

    embedders = {c: getattr(args, f"embedder_{SHORT[c]}") for c in CATEGORIES}

    queries = read_jsonl(os.path.join(args.dataset_dir, "test_queries.jsonl"))
    qrels_rows = read_jsonl(os.path.join(args.dataset_dir, "test_qrels.jsonl"))
    print(f"[load] {len(queries):,} queries, {len(qrels_rows):,} qrels rows")

    routing = run_routing(args, queries)
    bucket_dirs = make_buckets(args, queries, qrels_rows, routing)
    run_bucket_evals(args, bucket_dirs, embedders)
    merge_and_score(args, bucket_dirs, qrels_rows, routing, len(queries))


if __name__ == "__main__":
    main()
