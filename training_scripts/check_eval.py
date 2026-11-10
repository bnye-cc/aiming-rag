"""
checks stats qabout eval dicts
"""

import os
import sys
import json
from collections import Counter


EVAL_DIR = "datadictBALALL"          # directory holding the three files
EXPECT   = None     

CORPUS_NAME  = "corpus.jsonl"
QUERY_NAME   = "test_queries.jsonl"
QRELS_NAME   = "test_qrels.jsonl"


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] bad JSON in {os.path.basename(path)} line {line_num}: {e}")
    return rows


def dup_ids(rows, key):
    c = Counter(r[key] for r in rows if key in r)
    return {k: n for k, n in c.items() if n > 1}


def main():
    cpath = os.path.join(EVAL_DIR, CORPUS_NAME)
    qpath = os.path.join(EVAL_DIR, QUERY_NAME)
    rpath = os.path.join(EVAL_DIR, QRELS_NAME)

    corpus  = load_jsonl(cpath)
    queries = load_jsonl(qpath)
    qrels   = load_jsonl(rpath)

    print(f"\n=== {EVAL_DIR} ===")
    for label, rows, path in [("corpus", corpus, cpath),
                              ("queries", queries, qpath),
                              ("qrels", qrels, rpath)]:
        if rows is None:
            print(f"  [MISSING] {path}")
    if queries is None or qrels is None or corpus is None:
        print("  -> can't run consistency checks until all three files exist.")
        sys.exit(1)

    # --- raw counts ---
    print(f"  corpus docs   : {len(corpus):,}")
    print(f"  queries       : {len(queries):,}")
    print(f"  qrels rows    : {len(qrels):,}")

    # --- id sets ---
    q_ids   = {q["id"] for q in queries}
    doc_ids = {d["id"] for d in corpus}
    qrel_qids  = {r["qid"] for r in qrels}
    qrel_docids = {r["docid"] for r in qrels}

    # --- duplicates ---
    for label, rows, key in [("queries", queries, "id"),
                             ("corpus", corpus, "id")]:
        d = dup_ids(rows, key)
        if d:
            print(f"  [DUP] {label}: {len(d):,} ids appear more than once "
                  f"(e.g. {list(d.items())[:3]})")
    qrel_pairs = Counter((r["qid"], r["docid"]) for r in qrels)
    dup_pairs = {k: n for k, n in qrel_pairs.items() if n > 1}
    if dup_pairs:
        print(f"  [DUP] qrels: {len(dup_pairs):,} (qid,docid) pairs duplicated")

    
    judged_queries = q_ids & qrel_qids
    print(f"  queries WITH >=1 qrel : {len(judged_queries):,}")

    no_qrel = q_ids - qrel_qids
    if no_qrel:
        print(f"  [WARN] {len(no_qrel):,} queries have NO qrel row "
              f"-> not scored, effective eval size is smaller than the query count")

    orphan_qids = qrel_qids - q_ids
    if orphan_qids:
        print(f"  [WARN] {len(orphan_qids):,} qrel qids are NOT in test_queries (orphan judgements)")

    missing_docs = qrel_docids - doc_ids
    if missing_docs:
        print(f"  [ERROR] {len(missing_docs):,} qrel docids are NOT in corpus "
              f"-> gold docs unretrievable (e.g. {list(missing_docs)[:3]})")

    # --- relevance value distribution ---
    rel_vals = Counter(r.get("relevance") for r in qrels)
    print(f"  relevance values: {dict(rel_vals)}")

    # --- expectation check ---
    if EXPECT is not None:
        ok = len(queries) == EXPECT
        print(f"  expected queries: {EXPECT:,}  ->  "
              f"{'OK' if ok else f'MISMATCH (off by {len(queries) - EXPECT:+,})'}")

    print()


if __name__ == "__main__":
    main()
