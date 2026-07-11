import os
import json
import hashlib
from tqdm import tqdm


MAX_NUM = 10000
POSITIVE_PAIRS =  "pairs_start-1900_clean5.jsonl"#"pairs_clean5_math_GT.jsonl"# CO AG NT ST CA GT ALL
NEGATIVE_PAIRS = "hard_negatives_start-1900_clean5.jsonl"
# Exclude queries appearing in ANY of these training sets
TRAIN_FILES = [
    "flag_train_start-1900_clean5_50k.jsonl",
    "flag_train_start-1900_mathAG_50k.jsonl",
    "flag_train_start-1900_mathCO_50k.jsonl",
    "flag_train_start-1900_mathFA_50k.jsonl",
    "flag_train_start-1900_mathGT_50k.jsonl",
    "flag_train_start-1900_mathNT_50k.jsonl",
    "flag_train_start-1900_mathPR_50k.jsonl",


]

CORPUS_FILE = "corpus.jsonl"
QUERY_FILE = "test_queries.jsonl"
RELEVANCE_FILE = "test_qrels.jsonl"


def _qhash(s):
    return hashlib.sha1((s or "").strip().encode("utf-8")).hexdigest()


def load_train_queries(paths):
    paths = [p for p in (paths or []) if p]
    if not paths:
        print("[leakage] no training files given; SKIPPING train/test exclusion "
              "-- eval may overlap training, numbers will be inflated!")
        return set()
    seen = set()
    for path in paths:
        if not os.path.exists(path):
            print(f"[leakage] WARNING: training file not found, skipping: {path!r}")
            continue
        before = len(seen)
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, unit="line", desc=f"train queries [{os.path.basename(path)}]"):
                line = line.strip()
                if not line:
                    continue
                try:
                    seen.add(_qhash(json.loads(line)["query"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[leakage]   +{len(seen) - before:,} new from {os.path.basename(path)}")
    print(f"[leakage] {len(seen):,} unique training queries to exclude (union of {len(paths)} files)")
    return seen


def read_jsonl():
    train_q = load_train_queries(TRAIN_FILES)
    stmts = {}                 # stmt_id -> corpus docid
    kept_qids = set()
    n_q = n_skip_leak = n_qrel = 0
    exhausted = True

    with open(QUERY_FILE, "w", encoding="utf-8") as qf, \
         open(CORPUS_FILE, "w", encoding="utf-8") as af, \
         open(RELEVANCE_FILE, "w", encoding="utf-8") as rf:

        with open(POSITIVE_PAIRS, "r", encoding="utf-8") as pp:
            for line_num, line in enumerate(tqdm(pp, unit="line", desc="positives"), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[warn] bad JSON pos line {line_num}: {e}")
                    continue

                pair_id = obj["pair_id"]
                qid = pair_id[pair_id.find("pair") + 4:]
                if _qhash(obj["query"]) in train_q:
                    n_skip_leak += 1
                    continue

                kept_qids.add(qid)
                qf.write(json.dumps({"id": qid, "text": obj["query"]}, ensure_ascii=False) + "\n")

                sid = obj["positive_stmt_id"]
                if sid not in stmts:
                    stmts[sid] = len(stmts)
                    af.write(json.dumps({"id": str(stmts[sid]), "title": "",
                                         "text": obj["positive_text"]}, ensure_ascii=False) + "\n")
                rf.write(json.dumps({"qid": qid, "docid": str(stmts[sid]),
                                     "relevance": 1}, ensure_ascii=False) + "\n")
                n_q += 1
                n_qrel += 1
                if n_q >= MAX_NUM:                # hit the target numb
                    exhausted = False
                    break


        with open(NEGATIVE_PAIRS, "r", encoding="utf-8") as npf:
            for line_num, line in enumerate(tqdm(npf, unit="line", desc="negatives"), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[warn] bad JSON neg line {line_num}: {e}")
                    continue

                pair_id = obj["pair_id"]
                qid = pair_id[pair_id.find("pair") + 4:]
                if qid not in kept_qids:          # one check covers out-of-range AND leaked queries
                    continue
                for neg in obj["negatives"]:
                    sid = neg["stmt_id"]
                    if sid not in stmts:
                        stmts[sid] = len(stmts)
                        af.write(json.dumps({"id": str(stmts[sid]), "title": "",
                                             "text": neg["text"]}, ensure_ascii=False) + "\n")
                    # deliberately NO relevance:0 row here -- see module docstring

    print("\n=== eval set ===")
    print(f"  queries written          : {n_q:,}   (target {MAX_NUM:,})")
    print(f"  skipped (in training)     : {n_skip_leak:,}   <- leakage prevented, did NOT reduce count")
    print(f"  corpus docs              : {len(stmts):,}")
    print(f"  qrels (relevance=1)      : {n_qrel:,}")
    if exhausted and n_q < MAX_NUM:
        print(f"  [WARN] input exhausted before reaching target: only {n_q:,} non-leaked pairs "
              f"available (wanted {MAX_NUM:,}). Use a larger POSITIVE_PAIRS or lower MAX_NUM.")
    if train_q and n_skip_leak == 0:
        print("  [check] 0 training overlaps -- confirm TRAIN_FILES are what you actually trained on")


if __name__ == "__main__":
    read_jsonl()