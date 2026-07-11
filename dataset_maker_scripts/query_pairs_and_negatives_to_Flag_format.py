import json
import random


PAIRS_IN = "pairs_clean5_math_PR.jsonl"#"pairs_clean5_math_CA.jsonl"#GT CO AG ST NT CA ALL
NEGATIVES_IN = "hard_negatives_start-1900_clean5.jsonl"
OUT = "flag_train_start-1900_PR_total.jsonl"


MAX_PAIRS = None         # number of pairs; None = all pairs
RANDOM_SAMPLE = True
SEED = 0
MAX_NEGATIVES_PER_PAIR = None


MIN_NEGATIVES = 4
QUERY_PROMPT = None



def build_neg_index(path):
    index = {}
    with open(path, "r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            index[json.loads(line)["pair_id"]] = offset
    print(f"[index] negatives records: {len(index):,}")
    return index


def fetch_negatives(neg_file, neg_index, pair_id):
    """Return the negatives list for a pair_id, or None if it has no record."""
    offset = neg_index.get(pair_id)
    if offset is None:
        return None
    neg_file.seek(offset)
    return json.loads(neg_file.readline()).get("negatives", [])



def iter_selected_pairs(pairs_path):
    if not RANDOM_SAMPLE:
        # file order; caller stops when it has written MAX_PAIRS
        with open(pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return

    # collect offsets (one int per line), shuffle, replay in random order
    offsets = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(off)
    random.Random(SEED).shuffle(offsets)

    with open(pairs_path, "r", encoding="utf-8") as f:
        for off in offsets:
            f.seek(off)
            yield json.loads(f.readline())



def build_dataset(pairs_path, negatives_path, out_path):
    neg_index = build_neg_index(negatives_path)
    neg_file = open(negatives_path, "r", encoding="utf-8")

    stats = dict(
        pairs_seen=0, written=0,
        dropped_no_neg_record=0, dropped_too_few_neg=0, dropped_empty_text=0,
        neg_equals_pos_removed=0, negatives_written=0,
    )

    with open(out_path, "w", encoding="utf-8") as out:
        for pair in iter_selected_pairs(pairs_path):
            if MAX_PAIRS is not None and stats["written"] >= MAX_PAIRS:
                break
            stats["pairs_seen"] += 1

            query = (pair.get("query") or "").strip()
            pos_text = (pair.get("positive_text") or "").strip()
            if not query or not pos_text:
                stats["dropped_empty_text"] += 1
                continue

            negatives = fetch_negatives(neg_file, neg_index, pair["pair_id"])
            if negatives is None:
                stats["dropped_no_neg_record"] += 1
                continue

            neg_texts = []
            for n in negatives:
                t = (n.get("text") or "").strip()
                if not t:
                    continue
                if t == pos_text:                 # identical text under a different id
                    stats["neg_equals_pos_removed"] += 1
                    continue
                neg_texts.append(t)

            if MAX_NEGATIVES_PER_PAIR is not None:
                neg_texts = neg_texts[:MAX_NEGATIVES_PER_PAIR]

            if len(neg_texts) < MIN_NEGATIVES:
                stats["dropped_too_few_neg"] += 1
                continue

            record = {"query": query, "pos": [pos_text], "neg": neg_texts}
            if QUERY_PROMPT is not None:
                record["prompt"] = QUERY_PROMPT

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1
            stats["negatives_written"] += len(neg_texts)

    neg_file.close()
    report(stats, out_path)


def report(s, out_path):
    print("\n=== FlagEmbedding dataset ===")
    print(f"  pairs seen            : {s['pairs_seen']:,}")
    print(f"  written               : {s['written']:,}")
    if s["written"]:
        print(f"  avg negatives / line  : {s['negatives_written'] / s['written']:.1f}")
    print(f"\n  dropped:")
    print(f"    no negatives record : {s['dropped_no_neg_record']:,}")
    print(f"    too few negatives   : {s['dropped_too_few_neg']:,}  (min {MIN_NEGATIVES})")
    print(f"    empty query/positive: {s['dropped_empty_text']:,}")
    print(f"  neg==pos removed      : {s['neg_equals_pos_removed']:,}")
    if MAX_PAIRS is not None and s["written"] < MAX_PAIRS:
        print(f"\n  [WARN] only {s['written']:,} pairs survived filtering (wanted {MAX_PAIRS:,}) "
              f"-- input exhausted. The category file doesn't have enough usable pairs.")
    print(f"\nTrain file -> {out_path}")


if __name__ == "__main__":
    build_dataset(PAIRS_IN, NEGATIVES_IN, OUT)