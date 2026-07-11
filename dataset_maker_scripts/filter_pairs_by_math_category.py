import json


PAIRS_IN  = "pairs_start-1900_clean5.jsonl"
METADATA  = "arxiv-metadata-oai-snapshot.json"
TARGET_CATEGORY = "math.PR"
PAIRS_OUT = f"pairs_clean5_{TARGET_CATEGORY.replace('.', '_')}.jsonl"
PREFIX_MATCH = False



def norm_id(arxiv_id):
    return arxiv_id.replace(".", "_").replace("/", "_")


def in_category(categories_str):
    cats = categories_str.split()
    if PREFIX_MATCH:
        return any(c.startswith(TARGET_CATEGORY) for c in cats)
    return TARGET_CATEGORY in cats


def main():
    # Pass 1: collect normalized paper ids in the target category
    print(f"Scanning metadata for '{TARGET_CATEGORY}' (prefix={PREFIX_MATCH}) ...")
    match_ids = set()
    n_meta = 0
    with open(METADATA, "r", encoding="utf-8") as f:
        for line in f:
            n_meta += 1
            if n_meta % 1_000_000 == 0:
                print(f"  ...{n_meta:,} metadata records, {len(match_ids):,} in category so far")
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if in_category(rec.get("categories") or ""):
                match_ids.add(norm_id(rec["id"]))
    print(f"Done: {len(match_ids):,} papers in '{TARGET_CATEGORY}' (of {n_meta:,} total)\n")
    if not match_ids:
        print("WARNING: no papers matched -- check TARGET_CATEGORY spelling/case. No pairs written.")
        return

    # Pass 2: stream pairs, keep those whose paper_id is in the matched set
    print(f"Filtering pairs from {PAIRS_IN} ...")
    n_pairs = n_kept = 0
    kept_papers = set()
    with open(PAIRS_IN, "r", encoding="utf-8") as fin, open(PAIRS_OUT, "w", encoding="utf-8") as fout:
        for line in fin:
            n_pairs += 1
            if n_pairs % 500_000 == 0:
                print(f"  ...{n_pairs:,} pairs scanned, {n_kept:,} kept")
            if not line.strip():
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = pair.get("paper_id", "")
            if pid in match_ids:
                fout.write(line if line.endswith("\n") else line + "\n")
                n_kept += 1
                kept_papers.add(pid)

    print()
    print(f"================ {TARGET_CATEGORY} ================")
    print(f"pairs in category   : {n_kept:,}")
    print(f"distinct papers     : {len(kept_papers):,}")
    print(f"total pairs scanned : {n_pairs:,}")
    print(f"written to          : {PAIRS_OUT}")


if __name__ == "__main__":
    main()