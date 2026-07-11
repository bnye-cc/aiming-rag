import os
import re
import json
import sys
import random
import hashlib


STATEMENTS_IN = "statements_start-1900_v3.jsonl"
PAIRS_IN      = "pairs_start-1900_replaced_v1.jsonl"
NEGATIVES_OUT = "hard_negatives_start-1900_replaced_v1.jsonl"

MAX_NEGATIVES_PER_POSITIVE = 32
RANDOM_SAMPLE_WHEN_CAPPED  = True

SKIP_MALFORMED_NEGATIVES   = True
MIN_NEG_LEN                = 25


LABEL_PATTERN = re.compile(r"\\label\{[^{}]+\}")
WS = re.compile(r"\s+")
ENV = re.compile(r"\\(begin|end)\{([^{}]+)\}")


def clean_text(text):
    text = LABEL_PATTERN.sub("", text)
    return WS.sub(" ", text).strip()


def env_balanced(text):
    """Stack-check \\begin/\\end. False on underflow or leftover."""
    stack = []
    for m in ENV.finditer(text):
        kind, env = m.group(1), m.group(2).rstrip("*")
        if kind == "begin":
            stack.append(env)
        else:
            if not stack or stack[-1] != env:
                return False
            stack.pop()
    return not stack


def build_paper_index(statements_path):
    """paper_id -> list of (stmt_id, byte_offset) for EVERY statement"""
    paper_stmts = {}
    n_lines = 0
    with open(statements_path, "r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            n_lines += 1
            rec = json.loads(line)
            pid = rec["paper_id"]
            paper_stmts.setdefault(pid, []).append((rec["stmt_id"], offset))
    print(f"[pass 1] statements read: {n_lines:,}, papers: {len(paper_stmts):,}")
    return paper_stmts


def fetch_statement(statements_file, offset):
    statements_file.seek(offset)
    return json.loads(statements_file.readline())


def pick_negatives(eligible, pair_id):
    """eligible is a list of (stmt_id, offset). Cap to MAX_NEGATIVES_PER_POSITIVE."""
    if len(eligible) <= MAX_NEGATIVES_PER_POSITIVE:
        return eligible
    if RANDOM_SAMPLE_WHEN_CAPPED:
        seed = int(hashlib.md5(pair_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        return rng.sample(eligible, MAX_NEGATIVES_PER_POSITIVE)
    return eligible[:MAX_NEGATIVES_PER_POSITIVE]


def mine_negatives(statements_path, pairs_path, negatives_path):
    paper_stmts = build_paper_index(statements_path)
    sfile = open(statements_path, "r", encoding="utf-8")

    stats = dict(
        pairs_read=0, negatives_written=0,
        pairs_with_zero_negatives=0, pairs_capped=0,
        paper_not_found=0, malformed_skipped=0, short_skipped=0,
        missing_invoked_field=0, missing_copos_field=0,
    )

    with open(pairs_path, "r", encoding="utf-8") as pf, \
         open(negatives_path, "w", encoding="utf-8") as out:

        for line in pf:
            pair = json.loads(line)
            stats["pairs_read"] += 1

            pid = pair["paper_id"]
            pair_id = pair["pair_id"]
            positive_id = pair["positive_stmt_id"]

            if "invoked_stmt_ids" not in pair:
                stats["missing_invoked_field"] += 1
            if "co_positive_ids" not in pair:
                stats["missing_copos_field"] += 1

            exclude = {positive_id}
            exclude.update(pair.get("co_positive_ids") or [])
            exclude.update(pair.get("invoked_stmt_ids") or [])

            all_stmts = paper_stmts.get(pid)
            if not all_stmts:
                stats["paper_not_found"] += 1

            eligible = [(sid, off) for (sid, off) in (all_stmts or [])
                        if sid not in exclude]
            eligible_count = len(eligible)
            if eligible_count > MAX_NEGATIVES_PER_POSITIVE:
                stats["pairs_capped"] += 1

            chosen = pick_negatives(eligible, pair_id)

            negatives = []
            for sid, off in chosen:
                rec = fetch_statement(sfile, off)
                text = clean_text(rec.get("text", ""))
                if len(text) < MIN_NEG_LEN:
                    stats["short_skipped"] += 1
                    continue
                if SKIP_MALFORMED_NEGATIVES and not env_balanced(text):
                    stats["malformed_skipped"] += 1
                    continue
                negatives.append({
                    "stmt_id": sid,
                    "type": rec.get("type"),
                    "text": text,
                })

            if not negatives:
                stats["pairs_with_zero_negatives"] += 1
            stats["negatives_written"] += len(negatives)

            out.write(json.dumps({
                "pair_id": pair_id,
                "paper_id": pid,
                "positive_stmt_id": positive_id,
                "num_eligible": eligible_count,   # pool size before the cap, for auditing
                "negatives": negatives,
            }, ensure_ascii=False) + "\n")

    sfile.close()

    print("\n--- Phase 5 (same-paper hard negatives) Complete ---")
    for k, v in stats.items():
        print(f"{k:>28}: {v:,}")
    if stats["pairs_read"]:
        avg = stats["negatives_written"] / stats["pairs_read"]
        print(f"{'avg_negatives_per_pair':>28}: {avg:.2f}")
    print(f"Negatives -> {negatives_path}")


if __name__ == "__main__":
    mine_negatives(STATEMENTS_IN, PAIRS_IN, NEGATIVES_OUT)