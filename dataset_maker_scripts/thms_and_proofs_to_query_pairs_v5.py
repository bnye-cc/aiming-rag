import re
import json

STATEMENTS_IN = "statements_start-1900_v3.jsonl"
PROOFS_IN = "proofs_start-1900_v3.jsonl"
PAIRS_OUT = "pairs_start-1900_replaced_v1.jsonl"
#STATEMENTS_IN = "statements_start-1900_sec.jsonl"
#PROOFS_IN = "proof_sections_start-1900_sec.jsonl"

#   "none"    : query ends at the target citation
#   "chars"   : fixed chars past the target
#   "thought" : extend past the target to the next end-of-thought marker
QUERY_AFTER_MODE = "thought"
QUERY_WINDOW_AFTER = 0            # "chars" mode only
AFTER_MAX_CHARS = 700            # "thought" mode char cap if no marker found
INCLUDE_SENTENCE_BOUNDARY = True # treat a sentence end as an end-of-thought

# End-of-thought markers
END_OF_THOUGHT = re.compile(
    r"(?P<blank>\n[ \t]*\n)"
    r"|(?P<lbreak>\\\\)"
    r"|(?P<cmd>\\(?:newline|par|bigskip|medskip|smallskip|vspace|qedhere|qed)\b)"
    r"|(?P<item>\\item\b)"
    r"|(?P<begin>\\begin\{[^{}]*\})"
    r"|(?P<end>\\end\{[^{}]*\})"
    r"|(?P<sent>\.\s+(?=[A-Z(\\$]))"
)


# Length floors in chars
MIN_QUERY_LEN = 80 #40
MIN_POSITIVE_LEN = 30

# filter out queries/positives with less then min words or min math
MIN_QUERY_WORDS = 30#5
MIN_QUERY_MATH = 2#1
MIN_POS_WORDS = 20#5
MIN_POS_MATH = 1

#if it has *_REF_LIMIT redacted refs then must also have *_REF_MIN_WORDS content words
QUERY_REF_LIMIT = 4
QUERY_REF_MIN_WORDS = 4
POS_REF_LIMIT = 2
POS_REF_MIN_WORDS = 4

# A "procedural" query ("same argument as in [TARGET]") is dropped only if, after
# removing the procedural phrase, fewer than this many content words remain.
PROCEDURAL_MIN_WORDS = 20#6

# A CONTENT WORD is an alphabetic token of >= 2 letters that is not a stopword.
# Math spans and LaTeX commands are removed before counting, so only normal english words count
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "by", "with", "as", "is", "are", "be", "we", "it", "its", "that",
    "this", "these", "those", "such", "so", "thus", "hence", "let", "all", "any",
    "can", "will", "have", "has", "from", "at", "which", "where", "there", "no",
    "not", "into", "again", "also", "some", "each", "every", "now", "note",
}

# A MATH EXPRESSION is any one of these delimited spans. Each whole span counts as one expression.
MATH_SPANS = [
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
    re.compile(r"\\begin\{(equation|align|aligned|gather|multline|eqnarray|"
               r"displaymath|array|split)\*?\}.*?\\end\{\1\*?\}", re.DOTALL),
    re.compile(r"\\\(.*?\\\)", re.DOTALL),
    re.compile(r"\$(?:\\\$|[^$])*?\$", re.DOTALL),
]

#patterns around target to filter out
# "proof of [TARGET]"  points at the target's proof so drop.
PROOF_REF = re.compile(
    r"proof\s+of\s+\[TARGET\]|\[TARGET\]\s*'?s?\s+proof",
    re.IGNORECASE,
)
#more stuff like was proved in target
DEFERRAL = re.compile(
    r"(?:was|is|are|were|been|already)\s+"
    r"(?:done|carried\s+out|proved|proven|established|verified|checked|shown|"
    r"treated|handled|discussed|computed|obtained|derived|performed|considered|"
    r"studied|addressed|worked\s+out)\s+(?:in|by)\s+\[TARGET\]",
    re.IGNORECASE,
)
# "same argument as in [TARGET]" -> reuses a technique drop if too few content words as set above.
PROCEDURAL = re.compile(
    r"(?:same|similar|analog(?:ous|y)|identical|like)\s+"
    r"(?:kind\s+of\s+|type\s+of\s+|line\s+of\s+|sort\s+of\s+)?"
    r"(?:argument|reasoning|proof|method|technique|way|manner|lines?|computation|"
    r"calculation|construction|idea|approach)"
    r"|mutatis\s+mutandis|verbatim|along\s+the\s+same\s+lines|as\s+(?:in|above|before)"
    r"|the\s+same\s+(?:proof|argument|way)",
    re.IGNORECASE,
)

# Two citations are CO-CITED when
# the only text between them is glue: whitespace, punctuation, or one of these
# words.
GLUE_WORDS = {
    "and", "together", "with", "combined", "also", "resp", "respectively",
    "as", "well", "plus",
    "lemma", "lemmas", "lemmata", "theorem", "theorems", "thm",
    "proposition", "propositions", "prop", "corollary", "corollaries", "cor",
    "definition", "definitions", "def", "claim", "claims",
    "conjecture", "conjectures", "equation", "equations", "eq", "eqn",
    "identity", "identities", "inequality", "inequalities",
    "formula", "formulae", "formulas", "relation", "relations",
}

# Regex statements for latex stuff

# \ref, \cref, \Cref, \autoref, \eqref, \nameref, \labelcref, \vref
REF_CMD = re.compile(r"\\(?:c|C|auto|eq|name|labelc|v)?ref\*?\{([^{}]+)\}")
# A whole citation plus any leading cue word, so redaction gets the cue too
REDACT = re.compile(
    r"(?:\b(?:by|using|from|applying|see|cf|thanks to|according to)\b[\s~]*)?"
    r"(?:\b(?:Lemmas?|Theorems?|Propositions?|Prop|Corollar(?:y|ies)|Cor|"
    r"Definitions?|Def|Claims?|Conjectures?|Equations?|Eq)\b\.?[\s~]*)?"
    r"\(?~?\\(?:c|C|auto|eq|name|labelc|v)?ref\*?\{[^{}]*\}\)?",
    re.IGNORECASE,
)
LABEL_PATTERN = re.compile(r"\\label\{[^{}]+\}")
REF_IN_POS = re.compile(r"\\(?:c|C|auto|eq|name|labelc|v)?ref\*?\{")
LATEX_CMD = re.compile(r"\\[A-Za-z]+\*?")
WORD = re.compile(r"[A-Za-z]{2,}")
WS = re.compile(r"\s+")
GLUE_PUNCT = re.compile(r"[\s,;&()\[\]{}~]+")
PLACEHOLDERS = ("[REF]", "[TARGET]")


# text measurements used by the cleaning rules
def n_math(text):
    """Number of math expressions (per MATH_SPANS)."""
    count = 0
    work = text
    for rx in MATH_SPANS:
        count += len(rx.findall(work))
        work = rx.sub(" ", work)
    return count


def n_words(text):
    """Number of content words: alphabetic tokens (>= 2 letters, not stopwords),
    after removing placeholders, math, and LaTeX commands."""
    work = text
    for ph in PLACEHOLDERS:
        work = work.replace(ph, " ")
    for rx in MATH_SPANS:
        work = rx.sub(" ", work)
    work = LATEX_CMD.sub(" ", work)
    return sum(1 for w in WORD.findall(work) if w.lower() not in STOPWORDS)


def n_query_refs(query):
    """Number of redacted references in a query ([REF] + [TARGET])."""
    return sum(query.count(ph) for ph in PLACEHOLDERS)


def n_pos_refs(text):
    """Number of surviving \\ref-family commands in a positive statement."""
    return len(REF_IN_POS.findall(text))


def env_balanced(text):
    """Stack-check \\begin/\\end. False on a dangling \\end (truncation artifact)."""
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^{}]+)\}", text):
        kind, env = m.group(1), m.group(2).rstrip("*")
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            return False
    return not stack


def target_context(query, before=80, after=40):
    """Window around the [TARGET] marker, where the phrase rules apply."""
    idx = query.rfind("[TARGET]")
    if idx == -1:
        return query[-before:]
    return query[max(0, idx - before): idx + len("[TARGET]") + after]


def clean_positive(text):
    """Strip inline \\label and collapse whitespace -> the stored positive text."""
    return WS.sub(" ", LABEL_PATTERN.sub("", text)).strip()



#  Co-citation: is the target cited together with another statement?
def _gap_is_glue(gap):
    """True if the text between two citations is only glue (per GLUE_WORDS /
    GLUE_PUNCT). Empty gap (cleveref siblings sharing one command) counts as glue."""
    if len(gap) > 40:
        return False
    tokens = GLUE_PUNCT.sub(" ", gap).strip().lower().split()
    return all(t in GLUE_WORDS for t in tokens)  # all([]) is True -> empty gap is glue


def is_co_cited(refs, sids, k, own_stmt, proof_text):
    """True if the target ref (index k) sits next to another *statement* citation
    separated only by glue -- i.e. they were invoked together as one step."""
    for nb in (k - 1, k + 1):
        if not 0 <= nb < len(refs):
            continue
        nb_sid = sids[nb]
        if nb_sid is None or nb_sid == sids[k] or nb_sid == own_stmt:
            continue
        lo, hi = sorted((k, nb))
        gap = proof_text[refs[lo][2]: refs[hi][1]]  # "" when spans overlap (cleveref)
        if _gap_is_glue(gap):
            return True
    return False



#  Query construction
def after_context(proof_text, start, max_chars=AFTER_MAX_CHARS):
    """Text from `start` to the first END_OF_THOUGHT marker. Tracks \\begin/\\end
    depth so breaks inside a display the target opened don't cut early; includes a
    sentence period; stops before an \\end that closes the target's own env.
    Returns '' if the thought ended at the citation."""
    region = proof_text[start: start + max_chars]
    depth = 0
    cut = len(region)
    for m in END_OF_THOUGHT.finditer(region):
        kind = m.lastgroup
        if kind == "begin":
            depth += 1
        elif kind == "end":
            if depth > 0:
                depth -= 1
            else:
                cut = m.start(); break          # closes the target's own env
        elif kind == "sent":
            if INCLUDE_SENTENCE_BOUNDARY and depth == 0:
                cut = m.start() + 1; break      # include the period
        elif depth == 0:                        # blank / lbreak / cmd / item
            cut = m.start(); break
    return region[:cut]


def build_query(proof_text, ref_start, ref_end):
    """Proof prefix up to the citation (plus optional after-context). Target ref
    -> [TARGET]; other refs -> [REF]; \\label stripped; whitespace collapsed."""
    if QUERY_AFTER_MODE == "chars":
        hi = min(len(proof_text), ref_end + QUERY_WINDOW_AFTER)
    elif QUERY_AFTER_MODE == "thought":
        hi = ref_end + len(after_context(proof_text, ref_end))
    else:  # "none"
        hi = ref_end

    def redact(m):
        return " [TARGET] " if m.start() <= ref_start < m.end() else " [REF] "

    window = REDACT.sub(redact, proof_text[:hi])
    window = LABEL_PATTERN.sub("", window)
    return WS.sub(" ", window).strip()



#  Statement index (offsets, so we never hold the corpus in RAM)
def build_label_index(statements_path):
    """(paper_id, label) -> (stmt_id, byte_offset) over labeled statements only."""
    index = {}
    n_lines = n_labeled = 0
    with open(statements_path, "r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            n_lines += 1
            rec = json.loads(line)
            labels = rec.get("labels") or []
            if labels:
                n_labeled += 1
                for lab in labels:
                    index[(rec["paper_id"], lab)] = (rec["stmt_id"], offset)
    print(f"[pass 1] statements: {n_lines:,} read, {n_labeled:,} labeled, "
          f"{len(index):,} label keys")
    return index


def fetch_statement(statements_file, offset):
    statements_file.seek(offset)
    return json.loads(statements_file.readline())


def iter_refs(proof_text):
    """Yield (label, start, end) for every cited label, splitting cleveref lists
    like \\cref{a,b,c} (siblings share the command's start/end span)."""
    for m in REF_CMD.finditer(proof_text):
        for lab in m.group(1).split(","):
            lab = lab.strip()
            if lab:
                yield lab, m.start(), m.end()



#  Main
def mine_pairs(statements_path, proofs_path, pairs_path):
    index = build_label_index(statements_path)

    counts = dict(
        proofs=0, proofs_with_ref=0, refs_total=0, resolved=0, self_refs=0,
        emitted=0,
        short=0, co_cited=0, thin_query=0, ref_heavy=0,
        proof_ref=0, deferral=0, procedural=0,
        thin_pos=0, malformed_pos=0, ref_dep_pos=0,
        with_after_context=0,
    )
    seen = set()              # dedupe
    positives = set()

    sfile = open(statements_path, "r", encoding="utf-8")
    with open(proofs_path, "r", encoding="utf-8") as pf, \
         open(pairs_path, "w", encoding="utf-8") as out:

        for line in pf:
            proof = json.loads(line)
            counts["proofs"] += 1
            pid = proof["paper_id"]
            ptext = proof["text"]
            own_stmt = proof.get("attached_stmt_id")

            refs = list(iter_refs(ptext))
            if refs:
                counts["proofs_with_ref"] += 1
            counts["refs_total"] += len(refs)

            sids = [(index.get((pid, lab)) or (None,))[0] for (lab, _, _) in refs]
            counts["resolved"] += sum(s is not None for s in sids)
            invoked_stmt_ids = sorted({s for s in sids if s is not None})

            for k, (lab, r_start, r_end) in enumerate(refs):
                target_sid = sids[k]
                if target_sid is None:
                    continue                              # equation/figure/section ref
                if target_sid == own_stmt:
                    counts["self_refs"] += 1              # proof citing its own statement
                    continue
                if is_co_cited(refs, sids, k, own_stmt, ptext):
                    counts["co_cited"] += 1               # jointly-correct -> drop for now
                    continue

                query = build_query(ptext, r_start, r_end)
                if len(query) < MIN_QUERY_LEN:
                    counts["short"] += 1
                    continue

                # ---- query-side rules
                tctx = target_context(query)
                if n_words(query) < MIN_QUERY_WORDS and n_math(query) < MIN_QUERY_MATH:
                    counts["thin_query"] += 1
                    continue
                if n_query_refs(query) >= QUERY_REF_LIMIT and n_words(query) < QUERY_REF_MIN_WORDS:
                    counts["ref_heavy"] += 1
                    continue
                if PROOF_REF.search(tctx):
                    counts["proof_ref"] += 1
                    continue
                if DEFERRAL.search(tctx):
                    counts["deferral"] += 1
                    continue
                if PROCEDURAL.search(tctx) and n_words(PROCEDURAL.sub(" ", query)) < PROCEDURAL_MIN_WORDS:
                    counts["procedural"] += 1
                    continue

                # ---- positive-side rules 
                offset = index[(pid, lab)][1]
                target = fetch_statement(sfile, offset)
                positive = clean_positive(target["text"])
                if len(positive) < MIN_POSITIVE_LEN:
                    counts["short"] += 1
                    continue
                if not env_balanced(positive):
                    counts["malformed_pos"] += 1
                    continue
                if n_words(positive) < MIN_POS_WORDS and n_math(positive) < MIN_POS_MATH:
                    counts["thin_pos"] += 1
                    continue
                if n_pos_refs(positive) >= POS_REF_LIMIT and n_words(positive) < POS_REF_MIN_WORDS:
                    counts["ref_dep_pos"] += 1
                    continue

                key = (query, target_sid)
                if key in seen:
                    continue
                seen.add(key)

                if query[query.rfind("[TARGET]") + len("[TARGET]"):].strip():
                    counts["with_after_context"] += 1

                out.write(json.dumps({
                    "pair_id": f"{pid}::pair{counts['emitted']:07d}",
                    "paper_id": pid,
                    "query": query,
                    "positive_stmt_id": target_sid,
                    "positive_type": target["type"],
                    "positive_text": positive,
                    "source_proof_id": proof["proof_id"],
                    "ref_label": lab,                       # provenance only
                    "invoked_stmt_ids": invoked_stmt_ids,   # Phase 5 exclusion set
                }, ensure_ascii=False) + "\n")
                counts["emitted"] += 1
                positives.add(target_sid)

    sfile.close()
    report(counts, len(positives), pairs_path)


def report(c, n_positives, pairs_path):
    resolved_pct = (100 * c["resolved"] / c["refs_total"]) if c["refs_total"] else 0
    drops = [
        ("short query/positive", c["short"]),
        ("co-cited (has co-positive)", c["co_cited"]),
        ("thin query", c["thin_query"]),
        ("ref-heavy query", c["ref_heavy"]),
        ("proof-of-target ref", c["proof_ref"]),
        ("deferral ref", c["deferral"]),
        ("procedural ref", c["procedural"]),
        ("thin positive", c["thin_pos"]),
        ("malformed positive", c["malformed_pos"]),
        ("ref-dependent positive", c["ref_dep_pos"]),
    ]
    total_dropped = sum(n for _, n in drops)

    print("\n=== Phase 2 ===")
    print(f"  proofs scanned     : {c['proofs']:,}  ({c['proofs_with_ref']:,} cite something)")
    print(f"  refs found         : {c['refs_total']:,}  "
          f"({c['resolved']:,} -> a statement, {resolved_pct:.0f}%)")
    print(f"  self-refs skipped  : {c['self_refs']:,}")
    print(f"\n  pairs emitted      : {c['emitted']:,}")
    print(f"  unique positives   : {n_positives:,}")
    print(f"  with after-context : {c['with_after_context']:,}")
    print(f"\n  dropped ({total_dropped:,} total):")
    for label, n in drops:
        print(f"    {label:<28}{n:>10,}")
    print(f"\nPairs -> {pairs_path}")


if __name__ == "__main__":
    mine_pairs(STATEMENTS_IN, PROOFS_IN, PAIRS_OUT)