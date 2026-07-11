import os
import regex as re
import json
import bisect
import sys
from tqdm import tqdm



ROOT_FOLDER = "/home/harris/PycharmProjects/PythonProject/math_src_EXTRACTED_start-1900"
STATEMENTS_OUT = "statements_start-1900_v3.jsonl"
PROOFS_OUT = "proofs_start-1900_v3.jsonl"
PROOF_SECTIONS_OUT = "proof_sections_start-1900_v3.jsonl"

CONTEXT_GRAB_LEN = 1000
MIN_STMT_LENGTH = 10
MAX_STMT_LENGTH = 3000
MAX_PROOF_LENGTH = 7000
ATTACH_GAP_HIGH_CONF = 1500

STATEMENT_KEYWORDS = (
    "theorem", "lemma", "proposition", "corollary", "definition",
    "claim", "conjecture",
)

DEFAULT_THM_DISPLAY = {
    "theorem": "Theorem", "thm": "Theorem", "mainthm": "Theorem", "maintheorem": "Theorem",
    "lemma": "Lemma", "lem": "Lemma", "lemm": "Lemma",
    "proposition": "Proposition", "prop": "Proposition", "propo": "Proposition",
    "corollary": "Corollary", "cor": "Corollary", "coro": "Corollary",
    "definition": "Definition", "defn": "Definition", "defi": "Definition", "def": "Definition",
    "claim": "Claim", "conjecture": "Conjecture", "conj": "Conjecture",
}

NEWTHEOREM_PATTERN = re.compile(
    r"^\s*\\newtheorem\*?\s*\{([^{}]+)\}\s*(?:\[[^\]]+\])?\s*\{([^{}]+)\}",
    re.MULTILINE,
)
LABEL_PATTERN = re.compile(r"\\label\{([^{}]+)\}")
# Matches a proof environment (with optional [..] title) nested inside a statement body.
PROOF_ENV_PATTERN = re.compile(
    r"\\begin\{proof\}(?:\[[^\]]*\])?.*?\\end\{proof\}",
    re.DOTALL | re.IGNORECASE,
)
# refs inside a \begin{proof}[...] optional title, e.g. "Proof of Theorem~\ref{thm:main}"
REF_PATTERN = re.compile(r"\\(?:c|C|auto)?ref\*?\{([^{}]+)\}")
COMMENT_PATTERN = re.compile(r"(?<!\\)%.*")

PROOF_SECTION_PATTERN = re.compile(r"[Pp]roof\s*of.*\\(?:c|C|auto)?ref\*?\{([^{}]+)\}")

NEW_DEF_PATTERN = re.compile(
    r'\\((?:(?:(?:re)?newcommand)|(?:[ex]?def))\*?)\s*(?:\{?(\\[^{}[\]#]+)\}?)\s*((?:\[\d+\])|(?:(?:\[?#[^#]\]?)*))?\s*(?:\[(?:[^\n]*)?\])?\s*(\{(?>[^{}]+|(?4))*\})',
    re.DOTALL
)


def strip_comments(text):
    """Remove % comments to end of line, leaving escaped \\% intact. Newlines preserved
    so character offsets stay internally consistent within the cleaned text."""
    return COMMENT_PATTERN.sub("", text)


def strip_proofs(text):
    """Remove any \\begin{proof}...\\end{proof} blocks nested inside a statement body,
    so a statement that wraps its own proof doesn't carry the proof into the index."""
    return PROOF_ENV_PATTERN.sub("", text)


def classify_kind(display_label):
    """Map a display label like 'Main Theorem' to a canonical kind, or None to skip."""
    low = display_label.lower()
    for kw in STATEMENT_KEYWORDS:
        if kw in low:
            return kw.capitalize()
    return None


def build_env_map(latex_content):
    """Return {env_name: display_label} discovered from \\newtheorem, merged with defaults."""
    env_map = dict(DEFAULT_THM_DISPLAY)
    for match in NEWTHEOREM_PATTERN.finditer(latex_content):
        env_name, label = match.groups()
        env_name = env_name.strip()
        if not env_name or "\\" in env_name or "%" in env_name:
            continue
        env_map[env_name] = label.strip()
    return env_map


def find_new_def(latex_content):
    """Parses LaTeX content to find def and newcommand.
    Return replacement dict {original: replacement}"""

    matches = re.findall(NEW_DEF_PATTERN, latex_content)
    no_para = filter(lambda m: m[2] == "", matches)
    return {m[1]: m[3] for m in no_para}


def replace_new_def(latex_content, replace_dict):
    """Replace custom defined commands with the base latex.
    Peturn replaced string"""
    if (len(replace_dict)):
        p = "(?:" + "|".join(map(re.escape, replace_dict.keys())) + r")(?![A-Za-z])"
        replace_pattern = re.compile(p)
        return re.sub(replace_pattern, lambda m: replace_dict[m.group(0)], latex_content)

    else:
        return latex_content


def find_environments(latex_content, env_name):
    """Yield (start, end, inner_text) for each \\begin{env}...\\end{env}.
    Supports an optional [..] argument right after \\begin{env}."""
    pattern = re.compile(
        r"\\begin\{" + re.escape(env_name) + r"\}(\[[^\]]*\])?(.*?)\\end\{" + re.escape(env_name) + r"\}",
        re.DOTALL,
    )
    for m in pattern.finditer(latex_content):
        yield m.start(), m.end(), m.group(1), m.group(2)


def extract_statements(latex_content, paper_id, env_map, replace_dict):
    """Find all theorem-like statements. Returns list of dicts (unsorted)."""
    statements = []
    seen_spans = set()  # guard against an env name appearing in both defaults and discovered

    for env, display in env_map.items():
        kind = classify_kind(display)
        if kind is None:
            continue
        try:
            spans = list(find_environments(latex_content, env))
        except re.error:
            continue

        for start, end, _opt_arg, content in spans:
            if (start, end) in seen_spans:
                continue
            seen_spans.add((start, end))

            # Drop any nested proof so the statement text (and its labels) stay clean.
            content = replace_new_def(strip_proofs(content).strip(), replace_dict)
            if not (MIN_STMT_LENGTH < len(content) < MAX_STMT_LENGTH):
                continue

            labels = LABEL_PATTERN.findall(content)
            before = latex_content[max(0, start - CONTEXT_GRAB_LEN):start]
            after = latex_content[end:end + CONTEXT_GRAB_LEN]

            statements.append({
                "paper_id": paper_id,
                "env": env,
                "type": kind,
                "type_display": display,
                "labels": labels,
                "text": content,
                "char_start": start,
                "char_end": end,
                "previous_context": before,
                "following_context": after,
            })
    return statements


def extract_proofs(latex_content, paper_id, replace_dict):
    """Find all proof environments. Returns list of dicts (unsorted)."""
    proofs = []
    for start, end, opt_arg, content in find_environments(latex_content, "proof"):
        content = replace_new_def(content.strip(), replace_dict)
        truncated = False
        if len(content) > MAX_PROOF_LENGTH:
            content = content[:MAX_PROOF_LENGTH]
            truncated = True

        title_arg = opt_arg[1:-1] if opt_arg else None  # strip the surrounding [ ]
        title_refs = REF_PATTERN.findall(title_arg) if title_arg else []

        proofs.append({
            "paper_id": paper_id,
            "title_arg": title_arg,
            "title_refs": title_refs,
            "text": content,
            "char_start": start,
            "char_end": end,
            "truncated": truncated,
        })
    return proofs


def find_sections(latex_content):
    """Yield (start, end, section_level, inner_text) for each sections"""
    pattern = re.compile(
        r'\\((?:section|subsection|subsubsection|paragraph|chapter)\*?)\s*(?:\[[^\]]+\])?(\{((?>[^{}]+|(?2))*)\})',
        re.DOTALL
    )
    for m in pattern.finditer(latex_content):
        yield m.start(), m.end(), m.group(1), m.group(3)


def extract_proof_section(latex_content, paper_id, replace_dict):
    """Find all proof sections. Returns list of dicts (unsorted)."""
    global ignored_counter
    sections = list(find_sections(latex_content))
    # print(sections)
    proof_sections = []
    for i in range(len(sections)):
        refs = PROOF_SECTION_PATTERN.findall(sections[i][3])
        if len(refs):
            start = sections[i][1]
            if i == len(sections) - 1:
                break
            else:
                end = sections[i + 1][0]
            # print(start, end)
            content = replace_new_def(latex_content[start: end].strip(), replace_dict)
            truncated = False
            if len(content) > MAX_PROOF_LENGTH:
                content = content[:MAX_PROOF_LENGTH]
                truncated = True

            proof_sections.append({
                "paper_id": paper_id,
                "title_arg": None,
                "title_refs": refs,
                "text": content,
                "char_start": start,
                "char_end": end,
                "truncated": truncated,
            })
    return proof_sections


def assign_ids_and_attach(statements, proofs, proof_sections, paper_id):
    """Sort by document position, assign stable ids, attach each proof to a statement."""
    statements.sort(key=lambda s: s["char_start"])
    for i, s in enumerate(statements):
        s["stmt_id"] = f"{paper_id}::s{i:04d}"

    # index of statement end positions for nearest-preceding lookup
    stmt_ends = [s["char_end"] for s in statements]

    proofs.sort(key=lambda p: p["char_start"])
    for j, p in enumerate(proofs):
        p["proof_id"] = f"{paper_id}::p{j:04d}"

        # nearest statement whose end is <= proof start
        idx = bisect.bisect_right(stmt_ends, p["char_start"]) - 1
        if idx >= 0:
            target = statements[idx]
            gap = p["char_start"] - target["char_end"]
            p["attached_stmt_id"] = target["stmt_id"]
            p["attached_gap"] = gap
            p["attach_confidence"] = "high" if gap <= ATTACH_GAP_HIGH_CONF else "low"
        else:
            p["attached_stmt_id"] = None
            p["attached_gap"] = None
            p["attach_confidence"] = "none"

        # if the proof explicitly names its target via \begin{proof}[Proof of \ref{...}],
        # that overrides the positional guess in Phase 2
        if p["title_refs"]:
            p["attach_confidence"] = "explicit"

    proof_sections.sort(key=lambda p: p["char_start"])
    for j, p in enumerate(proof_sections):
        p["proof_id"] = f"{paper_id}::ps{j:04d}"

        attached = False
        for stmt in statements:
            if p["title_refs"][0] in stmt["labels"]:
                p["attached_stmt_id"] = stmt["stmt_id"]
                p["attached_gap"] = None
                p["attach_confidence"] = "explicit"
                attached = True
        if not attached:
            p["attached_stmt_id"] = None
            p["attached_gap"] = None
            p["attach_confidence"] = "none"

    return statements, proofs, proof_sections


def read_paper_text(dirpath):
    """Concatenate all .tex/.ltx files in a paper directory (sorted for determinism)."""
    dir_text = ""
    filenames = sorted(f for f in os.listdir(dirpath) if f.lower().endswith((".tex", ".ltx")))
    for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8", errors="surrogateescape") as f:
                dir_text += f.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, "r", encoding="latin-1") as f:
                    dir_text += f.read()
            except Exception as e:
                print(f"\n[WARNING] Could not read file: {filepath}. Reason: {e}", file=sys.stderr)
        except IOError as e:
            print(f"\n[WARNING] IOError reading file: {filepath}. Reason: {e}", file=sys.stderr)
    return dir_text


def process_tex_directories():
    print(f"Starting processing of root folder: {ROOT_FOLDER}")
    if not os.path.exists(ROOT_FOLDER):
        print(f"[ERROR] Root directory not found: {ROOT_FOLDER}", file=sys.stderr)
        return

    paper_dirs = [
        dp for dp, _, fns in os.walk(ROOT_FOLDER)
        if any(f.lower().endswith((".tex", ".ltx")) for f in fns)
    ]
    if not paper_dirs:
        print(f"[ERROR] No subdirectories with .tex files found in {ROOT_FOLDER}.", file=sys.stderr)
        return

    papers_processed = papers_failed = n_statements = n_proofs = n_proof_sections = 0

    with open(STATEMENTS_OUT, "w", encoding="utf-8") as sf, \
            open(PROOFS_OUT, "w", encoding="utf-8") as pf, \
            open(PROOF_SECTIONS_OUT, "w", encoding="utf-8") as ps:

        pbar = tqdm(paper_dirs, unit="paper")
        for dirpath in pbar:
            paper_id = os.path.relpath(dirpath, ROOT_FOLDER)
            pbar.set_description(f"Processing {paper_id}")
            try:
                raw = read_paper_text(dirpath)
                if not raw:
                    continue
                cleaned = raw.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
                cleaned = strip_comments(cleaned)

                replace_dict = find_new_def(cleaned)

                env_map = build_env_map(cleaned)
                statements = extract_statements(cleaned, paper_id, env_map, replace_dict)
                proofs = extract_proofs(cleaned, paper_id, replace_dict)
                proof_sections = extract_proof_section(cleaned, paper_id, replace_dict)
                statements, proofs, proof_sections = assign_ids_and_attach(statements, proofs, proof_sections, paper_id)

                for s in statements:
                    sf.write(json.dumps(s, ensure_ascii=False) + "\n")
                for p in proofs:
                    pf.write(json.dumps(p, ensure_ascii=False) + "\n")
                for p in proof_sections:
                    ps.write(json.dumps(p, ensure_ascii=False) + "\n")

                n_statements += len(statements)
                n_proofs += len(proofs)
                n_proof_sections += len(proof_sections)
                papers_processed += 1
            except Exception as e:
                print(f"\n[CRITICAL ERROR] in {dirpath}: {e}", file=sys.stderr)
                papers_failed += 1

    print("\n\n--- Processing Complete ---")
    print(f"Papers processed:       {papers_processed}")
    print(f"Papers failed:          {papers_failed}")
    print(f"Statements found:       {n_statements}")
    print(f"Proofs found:           {n_proofs}")
    print(f"Proof sections found:   {n_proof_sections}")
    print(f"Statements -> {STATEMENTS_OUT}")
    print(f"Proofs     -> {PROOFS_OUT}")


if __name__ == "__main__":
    process_tex_directories()
#    print( re.findall(r"\\be\b", r"\beta"))