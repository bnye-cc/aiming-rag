import os
from logging import exception
import regex as re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList
import signal
import sys
from tqdm import tqdm
import multiprocessing as mp
import queue as pyqueue
import threading
from collections import Counter

NUM_GPUS = 4



MY_HF_TOKEN = "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"
MODEL_NAME = "Qwen/Qwen3.5-9B"#"Qwen/Qwen2.5-14B-Instruct"   # or "mistralai/Mistral-Nemo-Instruct-2407"


TREE_FILE = "trees_statements_v3_extra.jsonl"
ROOT_FOLDER = "/home/aiming/PycharmProjects/PythonProject/math_src_EXTRACTED_start-1900"
STATEMENTS_IN = "extra_statements_start-1900_v3.jsonl"
SELF_ENCAPSULATED_THEOREMS = "full_theorems.jsonl"
#DEFINITIONS_IN = "definitions_v1.jsonl"

class ForceEndThinkLogitsProcessor(LogitsProcessor):
    def __init__(self, max_think_tokens, end_think_token_id):
        self.max_think_tokens = max_think_tokens
        self.end_think_token_id = end_think_token_id
        self.current_tokens = 0
        self.completed = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Check if the model naturally output the closing tag on the very last step
        if input_ids[0, -1].item() == self.end_think_token_id:
            self.completed = True

        self.current_tokens += 1

        # If we hit the limit, AND the model hasn't closed the tag yet, force it!
        if self.current_tokens >= self.max_think_tokens and not self.completed:
            mask = torch.full_like(scores, float('-inf'))
            mask[:, self.end_think_token_id] = 0
            self.completed = True  # Flag it so we don't force it multiple times
            return mask

        return scores


def get_all_dependencies(stmt_id, dependency_tree, statements):
    deps_by_id = {node['stmt_id']: node.get('depends_on', [])
                  for node in dependency_tree}
    text_by_id = {s['stmt_id']: s['text'] for s in statements}

    ordered = []          # dependency ids in discovery (breadth-first) order
    seen = {stmt_id}      # don't revisit; ignore the start node itself
    frontier = list(deps_by_id.get(stmt_id, []))  # distance-1 (direct) deps

    while frontier:
        next_frontier = []
        for dep in frontier:
            if dep in seen:
                continue
            seen.add(dep)
            ordered.append(dep)
            next_frontier.extend(deps_by_id.get(dep, []))
        frontier = next_frontier

    return [text_by_id[dep] for dep in ordered if dep in text_by_id]


def clean_and_verify_json(raw_output: str):
    """
    Strips the thinking block, extracts the JSON array, and verifies it.
    Returns: (is_valid: bool, parsed_data: list | None)
    """
    # 1. Drop everything before and including the closing think tag
    if "</think>" in raw_output:
        # Split by the tag and take the absolute last part
        # (This handles cases where the model hallucinates multiple tags)
        clean_text = raw_output.split("</think>")[-1].strip()
    else:
        # Fallback just in case the model didn't output the tag
        clean_text = raw_output.strip()

    # 2. Extract strictly from the first '[' to the last ']'
    # re.DOTALL allows the regex to match across newlines
    match = re.search(r'\[.*\]', clean_text, re.DOTALL)

    if not match:
        print("Verification Failed: No JSON array brackets found after </think>.")
        return None

    json_string = match.group(0)

    # 3. Verify it is valid, parseable JSON
    try:
        parsed_json = json.loads(json_string)

        # Extra safety check: ensure the root object is actually a list
        if isinstance(parsed_json, list):
            return parsed_json
        else:
            print("Verification Failed: JSON was parsed, but it is not a list/array.")
            return None

    except json.JSONDecodeError as e:
        print(f"Verification Failed: Invalid JSON syntax. Error: {e}")
        return None



def load_model_and_tokenizer(gpu_id):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=MY_HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
        device_map={"": gpu_id},              # was "auto"
        token=MY_HF_TOKEN, quantization_config=quantization_config,
        attn_implementation="sdpa",
    )
    return model, tokenizer


def extract_definitions(main_stat, context_stats, model, tokenizer):
    """
    Uses the LLM to extract mathematical definitions from a LaTeX string.
    """
    example_mstate = {'stmt_id': '1705_10703::s0002', 'type': 'Lemma', 'text': '\\label{lem_ATTO_rank2_aux}\nIf $\\varphi=\\overline{\\chi}+\\psi$, where $\\chi\\in K_\\alpha$ and $\\psi\\in K_\\beta$, $\\psi(0)=0$, then the equality\n\\begin{align*}\n  \\langle A_\\varphi^{\\alpha,\\beta}f,g\\rangle=\\sum_{n=1}^\\infty\\langle(S^n_\\beta \\psi\\otimes S^n_\\alpha k^\\alpha_0 +S^n_\\beta k^\\beta_0\\otimes S^n_\\alpha \\chi)f,g\\rangle\n\\end{align*}\nholds for all $f\\in K_\\alpha^\\infty$ and $g\\in K_\\beta^\\infty$.'}
    example_cstate = ['As usual, let $H^2$ denote the classical Hardy space. The space $H^2$ can be seen as a space of functions analytic in the unit disk\n    \\mathbb{D}=\\{z:|z|<1\\}$ or as a closed\n    subspace of $L^2:=L^2(\\partial\\mathbb{D})$. In the first case $H^2$ consists of functions analytic in $\\mathbb{D}$ with square summable MacLaurin coefficients and in the second it consists of functions from $L^2$ such that their Fourier coefficients with negative indices vanish.', 'The unilateral shift $S$ on $H^2$ is the operator of multiplication by the independent variable, that is,\n      $$Sf(z)=z\\cdot f(z).$$\n    The adjoint of $S^*$ of $S$ is called the backward shift. A simple verification shows that\n      $$S^*f(z)=\\frac{f(z)-f(0)}{z}.$$', 'The famous Beurling theorem provides a characterization of all $S^*$-invariant subspaces of $H^2$. Namely, a closed nontrivial subspace of $H^2$ is $S^*$-invariant if and only if it is of the form\n      $$K_\\alpha=H^2\\ominus \\alpha H^2,$$\n    where $\\alpha$ is an inner function, i.e., $\\alpha$ belongs to the algebra $H^{\\infty}$ of bounded analytic functions and $|\\alpha|=1$ a.e. on $\\partial\n    \\mathbb{D}$. The space $K_{\\alpha}$ is called the model space associated with $\\alpha$.', 'Truncated Toeplitz operators are compressions of Toeplitz operators to model spaces. More precisely, a truncated Toeplitz operator $A_{\\varphi}^{\\alpha}$ with a symbol $\\varphi\\in L^2$ is defined on the model space $K_{\\alpha}$ by\n      $$A_{\\varphi}^{\\alpha}f=P_{\\alpha}(\\varphi f),$$\n    where $P_{\\alpha}$ is the orthogonal projection from $L^2$ onto $K_{\\alpha}$. In particular, $S_\\alpha=A^\\alpha_z$ is called the compressed shift. Since $K_\\alpha$ is $S^*$-invariant, it easily follows that $S^*_\\alpha=S^*_{|K_\\alpha}$.', 'Let $\\alpha$, $\\beta$ be two inner functions. An asymmetric\n    truncated Toeplitz operator $A_{\\varphi}^{\\alpha,\\beta}$ with a symbol $\\varphi\\in\n    L^2$ is the\n    operator from $K_{\\alpha}$ into $K_{\\beta}$ given by\n      $$A_{\\varphi}^{\\alpha,\\beta}f=P_{\\beta}(\\varphi f).$$\n    Clearly, $A_{\\varphi}^{\\alpha}=A_{\\varphi}^{\\alpha,\\alpha}$.\n    Put\n      $$\\mathscr{T}(\\alpha,\\beta)=\\{A_{\\varphi}^{\\alpha,\\beta}\\colon\\, \\varphi\\in\n      L^2\\ \\mathrm{and}\\ A_{\\varphi}^{\\alpha,\\beta}\\\n      \\mathrm{is\\ bounded}\\}.$$', 'Recall that  a model space $K_{\\alpha}$ is a reproducing kernel Hilbert space. That is to say that for every $f$ in the model space $K_{\\alpha}$ and each $w\\in\\mathbb{D}$,\n      $$f(w)=\\langle f, k_{w}^{\\alpha}\\rangle,$$\n    where the reproducing kernel function $k_{w}^{\\alpha}$ is of the form\n      $$k_{w}^{\\alpha}(z)=\\frac{1-\\overline{\\alpha(w)}\\alpha(z)}{1-\\overline{w}z}.$$\n      Observe that since $k_{w}^{\\alpha}\\in H^{\\infty}$, the set $K_{\\alpha}^{\\infty}=K_{\\alpha}\\cap H^{\\infty}$ is dense in $K_{\\alpha}$.']
    example_fullstate = r"""Let $H^2$ denote the classical Hardy space on the unit disk $\mathbb{D}$. For any inner function $\alpha$, let $K_\alpha = H^2 \ominus \alpha H^2$ be the associated model space, $P_\alpha$ the orthogonal projection from $L^2$ onto $K_\alpha$, and $k^\alpha_w$ the reproducing kernel for $K_\alpha$ at $w \in \mathbb{D}$. Let $K_\alpha^\infty = K_\alpha \cap H^\infty$ denote the space of bounded functions in $K_\alpha$, and let $S_\alpha f = P_\alpha(z f)$ denote the compressed shift. For two inner functions $\alpha$ and $\beta$, and a symbol $\varphi \in L^2$, let $A_{\varphi}^{\alpha,\beta} \colon K_\alpha \to K_\beta$ be the asymmetric truncated Toeplitz operator defined by $A_{\varphi}^{\alpha,\beta}f = P_\beta(\varphi f)$."""

    new_messages = [
        {
            "role": "system",
            "content": "You are an expert mathematician and academic editor specializing in LaTeX. Your task is to generate ONLY a mathematical preamble that sets up the required context for a target statement, based on a provided list of definitions.\n\nINPUT FORMAT:\n1. A target statement (Theorem, Lemma, Corollary, etc.). This is provided strictly so you know WHAT needs to be defined.\n2. A list of foundational definitions (dependencies) extracted from the original paper.\n\nYOUR OBJECTIVE:\nWrite a concise, rigorous mathematical preamble that introduces all necessary spaces, operators, functions, and variables from the dependency list that appear in the target statement. \n\nCRITICAL RULE - PREAMBLE ONLY:\nYou must ONLY generate the setup paragraph. DO NOT output the target statement itself. Do not attempt to rewrite, include, or append the target statement. Your output will be programmatically concatenated before the target statement in a downstream pipeline, so your text must end where the target statement begins.\n\nCONSTRAINTS & RULES:\n1. SYNTHESIZE, DO NOT CONCATENATE: Do not simply paste the dependency paragraphs verbatim. Distill them into a fluid, standard mathematical setup (e.g., 'Let $X$ denote...', 'For any inner function $\\alpha$, let $K_\\alpha$...').\n2. STRICT SCOPE: Only define terms that are explicitly present in the provided dependencies list and required by the target statement. Do not hallucinate external context.\n3. NO CHAT OR FILLER: Do not include conversational filler like 'Here is the preamble:' or 'Let's set this up.'\n4. NO FORMATTING WRAPPERS: Output STRICTLY the raw LaTeX text. Do NOT wrap the output in markdown code blocks (```latex ... ```), do NOT use blockquotes (>), and do NOT output JSON. Output only the final raw text string.\n\nEXPECTED OUTPUT:\n[A single concise mathematical paragraph defining the necessary context, and absolutely nothing else.]"
        },
        {
            "role": "user",
            "content": f"""---
        **Main Statement**:
        {json.dumps(example_mstate, indent=2)}
        ---
        
        ---
        **Context Statements**:
        {json.dumps(example_cstate, indent=2)}
        ---
        """
        },
        {
            "role": "assistant",
            "content": example_fullstate
        },
        {
            "role": "user",
            "content": f"""Excellent, that's the exact correct format. Now, apply the exact same process to the following new statements.

        ---
        **Main Statement**:
        {json.dumps(main_stat, indent=2)}
        ---
        
        ---
        **Context Statements**:
        {json.dumps(context_stats, indent=2)}
        ---
        """
        }
    ]

    inputs = tokenizer.apply_chat_template(
        new_messages,
        add_generation_prompt=True,
        return_tensors="pt",
        #enable_thinking=False,
        return_dict=True
    )

    # Move the dictionary of tensors to the GPU
    inputs = inputs.to(model.device)

    # 2. Extract token count safely
    num_input_tokens = inputs["input_ids"].shape[1]
    print(f"Number of tokens in inputs: {num_input_tokens}")

    if num_input_tokens > 150000:
        raise Exception("Token count exceeded 100k limit!")

    print("Generating...")
    end_think_token_id = tokenizer.encode("</think>", add_special_tokens=False)[-1]

    # Initialize the processor. Let's give it 5,000 tokens of "thinking budget".
    # Since max_new_tokens is 8192, this guarantees at least 3,192 tokens remain for the JSON.
    logits_processor = LogitsProcessorList([
        ForceEndThinkLogitsProcessor(max_think_tokens=10000, end_think_token_id=end_think_token_id)
    ])

    # 3. Generate (Notice the ** unpacking operator, this passes both input_ids and attention_mask!)
    outputs = model.generate(
        **inputs,
        max_new_tokens=25192,
        logits_processor=logits_processor,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # 4. Decode ONLY the newly generated tokens
    result = tokenizer.decode(outputs[0][num_input_tokens:], skip_special_tokens=True)
    #cleaned = clean_and_verify_json(result)
    if "</think>" in result:
        result = result.split("</think>")[-1]

    return result




def worker(gpu_id, task_queue, result_queue):
    try:
        torch.cuda.set_device(gpu_id)
        model, tokenizer = load_model_and_tokenizer(gpu_id)
    except Exception as e:
        result_queue.put(("FATAL", gpu_id, repr(e)))
        return
    result_queue.put(("READY", gpu_id))
    while True:
        item = task_queue.get()
        if item is None:
            break
        stmt_id, state, deps = item
        try:
            pre = extract_definitions(state, deps, model, tokenizer)
            if pre is None:
                result_queue.put(("SKIP", stmt_id, "oversize"))
            elif not pre.strip():
                result_queue.put(("SKIP", stmt_id, "empty"))
            else:
                full = pre.strip() + "\n\n" + state["text"].strip()
                result_queue.put(("OK", {stmt_id: full}))
        except Exception as e:
            result_queue.put(("ERR", stmt_id, str(e)))
    result_queue.put(("DONE", gpu_id))


def iter_tasks(statements, trees, processed):
    """Lazily yield one task per unprocessed ::s statement (deps built on the fly)."""
    for paper_id, stmt_list in statements.items():
        tree = trees.get(paper_id)
        if not tree:
            continue
        for state in stmt_list:
            sid = state["stmt_id"]
            if "::s" in sid and sid not in processed:
                deps = get_all_dependencies(sid, tree, stmt_list)
                yield (sid, state, deps)


def count_tasks(statements, trees, processed):
    """Cheap count pass (no dep building) so tqdm gets a real total."""
    n = 0
    for paper_id, stmt_list in statements.items():
        if not trees.get(paper_id):
            continue
        for state in stmt_list:
            sid = state["stmt_id"]
            if "::s" in sid and sid not in processed:
                n += 1
    return n


def feed(task_queue, task_iter, num_workers):
    for t in task_iter:
        task_queue.put(t)          # blocks when queue is full -> bounded memory
    for _ in range(num_workers):
        task_queue.put(None)


def main():
    processed = set()
    if os.path.exists(SELF_ENCAPSULATED_THEOREMS):
        with open(SELF_ENCAPSULATED_THEOREMS, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    processed.update(json.loads(line).keys())
    print(f"Found {len(processed)} already processed statements. Skipping those...")

    statements = {}
    with open(STATEMENTS_IN, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading Statements"):
            if not line.strip():
                continue
            for papid, stmt_list in json.loads(line).items():
                statements[papid] = stmt_list
    print(f"Loaded {len(statements)} papers' statements")

    trees = {}
    with open(TREE_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading Trees"):
            if not line.strip():
                continue
            for papid, tree in json.loads(line).items():
                trees[papid] = tree
    print(f"Loaded {len(trees)} papers' trees")

    total = count_tasks(statements, trees, processed)
    print(f"{total} statements to process across {NUM_GPUS} GPUs.")

    task_queue = mp.Queue(maxsize=2000)     # bounded -> feeder backpressure
    result_queue = mp.Queue()

    procs = [mp.Process(target=worker, args=(g, task_queue, result_queue))
             for g in range(NUM_GPUS)]
    for p in procs:
        p.start()

    feeder = threading.Thread(
        target=feed,
        args=(task_queue, iter_tasks(statements, trees, processed), NUM_GPUS),
        daemon=True,
    )
    feeder.start()

    alive = NUM_GPUS
    counts = Counter()
    with open(SELF_ENCAPSULATED_THEOREMS, "a", encoding="utf-8") as pf, \
         tqdm(total=total, desc="generating") as pbar:
        while alive > 0:
            try:
                msg = result_queue.get(timeout=300)
            except pyqueue.Empty:
                if not any(p.is_alive() for p in procs):
                    break
                continue
            tag = msg[0]
            if tag == "OK":
                pf.write(json.dumps(msg[1], ensure_ascii=False) + "\n"); pf.flush()
                counts["ok"] += 1; pbar.update(1)
            elif tag == "SKIP":
                counts[msg[2]] += 1; pbar.update(1)
            elif tag == "ERR":
                print(f"\n[ERROR] {msg[1]}: {msg[2]}")
                counts["err"] += 1; pbar.update(1)
            elif tag == "READY":
                print(f"GPU {msg[1]} ready.")
            elif tag == "FATAL":
                print(f"\n[FATAL] GPU {msg[1]} could not load model: {msg[2]}")
                alive -= 1
            elif tag == "DONE":
                alive -= 1

    task_queue.cancel_join_thread()
    result_queue.cancel_join_thread()
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    print("Preamble generation complete:", dict(counts))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
