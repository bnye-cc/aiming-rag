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
from collections import Counter

NUM_GPUS = 4


MY_HF_TOKEN = "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"
MODEL_NAME = "Qwen/Qwen3.5-9B"#"Qwen/Qwen2.5-14B-Instruct"   # or "mistralai/Mistral-Nemo-Instruct-2407"


TREE_FILE = "trees_statements_v3_extra.jsonl"
ROOT_FOLDER = "/home/aiming/PycharmProjects/PythonProject/math_src_EXTRACTED_start-1900"
STATEMENTS_IN = "extra_statements_start-1900_v3.jsonl"
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
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=MY_HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map={"": gpu_id},        # was "auto"
        token=MY_HF_TOKEN,
        quantization_config=quantization_config,
        attn_implementation="sdpa",
    )
    return model, tokenizer


def extract_definitions(latex_context, model, tokenizer):
    """
    Uses the LLM to extract mathematical definitions from a LaTeX string.
    """
    example_statements = [{'stmt_id': '0705_4330::s0000', 'type': 'Theorem', 'text': "\\label{LocSymmThm}\nLet $X$ be a finite-volume, noncompact, irreducible, complete, locally symmetric space of noncompact type, with no Euclidean factors {\\rm(}locally\\/{\\rm)}, such that $\\rank X \\ge 2$. Then there is a finite-volume, noncompact, irreducible, complete, locally symmetric space~$X'$, such that $X'$ admits a totally geodesic, proper immersion into~$X$, and the universal cover of~$X'$ is the symmetric space associated to either\\/\n$\\SL_3({{\\mathord{\\mathbb{R}}}})$, $\\SL_3({{\\mathord{\\mathbb{C}}}})$, or a direct product\\/ $\\SL_2({{\\mathord{\\mathbb{R}}}})^m \\times \\SL_2({{\\mathord{\\mathbb{C}}}})^n$, with $m + n \\ge 2$."}, {'stmt_id': '0705_4330::s0001', 'type': 'Definition', 'text': "Let us say that an abstract group~$\\Gamma$ is a \\emph{nonuniform lattice\nof higher rank} if\n there exists a connected, semisimple, linear (real) Lie group~$G$,\n such that\n \\begin{itemize}\n \\item $\\Gamma$ is isomorphic to an irreducible, \\emph{nonuniform}\nlattice in~$G$,\n and\n \\item ${\\fieldrank{\\real}} G \\ge 2$.\n \\end{itemize}\n (Recall that a discrete subgroup~$\\Gamma'$ of~$G$ is a \\emph{nonuniform\nlattice} if $G/\\Gamma'$ has finite volume, but is \\emph{not} compact. The\nlattice~$\\Gamma'$ is \\emph{irreducible} if no finite-index subgroup\nof~$\\Gamma'$ is isomorphic to a direct product $\\Gamma_1' \\times\n\\Gamma_2'$ with both $\\Gamma_1'$ and~$\\Gamma_2'$ infinite.)"}, {'stmt_id': '0705_4330::s0002', 'type': 'Definition', 'text': 'A nonuniform lattice~$\\Gamma$ of higher rank is \\emph{almost minimal} if no\nsubgroup of infinite index in~$\\Gamma$ is a nonuniform lattice of higher rank.'}, {'stmt_id': '0705_4330::s0003', 'type': 'Theorem', 'text': 'Every almost-minimal nonuniform lattice of higher rank is isomorphic to a nonuniform, irreducible lattice in either\\/ $\\SL_3({{\\mathord{\\mathbb{R}}}})$, $\\SL_3({{\\mathord{\\mathbb{C}}}})$, or a direct product\\/ $\\SL_2({{\\mathord{\\mathbb{R}}}})^m \\times \\SL_2({{\\mathord{\\mathbb{C}}}})^n$, with $m + n \\ge 2$.'}, {'stmt_id': '0705_4330::s0004', 'type': 'Theorem', 'text': '\\label{LattThm}\n Every nonuniform lattice of higher rank contains a subgroup that is isomorphic to a finite-index\nsubgroup of a lattice described in Example\\/ \\ref{Qrank2eg}, \\fullref{SL2Egs}{any}, or\\/~\\ref{SU2Egs}.'}, {'stmt_id': '0705_4330::s0005', 'type': 'Theorem', 'text': '\\label{MainQuasisplit}\n Suppose\\/ ${\\mathbf{G}}$ is an isotropic, almost simple algebraic group over\\/~${{\\mathord{\\mathbb{Q}}}}$, such that\\/ ${\\fieldrank{\\real}} {\\mathbf{G}} \\ge 2$. Then\\/ ${\\mathbf{G}}$ has a connected, isotropic, almost simple\\/\n${{\\mathord{\\mathbb{Q}}}}$-subgroup\\/~${\\mathbf{H}}$, such that\\/ ${\\mathbf{H}}$ is quasisplit over\\/~${{\\mathord{\\mathbb{Q}}}}$, and\\/ ${\\fieldrank{\\real}} {\\mathbf{H}} \\ge 2$.'}, {'stmt_id': '0705_4330::s0006', 'type': 'Definition', 'text': 'Suppose ${\\mathbf{G}}$ is an isotropic, almost simple algebraic group over\\/~${{\\mathord{\\mathbb{Q}}}}$, such that\\/ \n${\\fieldrank{\\real}} {\\mathbf{G}} \\ge 2$.\nFor convenience, let us say that ${\\mathbf{G}}$ is \\emph{minimal} if no proper, isotropic, almost simple\\/ ${{\\mathord{\\mathbb{Q}}}}$-subgroup of~${\\mathbf{G}}$ has real rank $\\ge 2$.'}, {'stmt_id': '0705_4330::s0007', 'type': 'Theorem', 'text': '\\label{QgrpExplicitThm}\n Suppose\\/ ${\\mathbf{G}}$ is an isotropic, almost simple algebraic group over\\/~${{\\mathord{\\mathbb{Q}}}}$, such that\\/ \n${\\fieldrank{\\real}} {\\mathbf{G}} \\ge 2$. If\\/ ${\\mathbf{G}}$ is minimal, then\\/ ${\\mathbf{G}}$ is isogenous to either:\n\\begin{enumerate} \\renewcommand{{\\roman{enumi}}}{\\roman{enumi}}\n\\item \\label{QgrpExplicitThm-SL3}\n$\\aSL_3$,\nor\n\\item \\label{QgrpExplicitThm-SUreal}\n $\\aSU_3(L,f,\\tau)$, where $L$ is a real quadratic extension of\\/~${{\\mathord{\\mathbb{Q}}}}$, $\\tau$~is the Galois automorphism of~$L$ over\\/~${{\\mathord{\\mathbb{Q}}}}$, and \n\t\\begin{equation} \\label{QgrpExplicitThm-f}\n\tf(x_1,x_2,x_3) = \\tau(x_1) \\, x_1 - \\tau(x_2) \\, x_2 - \\tau(x_3) \\, x_3 \n\t, \\end{equation}\nor\n\\item \\label{QgrpExplicitThm-SUcplx}\n $\\res{K/{{\\mathord{\\mathbb{Q}}}}} \\aSU_3(L,f,\\tau)$, where $K$ is an imaginary quadratic extension of\\/~${{\\mathord{\\mathbb{Q}}}}$, $L$~is a quadratic extension of~$K$, $\\tau$~is the Galois automorphism of~$L$ over~$K$, and $f$~is given by \\pref{QgrpExplicitThm-f},\nor\n\\item \\label{QgrpExplicitThm-SL2}\n $\\res{K/{{\\mathord{\\mathbb{Q}}}}} \\aSL_2$, for some finite extension~$K$ of\\/~${{\\mathord{\\mathbb{Q}}}}$, such that $K$ is neither\\/~${{\\mathord{\\mathbb{Q}}}}$, nor an imaginary quadratic extension of\\/~${{\\mathord{\\mathbb{Q}}}}$.\n\\end{enumerate}'}, {'stmt_id': '0705_4330::s0008', 'type': 'Definition', 'text': 'We say ${\\mathbf{G}}$ is \\emph{minimal} if ${S_{\\aG}} \\neq \\emptyset$, and there does not exist a proper, isotropic, almost simple $F$-subgroup~${\\mathbf{H}}$ of~${\\mathbf{G}}$, such that ${\\fieldrank{F_v}} {\\mathbf{H}} \\ge 2$ for every $v \\in {S_{\\aG}}$.'}, {'stmt_id': '0705_4330::s0009', 'type': 'Theorem', 'text': '\\label{MainForF}\n Suppose\\/ ${\\mathbf{G}}$ is an isotropic, almost simple algebraic group over an algebraic number field~$F$, such that ${S_{\\aG}} \\neq \\emptyset$. If\\/ ${\\mathbf{G}}$ is minimal, then\\/ ${\\mathbf{G}}$ is isogenous to either:\n\\begin{enumerate} \\renewcommand{{\\roman{enumi}}}{\\roman{enumi}}\n\\item \\label{MainForF-SL3}\n$\\aSL_3$,\nor\n\\item \\label{MainForF-SU}\n $\\aSU_3(L,f,\\tau)$, where \n \\begin{itemize}\n \\item $L$ is a quadratic extension of~$F$, such that $L \\subset F_v$, for some archimedean place~$v$ of~$F$,\n \\item $\\tau$~is the Galois automorphism of~$L$ over~$F$, \n and \n\\item $f(x_1,x_2,x_3) = \\tau(x_1) \\, x_1 - \\tau(x_2) \\, x_2 - \\tau(x_3) \\, x_3 $,\n\\end{itemize}\nor\n\\item \\label{MainForF-SUres}\n $\\res{K/F} \\aSU_3(L,f,\\tau)$, where \n \\begin{itemize}\n \\item $K$ is a quadratic extension of~$F$, such that $K \\not\\subset F_v$, for some archimedean place~$v$ of~$F$,\n \\item $L$~is a quadratic extension of~$K$, \n \\item $\\tau$~is the Galois automorphism of~$L$ over~$K$, \n and \n \\item $f(x_1,x_2,x_3) = \\tau(x_1) \\, x_1 - \\tau(x_2) \\, x_2 - \\tau(x_3) \\, x_3 $,\n \\end{itemize}\nor\n\\item \\label{MainForF-SL2}\n $\\res{K/F} \\aSL_2$, for some nontrivial finite extension~$K$ of~$F$, such that either $|K:F| > 2$, or $K \\subset F_v$, for some archimedean place~$v$ of~$F$.\n\\end{enumerate}'}, {'stmt_id': '0705_4330::s0010', 'type': 'Corollary', 'text': 'Suppose\\/ ${\\mathbf{G}}$ is an isotropic, almost simple algebraic group over an algebraic number field~$F$, such that ${S_{\\aG}} \\neq \\emptyset$. Then ${\\mathbf{G}}$ contains an isotropic, almost simple $F$-subgroup~${\\mathbf{H}}$, such that ${\\fieldrank{F_v}} {\\mathbf{H}} \\ge 2$ for every $v \\in {S_{\\aG}}$, and ${\\mathbf{H}}$ is isogenous to a subgroup described in \\pref{MainForF-SL3}, \\pref{MainForF-SU}, \\pref{MainForF-SUres}, or~\\pref{MainForF-SL2} of Theorem~\\ref{MainForF}.'}, {'stmt_id': '0705_4330::s0011', 'type': 'Lemma', 'text': '\\label{SO=SL_2(F[a])}\n If $a \\in F^*$ and $\\sqrt{a} \\notin F$, then \n \t$$\\aSO_4(x_1^2 - x_2^2 - x_3^2 + a x_4^2) \\approx \\res{F[\\sqrt{a}]/F} \\aSL_2 .$$'}, {'stmt_id': '0705_4330::s0012', 'type': 'Lemma', 'text': '\\label{FrankMust1}\nIf\\/ ${\\mathbf{G}}$ is minimal, then either:\n\\begin{enumerate}\n\\item \\label{FrankMust1-Frank1}\n ${\\fieldrank{F}} {\\mathbf{G}} = 1$,\n or\n\\item \\label{FrankMust1-SL3}\n${\\mathbf{G}}$ is isogenous to $\\aSL_3$ {\\rm(}so \\fullref{MainForF}{SL3} holds\\/{\\rm)}.\n\\end{enumerate}'}, {'stmt_id': '0705_4330::s0013', 'type': 'Lemma', 'text': '\\label{AbsSimple}\nIf\\/ ${\\mathbf{G}}$ is minimal, then either:\n\\begin{enumerate}\n\\item \\label{AbsSimple-SL}\n ${\\mathbf{G}}$ is isogenous to $\\res{K/F} \\aSL_2$, with $K$ as described in Theorem~\\fullref{MainForF}{SL2},\nor\n\\item \\label{AbsSimple-Simple}\n ${\\mathbf{G}}$ is absolutely almost simple, \nor\n\\item \\label{AbsSimple-Imag}\n ${\\mathbf{G}}$ is isogenous to $\\res{K/F}{\\mathbf{G}}_0$, where ${\\mathbf{G}}_0$ is an absolutely almost simple group over a quadratic extension~$K$ of~$F$, such that $K \\not\\subset F_v$, for some $v \\in {S_{\\aG}}$.\n\\end{enumerate}'}, {'stmt_id': '0705_4330::s0014', 'type': 'Lemma', 'text': '\\label{CanAbsSimple}\nIf Theorem~\\ref{MainForF} holds\\/ {\\rm(}for all algebraic number fields\\/{\\rm)} under the additional assumption that\\/ ${\\mathbf{G}}$ is absolutely almost simple, then it holds in general.'}, {'stmt_id': '0705_4330::s0015', 'type': 'Corollary', 'text': '\\label{MostExceptCor}\nIf\\/ ${\\mathbf{G}}$ is minimal, then\\/ ${\\mathbf{G}}$ is not of type $E_7$, $E_8$, or~$G_2$.'}, {'stmt_id': '0705_4330::s0016', 'type': 'Lemma', 'text': "\\label{NormalizeHerm}\nLet\n\\begin{itemize}\n\\item $D$ be a quaternion algebra over a field~$L$,\n\\item $\\tau$ be an involution of~$D$ {\\rm(}of either the first or second kind{\\rm)},\n\\item $f(x,y) = \\tau(x_1) \\, a_1 \\, y_1 + \\tau(x_2) \\, a_2 \\, y_2 + \\cdots + \\tau(x_n) \\, a_n \\, y_n$ be a nondegenerate $\\tau$-Hermitian form on~$D^n$, for some~$n$,\n\\item $d \\in D$, such that $\\tau(d) = d$,\n\\item $\\tau' = {\\mathop{\\rm int}}(d) \\circ \\tau$, where ${\\mathop{\\rm int}}(d)$ is the inner conjugation in~$D$ by~$d$,\nand\n\\item $f'(x,y) = d \\, f(x,y) = \\tau'(x_1) \\, d a_1 \\, y_1 + \\tau'(x_2) \\, d a_2 \\, y_2 + \\cdots + \\tau'(x_n) \\, d a_n \\, y_n$.\n\\end{itemize}\nThen:\n\\begin{enumerate}\n\\item $\\tau'$ is an involution {\\rm(}of the same kind as~$\\tau${\\rm)},\n\\item $f'$ is $\\tau'$-Hermitian,\nand\n\\item $\\aSU_n(D, f', \\tau') = \\aSU_n(D, f, \\tau)$.\n\\end{enumerate}"}, {'stmt_id': '0705_4330::s0017', 'type': 'Definition', 'text': '{[T66]}}]\nRecall that if ${\\mathbf{S}}$ is a maximal $F$-split torus in~${\\mathbf{G}}$,\nthen the semisimple part of the centralizer ${\\mathbf{C}}_{\\mathbf{G}}({\\mathbf{S}})$ is called the\n\\emph{semisimple $F$-anisotropic kernel} of~${\\mathbf{G}}$. It is unique up\nto $F$-isomorphism.'}, {'stmt_id': '0705_4330::s0018', 'type': 'Definition', 'text': 'A connected, semisimple subgroup~${\\mathbf{H}}_0$ of~${\\mathbf{G}}$ is \n \\emph{standard} if ${\\mathbf{H}}_0$ is normalized by a maximal torus~${\\mathbf{T}}$\n of~${\\mathbf{G}}$. (We remark that neither~${\\mathbf{H}}_0$ nor~${\\mathbf{T}}$ is assumed\n to be defined over~$F$.) Equivalently, there exist\n roots $\\beta_1,\\ldots,\\beta_r$ of~${\\mathbf{G}}$ (with respect to~${\\mathbf{T}}$),\n such that ${\\mathbf{H}}_0$ is generated by the root subgroups \n $U_{\\pm \\beta_1}, \\ldots, U_{\\pm \\beta_r}$. \n For short, we may say that ${\\mathbf{H}}_0$ is \\emph{generated by the \n roots $\\pm\\beta_1,\\ldots,\\pm\\beta_r$}.'}, {'stmt_id': '0705_4330::s0019', 'type': 'Proposition', 'text': '\\label{Malpha}\nLet\n\\begin{itemize}\n\\item ${\\mathbf{M}}$ be an anisotropic, semisimple group over~$F$, such that $-1$ is in the Weyl group of\\/~${\\mathbf{M}}$,\n\\item $L$ be a quadratic extension of~$F$, such that\\/ ${\\mathbf{M}}$ is quasisplit over~$L$,\nand\n\\item $\\alpha$ be a simple root of\\/~${\\mathbf{M}}$ that is fixed in the $*$-action shown in the Tits index of\\/~${\\mathbf{M}}$. \n\\end{itemize}\nThen there is a maximal $F$-torus\\/ ${\\mathbf{T}}$ of\\/~${\\mathbf{M}}$, such that the standard subgroup\\/~${\\mathbf{M}}_{\\alpha}$ generated by the roots $\\pm \\alpha$ is defined over~$F$.\n\nFurthermore, if\\/ ${\\mathbf{M}}$ is split over~$L$, then\\/ ${\\mathbf{T}}$ may be chosen to be split over~$L$.'}, {'stmt_id': '0705_4330::s0020', 'type': 'Claim', 'text': 'For every $v \\in {S_{\\aG}}$, there exist \n $b_{v,4},b_{v,5},\\ldots,b_{v,n} \\in L_v$, such that \n $$ \\text{$a_4 \\, b_{v,4}^{\\tau_v} \\, b_{v,4} + a_5 \\, b_{v,5}^{\\tau_v} \\, b_{v,5} + \\cdots + a_n \\, b_{v,n}^{\\tau_v} \\, b_{v,n}$ is a nonzero square in~$F_v$.} $$'}, {'stmt_id': '0705_4330::s0021', 'type': 'Proposition', 'text': 'Let\\/ ${\\mathbf{G}}$ be an absolutely almost simple $F$-group of type $F_4$, such that ${\\fieldrank{F}} {\\mathbf{G}} = 1$.\nThen\\/ ${\\mathbf{G}}$ contains an isotropic, simply connected, absolutely almost simply $F$-subgroup\\/~${\\mathbf{H}}$ of type~$C_3$, such that ${\\fieldrank{F_v}} {\\mathbf{H}} \\ge 2$ for every $v \\in {S_{\\aG}}$.'}, {'stmt_id': '0705_4330::s0022', 'type': 'Corollary', 'text': '\\label{F4NotMin}\nIf\\/ ${\\mathbf{G}}$ is of type~$F_4$, then ${\\mathbf{G}}$ is not minimal.'}, {'stmt_id': '0705_4330::s0023', 'type': 'Theorem', 'text': '\\label{TrialityHasSL2}\nLet\\/ ${\\mathbf{G}}$ be an absolutely almost simple $F$-group of type $\\out3D4$ or~$\\out6D4$, such that ${\\fieldrank{F}} {\\mathbf{G}} = 1$.\nThen there exists an extension field~$K$ of~$F$, such that\\/ $\\res{K/F} \\aSL_2$ is isogenous to an $F$-subgroup of\\/~${\\mathbf{G}}$, and\\/ $|K:F| = 4$.'}, {'stmt_id': '0705_4330::s0024', 'type': 'Corollary', 'text': 'If\\/ ${\\mathbf{G}}$ is of type $\\out3D4$ or $\\out6D4$, then\\/ ${\\mathbf{G}}$ is not minimal.'}, {'stmt_id': '0705_4330::s0025', 'type': 'Theorem', 'text': '\\label{E6(D4kernel)}\nIf\\/ ${\\mathbf{G}}$ is a simply connected, absolutely almost simple $F$-group of type $\\out2E{6,1}^{29}$, then\\/ ${\\mathbf{G}}$ contains an isotropic, simply connected, absolutely almost simple $F$-subgroup\\/~${\\mathbf{H}}$ of type $\\out2A5$, such that ${\\fieldrank{F_v}} {\\mathbf{H}} \\ge 2$, for every archimedean place~$v$ of~$F$.'}, {'stmt_id': '0705_4330::s0026', 'type': 'Lemma', 'text': '{GaribaldiPetersson}}] \\label{anis(E6)=spin}\nIf\\/ ${\\mathbf{G}}$ is an absolutely almost simple $F$-group of type $\\out2E{6,1}^{29}$, then the semisimple anisotropic kernel of\\/~${\\mathbf{G}}$ is isomorphic to\\/ $\\aSpin_8(f)$, for some quadratic form~$f$ on~$F^8$ with nontrivial discriminant.'}, {'stmt_id': '0705_4330::s0027', 'type': 'Theorem', 'text': '\\label{E6(A5kernel)}\nIf\\/ ${\\mathbf{G}}$ is an absolutely almost simple $F$-group of type $\\out2E{6,1}^{35}$, then\\/ ${\\mathbf{G}}$ contains an isotropic, simply connected, absolutely almost simple $F$-subgroup~${\\mathbf{H}}$ of type $\\out3D4$ or $\\out6D4$.'}, {'stmt_id': '0705_4330::s0028', 'type': 'Claim', 'text': '$D$ is a cubic division algebra over~$K$ {\\rm(}and $m = 2${\\rm)}.'}, {'stmt_id': '0705_4330::s0029', 'type': 'Corollary', 'text': '\\label{E6(A5kernel)notmin}\nIf\\/ ${\\mathbf{G}}$ is of type $E_6$, then\\/ ${\\mathbf{G}}$ is not minimal.'}]

    example_tree = [
  {
    "stmt_id": "0705_4330::s0000",
    "type": "Theorem",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0001",
    "type": "Definition",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0002",
    "type": "Definition",
    "depends_on": [
      "0705_4330::s0001"
    ]
  },
  {
    "stmt_id": "0705_4330::s0003",
    "type": "Theorem",
    "depends_on": [
      "0705_4330::s0001",
      "0705_4330::s0002"
    ]
  },
  {
    "stmt_id": "0705_4330::s0004",
    "type": "Theorem",
    "depends_on": [
      "0705_4330::s0001"
    ]
  },
  {
    "stmt_id": "0705_4330::s0005",
    "type": "Theorem",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0006",
    "type": "Definition",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0007",
    "type": "Theorem",
    "depends_on": [
      "0705_4330::s0006"
    ]
  },
  {
    "stmt_id": "0705_4330::s0008",
    "type": "Definition",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0009",
    "type": "Theorem",
    "depends_on": [
      "0705_4330::s0008"
    ]
  },
  {
    "stmt_id": "0705_4330::s0010",
    "type": "Corollary",
    "depends_on": [
      "0705_4330::s0009"
    ]
  },
  {
    "stmt_id": "0705_4330::s0011",
    "type": "Lemma",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0012",
    "type": "Lemma",
    "depends_on": [
      "0705_4330::s0008",
      "0705_4330::s0009"
    ]
  },
  {
    "stmt_id": "0705_4330::s0013",
    "type": "Lemma",
    "depends_on": [
      "0705_4330::s0008",
      "0705_4330::s0009"
    ]
  },
  {
    "stmt_id": "0705_4330::s0014",
    "type": "Lemma",
    "depends_on": [
      "0705_4330::s0009"
    ]
  },
  {
    "stmt_id": "0705_4330::s0015",
    "type": "Corollary",
    "depends_on": [
      "0705_4330::s0008"
    ]
  },
  {
    "stmt_id": "0705_4330::s0016",
    "type": "Lemma",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0017",
    "type": "Definition",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0018",
    "type": "Definition",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0019",
    "type": "Proposition",
    "depends_on": [
      "0705_4330::s0018"
    ]
  },
  {
    "stmt_id": "0705_4330::s0020",
    "type": "Claim",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0021",
    "type": "Proposition",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0022",
    "type": "Corollary",
    "depends_on": [
      "0705_4330::s0008",
      "0705_4330::s0021"
    ]
  },
  {
    "stmt_id": "0705_4330::s0023",
    "type": "Theorem",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0024",
    "type": "Corollary",
    "depends_on": [
      "0705_4330::s0008",
      "0705_4330::s0023"
    ]
  },
  {
    "stmt_id": "0705_4330::s0025",
    "type": "Theorem",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0026",
    "type": "Lemma",
    "depends_on": [
      "0705_4330::s0017"
    ]
  },
  {
    "stmt_id": "0705_4330::s0027",
    "type": "Theorem",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0028",
    "type": "Claim",
    "depends_on": []
  },
  {
    "stmt_id": "0705_4330::s0029",
    "type": "Corollary",
    "depends_on": [
      "0705_4330::s0008",
      "0705_4330::s0025",
      "0705_4330::s0027"
    ]
  }
]
    example_statements2 = {"1705_10703": [{"stmt_id": "1705_10703::a0000", "type": "AI Defs", "text": "As usual, let $H^2$ denote the classical Hardy space. The space $H^2$ can be seen as a space of functions analytic in the unit disk\n    \\mathbb{D}=\\{z:|z|<1\\}$ or as a closed\n    subspace of $L^2:=L^2(\\partial\\mathbb{D})$. In the first case $H^2$ consists of functions analytic in $\\mathbb{D}$ with square summable MacLaurin coefficients and in the second it consists of functions from $L^2$ such that their Fourier coefficients with negative indices vanish."}, {"stmt_id": "1705_10703::a0001", "type": "AI Defs", "text": "The unilateral shift $S$ on $H^2$ is the operator of multiplication by the independent variable, that is,\n      $$Sf(z)=z\\cdot f(z).$$\n    The adjoint of $S^*$ of $S$ is called the backward shift. A simple verification shows that\n      $$S^*f(z)=\\frac{f(z)-f(0)}{z}.$$"}, {"stmt_id": "1705_10703::a0002", "type": "AI Defs", "text": "The famous Beurling theorem provides a characterization of all $S^*$-invariant subspaces of $H^2$. Namely, a closed nontrivial subspace of $H^2$ is $S^*$-invariant if and only if it is of the form\n      $$K_\\alpha=H^2\\ominus \\alpha H^2,$$\n    where $\\alpha$ is an inner function, i.e., $\\alpha$ belongs to the algebra $H^{\\infty}$ of bounded analytic functions and $|\\alpha|=1$ a.e. on $\\partial\n    \\mathbb{D}$. The space $K_{\\alpha}$ is called the model space associated with $\\alpha$."}, {"stmt_id": "1705_10703::a0003", "type": "AI Defs", "text": "Truncated Toeplitz operators are compressions of Toeplitz operators to model spaces. More precisely, a truncated Toeplitz operator $A_{\\varphi}^{\\alpha}$ with a symbol $\\varphi\\in L^2$ is defined on the model space $K_{\\alpha}$ by\n      $$A_{\\varphi}^{\\alpha}f=P_{\\alpha}(\\varphi f),$$\n    where $P_{\\alpha}$ is the orthogonal projection from $L^2$ onto $K_{\\alpha}$. In particular, $S_\\alpha=A^\\alpha_z$ is called the compressed shift. Since $K_\\alpha$ is $S^*$-invariant, it easily follows that $S^*_\\alpha=S^*_{|K_\\alpha}$."}, {"stmt_id": "1705_10703::a0004", "type": "AI Defs", "text": "Let $\\alpha$, $\\beta$ be two inner functions. An asymmetric\n    truncated Toeplitz operator $A_{\\varphi}^{\\alpha,\\beta}$ with a symbol $\\varphi\\in\n    L^2$ is the\n    operator from $K_{\\alpha}$ into $K_{\\beta}$ given by\n      $$A_{\\varphi}^{\\alpha,\\beta}f=P_{\\beta}(\\varphi f).$$\n    Clearly, $A_{\\varphi}^{\\alpha}=A_{\\varphi}^{\\alpha,\\alpha}$.\n    Put\n      $$\\mathscr{T}(\\alpha,\\beta)=\\{A_{\\varphi}^{\\alpha,\\beta}\\colon\\, \\varphi\\in\n      L^2\\ \\mathrm{and}\\ A_{\\varphi}^{\\alpha,\\beta}\\\n      \\mathrm{is\\ bounded}\\}.$$"}, {"stmt_id": "1705_10703::a0005", "type": "AI Defs", "text": "Recall that  a model space $K_{\\alpha}$ is a reproducing kernel Hilbert space. That is to say that for every $f$ in the model space $K_{\\alpha}$ and each $w\\in\\mathbb{D}$,\n      $$f(w)=\\langle f, k_{w}^{\\alpha}\\rangle,$$\n    where the reproducing kernel function $k_{w}^{\\alpha}$ is of the form\n      $$k_{w}^{\\alpha}(z)=\\frac{1-\\overline{\\alpha(w)}\\alpha(z)}{1-\\overline{w}z}.$$\n      Observe that since $k_{w}^{\\alpha}\\in H^{\\infty}$, the set $K_{\\alpha}^{\\infty}=K_{\\alpha}\\cap H^{\\infty}$ is dense in $K_{\\alpha}$."}, {"stmt_id": "1705_10703::a0006", "type": "AI Defs", "text": "A conjugate kernel is the function\n    $\\widetilde{k}_{w}^{\\alpha}=C_{\\alpha}{k}_{w}^{\\alpha}$, where $C_{\\alpha}:L^2\\to L^2$ is given by\n      $$C_{\\alpha}f(z)=\\widetilde{f}(z)=\\alpha(z)\\overline{z}\\overline{f(z)},\\quad |z|=1.$$\n    It can be seen that $C_\\alpha$, as defined on $L^2$, is an antilinear isometric involution (a map with such property is called a conjugation). It can also be verified that the conjugation $C_\\alpha$ preserves $K_\\alpha$. Therefore $\\widetilde{k}_{w}^{\\alpha}\\in K_\\alpha$ for all $w\\in\\mathbb{D}$ and a simple computation gives\n      $$\\widetilde{k}_{w}^{\\alpha}(z)=\\frac{\\alpha(z)-\\alpha(w)}{z-w}.$$"}, {"stmt_id": "1705_10703::a0000", "type": "AI Defs", "text": "As usual, let $H^2$ denote the classical Hardy space. The space $H^2$ can be seen as a space of functions analytic in the unit disk\n    \\mathbb{D}=\\{z:|z|<1\\}$ or as a closed\n    subspace of $L^2:=L^2(\\partial\\mathbb{D})$. In the first case $H^2$ consists of functions analytic in $\\mathbb{D}$ with square summable MacLaurin coefficients and in the second it consists of functions from $L^2$ such that their Fourier coefficients with negative indices vanish."}, {"stmt_id": "1705_10703::a0001", "type": "AI Defs", "text": "The unilateral shift $S$ on $H^2$ is the operator of multiplication by the independent variable, that is,\n      $$Sf(z)=z\\cdot f(z).$$\n    The adjoint of $S^*$ of $S$ is called the backward shift. A simple verification shows that\n      $$S^*f(z)=\\frac{f(z)-f(0)}{z}.$$"}, {"stmt_id": "1705_10703::a0002", "type": "AI Defs", "text": "The famous Beurling theorem provides a characterization of all $S^*$-invariant subspaces of $H^2$. Namely, a closed nontrivial subspace of $H^2$ is $S^*$-invariant if and only if it is of the form\n      $$K_\\alpha=H^2\\ominus \\alpha H^2,$$\n    where $\\alpha$ is an inner function, i.e., $\\alpha$ belongs to the algebra $H^{\\infty}$ of bounded analytic functions and $|\\alpha|=1$ a.e. on $\\partial\n    \\mathbb{D}$. The space $K_{\\alpha}$ is called the model space associated with $\\alpha$."}, {"stmt_id": "1705_10703::a0003", "type": "AI Defs", "text": "Truncated Toeplitz operators are compressions of Toeplitz operators to model spaces. More precisely, a truncated Toeplitz operator $A_{\\varphi}^{\\alpha}$ with a symbol $\\varphi\\in L^2$ is defined on the model space $K_{\\alpha}$ by\n      $$A_{\\varphi}^{\\alpha}f=P_{\\alpha}(\\varphi f),$$\n    where $P_{\\alpha}$ is the orthogonal projection from $L^2$ onto $K_{\\alpha}$. In particular, $S_\\alpha=A^\\alpha_z$ is called the compressed shift. Since $K_\\alpha$ is $S^*$-invariant, it easily follows that $S^*_\\alpha=S^*_{|K_\\alpha}$."}, {"stmt_id": "1705_10703::a0004", "type": "AI Defs", "text": "Let $\\alpha$, $\\beta$ be two inner functions. An asymmetric\n    truncated Toeplitz operator $A_{\\varphi}^{\\alpha,\\beta}$ with a symbol $\\varphi\\in\n    L^2$ is the\n    operator from $K_{\\alpha}$ into $K_{\\beta}$ given by\n      $$A_{\\varphi}^{\\alpha,\\beta}f=P_{\\beta}(\\varphi f).$$\n    Clearly, $A_{\\varphi}^{\\alpha}=A_{\\varphi}^{\\alpha,\\alpha}$.\n    Put\n      $$\\mathscr{T}(\\alpha,\\beta)=\\{A_{\\varphi}^{\\alpha,\\beta}\\colon\\, \\varphi\\in\n      L^2\\ \\mathrm{and}\\ A_{\\varphi}^{\\alpha,\\beta}\\\n      \\mathrm{is\\ bounded}\\}.$$"}, {"stmt_id": "1705_10703::a0005", "type": "AI Defs", "text": "Recall that  a model space $K_{\\alpha}$ is a reproducing kernel Hilbert space. That is to say that for every $f$ in the model space $K_{\\alpha}$ and each $w\\in\\mathbb{D}$,\n      $$f(w)=\\langle f, k_{w}^{\\alpha}\\rangle,$$\n    where the reproducing kernel function $k_{w}^{\\alpha}$ is of the form\n      $$k_{w}^{\\alpha}(z)=\\frac{1-\\overline{\\alpha(w)}\\alpha(z)}{1-\\overline{w}z}.$$\n      Observe that since $k_{w}^{\\alpha}\\in H^{\\infty}$, the set $K_{\\alpha}^{\\infty}=K_{\\alpha}\\cap H^{\\infty}$ is dense in $K_{\\alpha}$."}, {"stmt_id": "1705_10703::a0000", "type": "AI Defs", "text": "As usual, let $H^2$ denote the classical Hardy space. The space $H^2$ can be seen as a space of functions analytic in the unit disk\n    \\mathbb{D}=\\{z:|z|<1\\}$ or as a closed\n    subspace of $L^2:=L^2(\\partial\\mathbb{D})$. In the first case $H^2$ consists of functions analytic in $\\mathbb{D}$ with square summable MacLaurin coefficients and in the second it consists of functions from $L^2$ such that their Fourier coefficients with negative indices vanish."}, {"stmt_id": "1705_10703::a0001", "type": "AI Defs", "text": "The unilateral shift $S$ on $H^2$ is the operator of multiplication by the independent variable, that is,\n      $$Sf(z)=z\\cdot f(z).$$\n    The adjoint of $S^*$ of $S$ is called the backward shift. A simple verification shows that\n      $$S^*f(z)=\\frac{f(z)-f(0)}{z}.$$"}, {"stmt_id": "1705_10703::a0002", "type": "AI Defs", "text": "The famous Beurling theorem provides a characterization of all $S^*$-invariant subspaces of $H^2$. Namely, a closed nontrivial subspace of $H^2$ is $S^*$-invariant if and only if it is of the form\n      $$K_\\alpha=H^2\\ominus \\alpha H^2,$$\n    where $\\alpha$ is an inner function, i.e., $\\alpha$ belongs to the algebra $H^{\\infty}$ of bounded analytic functions and $|\\alpha|=1$ a.e. on $\\partial\n    \\mathbb{D}$. The space $K_{\\alpha}$ is called the model space associated with $\\alpha$."}, {"stmt_id": "1705_10703::a0003", "type": "AI Defs", "text": "Truncated Toeplitz operators are compressions of Toeplitz operators to model spaces. More precisely, a truncated Toeplitz operator $A_{\\varphi}^{\\alpha}$ with a symbol $\\varphi\\in L^2$ is defined on the model space $K_{\\alpha}$ by\n      $$A_{\\varphi}^{\\alpha}f=P_{\\alpha}(\\varphi f),$$\n    where $P_{\\alpha}$ is the orthogonal projection from $L^2$ onto $K_{\\alpha}$. In particular, $S_\\alpha=A^\\alpha_z$ is called the compressed shift. Since $K_\\alpha$ is $S^*$-invariant, it easily follows that $S^*_\\alpha=S^*_{|K_\\alpha}$."}, {"stmt_id": "1705_10703::a0004", "type": "AI Defs", "text": "Let $\\alpha$, $\\beta$ be two inner functions. An asymmetric\n    truncated Toeplitz operator $A_{\\varphi}^{\\alpha,\\beta}$ with a symbol $\\varphi\\in\n    L^2$ is the\n    operator from $K_{\\alpha}$ into $K_{\\beta}$ given by\n      $$A_{\\varphi}^{\\alpha,\\beta}f=P_{\\beta}(\\varphi f).$$\n    Clearly, $A_{\\varphi}^{\\alpha}=A_{\\varphi}^{\\alpha,\\alpha}$.\n    Put\n      $$\\mathscr{T}(\\alpha,\\beta)=\\{A_{\\varphi}^{\\alpha,\\beta}\\colon\\, \\varphi\\in\n      L^2\\ \\mathrm{and}\\ A_{\\varphi}^{\\alpha,\\beta}\\\n      \\mathrm{is\\ bounded}\\}.$$"}, {"stmt_id": "1705_10703::a0000", "type": "AI Defs", "text": "As usual, let $H^2$ denote the classical Hardy space. The space $H^2$ can be seen as a space of functions analytic in the unit disk\n    \\mathbb{D}=\\{z:|z|<1\\}$ or as a closed\n    subspace of $L^2:=L^2(\\partial\\mathbb{D})$. In the first case $H^2$ consists of functions analytic in $\\mathbb{D}$ with square summable MacLaurin coefficients and in the second it consists of functions from $L^2$ such that their Fourier coefficients with negative indices vanish."}, {"stmt_id": "1705_10703::a0001", "type": "AI Defs", "text": "The unilateral shift $S$ on $H^2$ is the operator of multiplication by the independent variable, that is,\n      $$Sf(z)=z\\cdot f(z).$$\n    The adjoint of $S^*$ of $S$ is called the backward shift. A simple verification shows that\n      $$S^*f(z)=\\frac{f(z)-f(0)}{z}.$$"}, {"stmt_id": "1705_10703::a0002", "type": "AI Defs", "text": "The famous Beurling theorem provides a characterization of all $S^*$-invariant subspaces of $H^2$. Namely, a closed nontrivial subspace of $H^2$ is $S^*$-invariant if and only if it is of the form\n      $$K_\\alpha=H^2\\ominus \\alpha H^2,$$\n    where $\\alpha$ is an inner function, i.e., $\\alpha$ belongs to the algebra $H^{\\infty}$ of bounded analytic functions and $|\\alpha|=1$ a.e. on $\\partial\n    \\mathbb{D}$. The space $K_{\\alpha}$ is called the model space associated with $\\alpha$."}, {"stmt_id": "1705_10703::a0003", "type": "AI Defs", "text": "Truncated Toeplitz operators are compressions of Toeplitz operators to model spaces. More precisely, a truncated Toeplitz operator $A_{\\varphi}^{\\alpha}$ with a symbol $\\varphi\\in L^2$ is defined on the model space $K_{\\alpha}$ by\n      $$A_{\\varphi}^{\\alpha}f=P_{\\alpha}(\\varphi f),$$\n    where $P_{\\alpha}$ is the orthogonal projection from $L^2$ onto $K_{\\alpha}$. In particular, $S_\\alpha=A^\\alpha_z$ is called the compressed shift. Since $K_\\alpha$ is $S^*$-invariant, it easily follows that $S^*_\\alpha=S^*_{|K_\\alpha}$."}, {"stmt_id": "1705_10703::a0000", "type": "AI Defs", "text": "As usual, let $H^2$ denote the classical Hardy space. The space $H^2$ can be seen as a space of functions analytic in the unit disk\n    \\mathbb{D}=\\{z:|z|<1\\}$ or as a closed\n    subspace of $L^2:=L^2(\\partial\\mathbb{D})$. In the first case $H^2$ consists of functions analytic in $\\mathbb{D}$ with square summable MacLaurin coefficients and in the second it consists of functions from $L^2$ such that their Fourier coefficients with negative indices vanish."}, {"stmt_id": "1705_10703::a0001", "type": "AI Defs", "text": "The unilateral shift $S$ on $H^2$ is the operator of multiplication by the independent variable, that is,\n      $$Sf(z)=z\\cdot f(z).$$\n    The adjoint of $S^*$ of $S$ is called the backward shift. A simple verification shows that\n      $$S^*f(z)=\\frac{f(z)-f(0)}{z}.$$"}, {"stmt_id": "1705_10703::a0002", "type": "AI Defs", "text": "The famous Beurling theorem provides a characterization of all $S^*$-invariant subspaces of $H^2$. Namely, a closed nontrivial subspace of $H^2$ is $S^*$-invariant if and only if it is of the form\n      $$K_\\alpha=H^2\\ominus \\alpha H^2,$$\n    where $\\alpha$ is an inner function, i.e., $\\alpha$ belongs to the algebra $H^{\\infty}$ of bounded analytic functions and $|\\alpha|=1$ a.e. on $\\partial\n    \\mathbb{D}$. The space $K_{\\alpha}$ is called the model space associated with $\\alpha$."}, {"stmt_id": "1705_10703::a0000", "type": "AI Defs", "text": "As usual, let $H^2$ denote the classical Hardy space. The space $H^2$ can be seen as a space of functions analytic in the unit disk\n    \\mathbb{D}=\\{z:|z|<1\\}$ or as a closed\n    subspace of $L^2:=L^2(\\partial\\mathbb{D})$. In the first case $H^2$ consists of functions analytic in $\\mathbb{D}$ with square summable MacLaurin coefficients and in the second it consists of functions from $L^2$ such that their Fourier coefficients with negative indices vanish."}, {"stmt_id": "1705_10703::a0001", "type": "AI Defs", "text": "The unilateral shift $S$ on $H^2$ is the operator of multiplication by the independent variable, that is,\n      $$Sf(z)=z\\cdot f(z).$$\n    The adjoint of $S^*$ of $S$ is called the backward shift. A simple verification shows that\n      $$S^*f(z)=\\frac{f(z)-f(0)}{z}.$$"}, {"stmt_id": "1705_10703::a0000", "type": "AI Defs", "text": "As usual, let $H^2$ denote the classical Hardy space. The space $H^2$ can be seen as a space of functions analytic in the unit disk\n    \\mathbb{D}=\\{z:|z|<1\\}$ or as a closed\n    subspace of $L^2:=L^2(\\partial\\mathbb{D})$. In the first case $H^2$ consists of functions analytic in $\\mathbb{D}$ with square summable MacLaurin coefficients and in the second it consists of functions from $L^2$ such that their Fourier coefficients with negative indices vanish."}, {"stmt_id": "1705_10703::s0000", "type": "Theorem", "text": "\\label{thm_ATTO_rank_two_char}\nLet $A$ be a bounded linear operator from $K_\\alpha$ into $K_\\beta$. Then $A\\in\\mathscr{T(\\alpha,\\beta)}$ if and only if there exist $\\psi\\in K_\\beta$ and $\\chi\\in K_\\alpha$ such that\n\\begin{align}\\label{eq_ATTO_rank_two_char}\n  A-S_\\beta A S^*_\\alpha=\\psi\\otimes k^\\alpha_0 +k^\\beta_0\\otimes \\chi.\n\\end{align}"}, {"stmt_id": "1705_10703::s0001", "type": "Lemma", "text": "\\label{lem_ATTO_rank2_char_aux1}\nFor every $\\varphi\\in L^2$ the equality\n\\begin{align*}\n  A^{\\alpha,\\beta}_\\varphi-S_\\beta A^{\\alpha,\\beta}_\\varphi S^*_\\alpha=\\psi\\otimes k^\\alpha_0 +k^\\beta_0\\otimes \\chi\n\\end{align*}\nholds on $H^2$, where\n\\begin{align*}\n  &\\psi=S_\\beta P_\\beta\\left(\\overline{z}\\varphi\\right) \\in K_\\beta,\\\\\n  &\\chi=P_\\alpha\\left(\\overline{\\varphi}\\right) \\in K_\\alpha.\n\\end{align*}"}, {"stmt_id": "1705_10703::s0002", "type": "Lemma", "text": "\\label{lem_ATTO_rank2_aux}\nIf $\\varphi=\\overline{\\chi}+\\psi$, where $\\chi\\in K_\\alpha$ and $\\psi\\in K_\\beta$, $\\psi(0)=0$, then the equality\n\\begin{align*}\n  \\langle A_\\varphi^{\\alpha,\\beta}f,g\\rangle=\\sum_{n=1}^\\infty\\langle(S^n_\\beta \\psi\\otimes S^n_\\alpha k^\\alpha_0 +S^n_\\beta k^\\beta_0\\otimes S^n_\\alpha \\chi)f,g\\rangle\n\\end{align*}\nholds for all $f\\in K_\\alpha^\\infty$ and $g\\in K_\\beta^\\infty$."}, {"stmt_id": "1705_10703::s0003", "type": "Corollary", "text": "\\label{cor_ATTO_rank2}\nLet $A$ be a bounded linear operator form $K_\\alpha$ into $K_\\beta$. Then $A\\in\\mathscr{T}(\\alpha,\\beta)$ if and only if\n\\begin{equation}\\label{eq_ATTO_rank_two_char2}\nA-S^*_\\beta A S_\\alpha=\\psi\\otimes \\widetilde{k}^\\alpha_0 +\\widetilde{k}^\\beta_0\\otimes \\chi\n\\end{equation}\nfor some $\\psi\\in K_\\beta$ and $\\chi\\in K_\\alpha$."}, {"stmt_id": "1705_10703::s0004", "type": "Corollary", "text": "\\label{cor_ATTO_rank2_MCS}\nLet $A$ be a bounded linear operator form $K_\\alpha$ into $K_\\beta$. Then $A\\in\\mathscr{T}(\\alpha,\\beta)$ if and only if one (and all) of the following conditions holds\n\\begin{enumerate}\n  \\item[(a)] $A-S_{\\beta,b} A S^*_{\\alpha,a}=\\psi\\otimes k^\\alpha_0 +k^\\beta_0\\otimes \\chi$;\n  \\item[(b)] $A-S^*_{\\beta,b} A S_{\\alpha,a}=\\psi\\otimes \\widetilde{k}^\\alpha_0 +\\widetilde{k}^\\beta_0\\otimes \\chi$;\n\\end{enumerate}\nfor some $a,b\\in\\mathbb{C}$, $\\psi\\in K_\\beta$ and $\\chi\\in K_\\alpha$ (possibly different for different conditions), where\n  $$S_{\\alpha,a}=S_\\alpha+a (k^\\alpha_0\\otimes \\widetilde{k}^\\alpha_0),\\qquad\n  S_{\\beta,b}=S_\\beta+ b(k^\\beta_0\\otimes \\widetilde{k}^\\beta_0)$$\nare the modified compressed shifts."}, {"stmt_id": "1705_10703::s0005", "type": "Corollary", "text": "$\\mathscr{T}(\\alpha,\\beta)$ is closed in the weak operator topology."}, {"stmt_id": "1705_10703::s0006", "type": "Corollary", "text": "\\label{cor_ATTO_rank2_SI}\nLet $A$ be a bounded linear operator form $K_\\alpha$ into $K_\\beta$. Then $A\\in\\mathscr{T}(\\alpha,\\beta)$ if and only if it has the following property\n$$\\left\\langle AS f, S g\\right\\rangle=\\left\\langle A f,g \\right\\rangle$$\n for all $f\\in K_\\alpha$, $g\\in K_\\beta$ such that $Sf\\in K_\\alpha$, $Sg\\in K_\\beta$."}]}
    example_tree2 = {
  "1705_10703": [
    {
      "stmt_id": "1705_10703::a0000",
      "type": "AI Defs",
      "depends_on": []
    },
    {
      "stmt_id": "1705_10703::a0001",
      "type": "AI Defs",
      "depends_on": [
        "1705_10703::a0000"
      ]
    },
    {
      "stmt_id": "1705_10703::a0002",
      "type": "AI Defs",
      "depends_on": [
        "1705_10703::a0000",
        "1705_10703::a0001"
      ]
    },
    {
      "stmt_id": "1705_10703::a0003",
      "type": "AI Defs",
      "depends_on": [
        "1705_10703::a0000",
        "1705_10703::a0001",
        "1705_10703::a0002"
      ]
    },
    {
      "stmt_id": "1705_10703::a0004",
      "type": "AI Defs",
      "depends_on": [
        "1705_10703::a0000",
        "1705_10703::a0002",
        "1705_10703::a0003"
      ]
    },
    {
      "stmt_id": "1705_10703::a0005",
      "type": "AI Defs",
      "depends_on": [
        "1705_10703::a0000",
        "1705_10703::a0002"
      ]
    },
    {
      "stmt_id": "1705_10703::a0006",
      "type": "AI Defs",
      "depends_on": [
        "1705_10703::a0000",
        "1705_10703::a0005"
      ]
    },
    {
      "stmt_id": "1705_10703::s0000",
      "type": "Theorem",
      "depends_on": [
        "1705_10703::a0000",
        "1705_10703::a0001",
        "1705_10703::a0002",
        "1705_10703::a0003",
        "1705_10703::a0004",
        "1705_10703::a0005"
      ]
    },
    {
      "stmt_id": "1705_10703::s0001",
      "type": "Lemma",
      "depends_on": [
        "1705_10703::a0000",
        "1705_10703::a0001",
        "1705_10703::a0002",
        "1705_10703::a0003",
        "1705_10703::a0004",
        "1705_10703::a0005"
      ]
    },
    {
      "stmt_id": "1705_10703::s0002",
      "type": "Lemma",
      "depends_on": [
        "1705_10703::a0000",
        "1705_10703::a0001",
        "1705_10703::a0002",
        "1705_10703::a0003",
        "1705_10703::a0004",
        "1705_10703::a0005"
      ]
    },
    {
      "stmt_id": "1705_10703::s0003",
      "type": "Corollary",
      "depends_on": [
        "1705_10703::a0000",
        "1705_10703::a0001",
        "1705_10703::a0002",
        "1705_10703::a0003",
        "1705_10703::a0004",
        "1705_10703::a0005",
        "1705_10703::a0006",
        "1705_10703::s0000"
      ]
    },
    {
      "stmt_id": "1705_10703::s0004",
      "type": "Corollary",
      "depends_on": [
        "1705_10703::a0000",
        "1705_10703::a0001",
        "1705_10703::a0002",
        "1705_10703::a0003",
        "1705_10703::a0004",
        "1705_10703::a0005",
        "1705_10703::a0006",
        "1705_10703::s0000",
        "1705_10703::s0003"
      ]
    },
    {
      "stmt_id": "1705_10703::s0005",
      "type": "Corollary",
      "depends_on": [
        "1705_10703::a0000",
        "1705_10703::a0001",
        "1705_10703::a0002",
        "1705_10703::a0003",
        "1705_10703::a0004",
        "1705_10703::a0005",
        "1705_10703::a0006",
        "1705_10703::s0000",
        "1705_10703::s0003",
        "1705_10703::s0004"
      ]
    },
    {
      "stmt_id": "1705_10703::s0006",
      "type": "Corollary",
      "depends_on": [
        "1705_10703::a0000",
        "1705_10703::a0001",
        "1705_10703::a0002",
        "1705_10703::a0003",
        "1705_10703::a0004",
        "1705_10703::a0005",
        "1705_10703::s0000"
      ]
    }
  ]
}

    new_messages = [
        {
            "role": "system",
            "content": """You are an expert mathematician and data extractor. Your sole task is to analyze a list of mathematical statements and build a dependency tree.

    A dependency tree is a directed graph where statements are nodes and their dependencies are edges. 
    Criteria for a dependency:
    1. Explicit references: A statement explicitly calls back to another using \\ref{...} or in natural language.
    2. Implicit reliance: A theorem or lemma heavily relies on a concept, definition, object, etc... that is newly established in a previous statement.

    Output the dependency tree as a strictly formatted JSON array of objects. Each object must have:
    - "stmt_id": The ID of the statement.
    - "type": The type of the statement.
    - "depends_on": A list of "stmt_id" strings representing the statements this node depends on.

    OUTPUT NOTHING BUT VALID JSON. Do not add any conversational text or commentary."""
        },
        {
            "role": "user",
            "content": f"""---
    **Input Statements**:
    {json.dumps(example_statements2, indent=2)}
    ---
    """
        },
        {
            "role": "assistant",
            "content": json.dumps(example_tree2, indent=2)
        },
        {
            "role": "user",
            "content": f"""Excellent, that's the exact correct format. Now, apply the exact same process to the following new statements.

    ---
    **Input Statements**:
    {json.dumps(latex_context, indent=2)}
    ---
    """
        }
    ]
    old_messages = [
    {
        "role": "user",
        "content": f"""You are an expert mathematician and data extractor. Your sole task is to analyze a list of mathematical statements and build a dependency tree.

A dependency tree is a directed graph where statements are nodes and their dependencies are edges. 
Criteria for a dependency:
1. Explicit references: A statement explicitly calls back to another using \\ref{{...}} or in natural language.
2. Implicit reliance: A theorem or lemma heavily relies on a concept, definition, object, etc... that is newly established in a previous statement.

Output the dependency tree as a strictly formatted JSON array of objects. Each object must have:
- "stmt_id": The ID of the statement.
- "type": The type of the statement.
- "depends_on": A list of "stmt_id" strings representing the statements this node depends on.

OUTPUT NOTHING BUT VALID JSON. Do not add any conversational text or commentary.

---
**Input Statements**:**
{json.dumps(example_statements2, indent=2)}
---
"""
    },
    {
        "role": "assistant",
        "content": json.dumps(example_tree2, indent=2)
    },
    {
        "role": "user",
        "content": f"""Excellent, that's the exact correct format. Now, apply the exact same process to the following new statements.

---
**Input Text:**
{json.dumps(latex_context, indent=2)}
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
        ForceEndThinkLogitsProcessor(max_think_tokens=9000, end_think_token_id=end_think_token_id)
    ])

    # 3. Generate (Notice the ** unpacking operator, this passes both input_ids and attention_mask!)
    outputs = model.generate(
        **inputs,
        max_new_tokens=15192,
        logits_processor=logits_processor,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # 4. Decode ONLY the newly generated tokens
    result = tokenizer.decode(outputs[0][num_input_tokens:], skip_special_tokens=True)
    cleaned = clean_and_verify_json(result)
    return cleaned




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
        paper_id, stmt_list = item
        try:
            tree = extract_definitions(stmt_list, model, tokenizer)
            if tree:
                result_queue.put(("OK", {paper_id: tree}))
            else:
                result_queue.put(("EMPTY", {paper_id: []}))   # record, don't retry
        except Exception as e:
            result_queue.put(("ERR", paper_id, str(e)))
    result_queue.put(("DONE", gpu_id))


def main():
    processed_paper_ids = set()
    if os.path.exists(TREE_FILE):
        with open(TREE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    processed_paper_ids.update(json.loads(line).keys())
    print(f"Found {len(processed_paper_ids)} already processed papers. Skipping those...")

    statements = {}
    with open(STATEMENTS_IN, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading Statements"):
            if not line.strip():
                continue
            for papid, stmt_list in json.loads(line).items():
                statements[papid] = stmt_list
    print(f"Loaded {len(statements)} papers ready for tree extraction.")

    tasks = [(pid, s) for pid, s in statements.items() if pid not in processed_paper_ids]
    print(f"{len(tasks)} papers to process across {NUM_GPUS} GPUs.")

    task_queue, result_queue = mp.Queue(), mp.Queue()
    for t in tasks:
        task_queue.put(t)
    for _ in range(NUM_GPUS):
        task_queue.put(None)

    procs = [mp.Process(target=worker, args=(g, task_queue, result_queue))
             for g in range(NUM_GPUS)]
    for p in procs:
        p.start()

    alive = NUM_GPUS
    counts = Counter()
    with open(TREE_FILE, "a", encoding="utf-8") as pf, \
         tqdm(total=len(tasks), desc="extracting") as pbar:
        while alive > 0:
            try:
                msg = result_queue.get(timeout=300)
            except pyqueue.Empty:
                if not any(p.is_alive() for p in procs):
                    break
                continue
            tag = msg[0]
            if tag in ("OK", "EMPTY"):
                pf.write(json.dumps(msg[1], ensure_ascii=False) + "\n"); pf.flush()
                counts[tag.lower()] += 1; pbar.update(1)
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
    print("Tree extraction complete:", dict(counts))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
