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


MY_HF_TOKEN = "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"
MODEL_NAME = "Qwen/Qwen3.5-9B"#"Qwen/Qwen2.5-14B-Instruct"   # or "mistralai/Mistral-Nemo-Instruct-2407"


EXTRA_STATEMENTS_FILE = "extra_statements_start-1900_v3.jsonl"
ROOT_FOLDER = "/home/aiming/math_src_EXTRACTED_start-1900"
STATEMENTS_IN = "statements_start-1900_v3.jsonl"
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
        device_map={"": gpu_id},        # whole model onto THIS gpu, not "auto"
        token=MY_HF_TOKEN,
        quantization_config=quantization_config,
        attn_implementation="sdpa",
    )
    return model, tokenizer


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

def get_paper(papid):
    filepath = ROOT_FOLDER + '/' + papid
    if os.path.exists(filepath):
        raw = read_paper_text(filepath)
        cleaned = raw.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
        return cleaned



def extract_definitions(latex_paper,statements, model, tokenizer):
    """
    Uses the LLM to extract mathematical definitions from a LaTeX string.
    """
    paper_1705_10703 = r"""\documentclass[12pt]{amsart}
\usepackage{graphicx}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{mathrsfs}
\usepackage[cp1250]{inputenc}
%\usepackage{showkeys}
\pagestyle{plain}
\textheight24cm \textwidth16cm \oddsidemargin0.25pc
\evensidemargin0.25pc \topmargin-1cm \footskip10mm
\def\sk{\vskip12pt}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{example}[theorem]{Example}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{lemma}[theorem]{Lemma}
\theoremstyle{definition}

\numberwithin{equation}{section}

%\renewcommand{\theequation}{\arabic{section}.\arabic{equation}}
\renewcommand{\Re}{\mathop{\rm Re}}
\renewcommand{\Im}{\mathop{\rm Im}}


\begin{document}
\title{Some characterizations of asymmetric truncated Toeplitz operators}
\author{Bartosz {\L}anucha, Ma{\l}gorzata Michalska}

\address{
Bartosz {\L}anucha,  \newline Institute of Mathematics,
\newline Maria Curie-Sk{\l}odowska University, \newline pl. M.
Curie-Sk{\l}odowskiej 1, \newline 20-031 Lublin, Poland}
\email{bartosz.lanucha@poczta.umcs.lublin.pl}

\address{
Ma{\l}gorzata Michalska,  \newline Institute of Mathematics,
\newline Maria Curie-Sk{\l}odowska University, \newline pl. M.
Curie-Sk{\l}odowskiej 1, \newline 20-031 Lublin, Poland}
\email{malgorzata.michalska@poczta.umcs.lublin.pl}


\date{\today}
\subjclass[2010]{47B32, 47B35, 30H10.}
\keywords{model space, truncated
Toeplitz operator, asymmetric truncated
Toeplitz operator}
\begin{abstract}
It was recently proved that in some special cases asymmetric truncated Toeplitz operators can be characterized in terms of compressed shifts and rank-two operators of special form. In this paper we show that such characterizations hold in all cases.
\end{abstract}
\maketitle

\baselineskip1.4\baselineskip

\section{Introduction}
As usual, let $H^2$ denote the classical Hardy space. The space $H^2$ can be seen as a space of functions analytic in the unit disk
$\mathbb{D}=\{z:|z|<1\}$ or as a closed
subspace of $L^2:=L^2(\partial\mathbb{D})$. In the first case $H^2$ consists of functions analytic in $\mathbb{D}$ with square summable MacLaurin coefficients and in the second it consists of functions from $L^2$ such that their Fourier coefficients with negative indices vanish.

The unilateral shift $S$ on $H^2$ is the operator of multiplication by the independent variable, that is,
  $$Sf(z)=z\cdot f(z).$$
The adjoint of $S^*$ of $S$ is called the backward shift. A simple verification shows that
  $$S^*f(z)=\frac{f(z)-f(0)}{z}.$$

The famous Beurling theorem provides a characterization of all $S^*$-invariant subspaces of $H^2$. Namely, a closed nontrivial subspace of $H^2$ is $S^*$-invariant if and only if it is of the form
  $$K_\alpha=H^2\ominus \alpha H^2,$$
where $\alpha$ is an inner function, i.e., $\alpha$ belongs to the algebra $H^{\infty}$ of bounded analytic functions and $|\alpha|=1$ a.e. on $\partial
\mathbb{D}$. The space $K_{\alpha}$ is called the model space associated with $\alpha$.

The operators $S$ and $S^*$ are two examples of classical Toeplitz operators. Let $P$ denote the orthogonal projection from $L^2$ onto $H^2$. A Toeplitz operator $T_{\varphi}$ with symbol $\varphi\in L^\infty$ is defined on $H^2$ by
  $$T_{\varphi}f=P(\varphi f).$$
Clearly, if $\varphi\in H^\infty$, then $T_\varphi$ is just the multiplication by $\varphi$. Also, the operator $T_{\varphi}$ is densely defined on bounded functions whenever $\varphi\in L^2$, and extends to a bounded operator on $H^2$ if and only if $\varphi \in L^\infty$. We have $S=T_z$ and $S^{*}=T_{\overline{z}}$.

Truncated Toeplitz operators are compressions of Toeplitz operators to model spaces. More precisely, a truncated Toeplitz operator $A_{\varphi}^{\alpha}$ with a symbol $\varphi\in L^2$ is defined on the model space $K_{\alpha}$ by
  $$A_{\varphi}^{\alpha}f=P_{\alpha}(\varphi f),$$
where $P_{\alpha}$ is the orthogonal projection from $L^2$ onto $K_{\alpha}$. In particular, $S_\alpha=A^\alpha_z$ is called the compressed shift. Since $K_\alpha$ is $S^*$-invariant, it easily follows that $S^*_\alpha=S^*_{|K_\alpha}$.

The study of the class of truncated Toeplitz operators was started in 2007 with D. Sarason's paper \cite{s} (see \cite{gar3}). Recently, the authors in \cite{ptak} and \cite{part2,part} initiated the study of so-called asymmetric truncated Toeplitz operators (see also \cite{blicharz1,blicharz2}).

Let $\alpha$, $\beta$ be two inner functions. An asymmetric
truncated Toeplitz operator $A_{\varphi}^{\alpha,\beta}$ with a symbol $\varphi\in
L^2$ is the
operator from $K_{\alpha}$ into $K_{\beta}$ given by
  $$A_{\varphi}^{\alpha,\beta}f=P_{\beta}(\varphi f).$$
Clearly, $A_{\varphi}^{\alpha}=A_{\varphi}^{\alpha,\alpha}$.
Put
  $$\mathscr{T}(\alpha,\beta)=\{A_{\varphi}^{\alpha,\beta}\colon\, \varphi\in
  L^2\ \mathrm{and}\ A_{\varphi}^{\alpha,\beta}\
  \mathrm{is\ bounded}\}.$$


A bounded linear operator $T$ on $H^2$ is a Toeplitz operator if and only if $T-S^* TS=0$. D. Sarason \cite{s} proved that a bounded linear operator $A$ on $K_\alpha$ is a truncated Toeplitz operator if and only if $A-S_{\alpha}^* AS_\alpha$ is an operator of a special kind and rank at most two. Similar characterization for the operators from $\mathscr{T}(\alpha,\beta)$ were proved for the case when $\beta$ divides $\alpha$ (that is, $\alpha/\beta$ is an inner function) \cite{ptak}, and for the case when $\alpha$ and $\beta$ are finite Blaschke products (in other words, when $K_\alpha$ and $K_\beta$ are finitely dimensional) \cite{L2}.

In this paper we provide such characterizations of the operators from $\mathscr{T}(\alpha,\beta)$ for all $\alpha, \beta$.



\section{Characterizations of asymmetric truncated Toeplitz operators}
Recall that  a model space $K_{\alpha}$ is a reproducing kernel Hilbert space. That is to say that for every $f$ in the model space $K_{\alpha}$ and each $w\in\mathbb{D}$,
  $$f(w)=\langle f, k_{w}^{\alpha}\rangle,$$
where the reproducing kernel function $k_{w}^{\alpha}$ is of the form
  $$k_{w}^{\alpha}(z)=\frac{1-\overline{\alpha(w)}\alpha(z)}{1-\overline{w}z}.$$
  Observe that since $k_{w}^{\alpha}\in H^{\infty}$, the set $K_{\alpha}^{\infty}=K_{\alpha}\cap H^{\infty}$ is dense in $K_{\alpha}$.

A conjugate kernel is the function
$\widetilde{k}_{w}^{\alpha}=C_{\alpha}{k}_{w}^{\alpha}$, where $C_{\alpha}:L^2\to L^2$ is given by
  $$C_{\alpha}f(z)=\widetilde{f}(z)=\alpha(z)\overline{z}\overline{f(z)},\quad |z|=1.$$
It can be seen that $C_\alpha$, as defined on $L^2$, is an antilinear isometric involution (a map with such property is called a conjugation). It can also be verified that the conjugation $C_\alpha$ preserves $K_\alpha$. Therefore $\widetilde{k}_{w}^{\alpha}\in K_\alpha$ for all $w\in\mathbb{D}$ and a simple computation gives
  $$\widetilde{k}_{w}^{\alpha}(z)=\frac{\alpha(z)-\alpha(w)}{z-w}.$$


\begin{theorem}\label{thm_ATTO_rank_two_char}
Let $A$ be a bounded linear operator from $K_\alpha$ into $K_\beta$. Then $A\in\mathscr{T(\alpha,\beta)}$ if and only if there exist $\psi\in K_\beta$ and $\chi\in K_\alpha$ such that
\begin{align}\label{eq_ATTO_rank_two_char}
  A-S_\beta A S^*_\alpha=\psi\otimes k^\alpha_0 +k^\beta_0\otimes \chi.
\end{align}
\end{theorem}
The proof of Theorem \ref{thm_ATTO_rank_two_char} follows the proof given by D. Sarason for truncated Toeplitz operators \cite[Theorem 4.1]{s} and requires some auxiliary lemmas.
\begin{lemma}\label{lem_ATTO_rank2_char_aux1}
For every $\varphi\in L^2$ the equality
\begin{align*}%\label{eq_ATTO_rank2_char_aux1}
  A^{\alpha,\beta}_\varphi-S_\beta A^{\alpha,\beta}_\varphi S^*_\alpha=\psi\otimes k^\alpha_0 +k^\beta_0\otimes \chi
\end{align*}
holds on $H^2$, where
\begin{align*}
  &\psi=S_\beta P_\beta\left(\overline{z}\varphi\right) \in K_\beta,\\
  &\chi=P_\alpha\left(\overline{\varphi}\right) \in K_\alpha.
\end{align*}
\end{lemma}

\begin{proof}
Let $f\in K_\alpha^\infty$ and $g\in K_\beta^\infty$. The functions
  $$S_\alpha^* f(z)=S^* f(z) =\frac{f(z)-f(0)}{z}$$
and
  $$S_\beta^* g(z)=S^* g(z)=\frac{g(z)-g(0)}{z}$$
clearly belong to $H^2\cap L^\infty=H^\infty$. Therefore,

\begin{align*}
  \left\langle S_\beta A_\varphi^{\alpha,\beta}S_\alpha^* f,g\right\rangle
  &=\left\langle P_\beta(\varphi S_\alpha^*f),S_\beta^* g\right\rangle
  =\left\langle \varphi \cdot\frac{f-f(0)}{z},S_\beta^* g\right\rangle\\
  &=\left\langle \overline{z}\varphi f,S_\beta^* g\right\rangle
  -f(0)\left\langle \overline{z}\varphi ,S_\beta^* g\right\rangle\\
  &=\left\langle \overline{z}\varphi f,\frac{g-g(0)}{z}\right\rangle
  -\left\langle f,k^\alpha_0 \right\rangle
  \left\langle S_\beta P_\beta(\overline{z}\varphi), g\right\rangle\\
  &=\left\langle \overline{z}\varphi f,\overline{z}g\right\rangle
  -\overline{g(0)}\left\langle \overline{z}\varphi f,\overline{z}\right\rangle
  -\left\langle (S_\beta P_\beta(\overline{z}\varphi)\otimes k^\alpha_0)f, g\right\rangle\\
  &=\left\langle A^{\alpha,\beta}_\varphi f, g\right\rangle
  -\langle k^\beta_0,g \rangle\left\langle f,P_\alpha(\overline{\varphi})\right\rangle
  -\left\langle (S_\beta P_\beta(\overline{z}\varphi)\otimes k^\alpha_0)f, g\right\rangle\\
  &=\left\langle A^{\alpha,\beta}_\varphi f, g\right\rangle
  -\langle (k^\beta_0\otimes P_\alpha(\overline{\varphi}))f,g\rangle
  -\left\langle (S_\beta P_\beta(\overline{z}\varphi)\otimes k^\alpha_0)f, g\right\rangle.
\end{align*}
From this
\begin{align}\label{eq_pr_ATTO_rank2_char_1}
   A^{\alpha,\beta}_\varphi f -S_\beta A^{\alpha,\beta}_\varphi S^*_\alpha f
   =(S_\beta P_\beta(\overline{z}\varphi)\otimes k^\alpha_0)f+(k^\beta_0\otimes P_\alpha(\overline{\varphi}))f \quad \text{for all } f\in K_\alpha^\infty.
\end{align}
For an arbitrary $f\in K_\alpha$ there exists $\{f_n\}\in K_\alpha^\infty$ such that $f_n\to f$ in $H^2$ norm and both $\varphi f_n\to \varphi f$ and $\varphi S^*_\alpha f_n\to\varphi S^*_\alpha f$ in $L^1$. Hence, $A_\varphi^{\alpha,\beta} f_n\to A_\varphi^{\alpha,\beta} f$ and $A_\varphi^{\alpha,\beta} S^*_\alpha f_n\to A_\varphi^{\alpha,\beta} S^*_\alpha f$ uniformly on compact subsets of $\mathbb{D}$, which implies that \eqref{eq_pr_ATTO_rank2_char_1} holds for all $f\in K_\alpha$.
\end{proof}


\begin{lemma}\label{lem_ATTO_rank2_aux}
If $\varphi=\overline{\chi}+\psi$, where $\chi\in K_\alpha$ and $\psi\in K_\beta$, $\psi(0)=0$, then the equality
\begin{align*}%\label{eq_ATTO_rank2_aux}
  \langle A_\varphi^{\alpha,\beta}f,g\rangle=\sum_{n=1}^\infty\langle(S^n_\beta \psi\otimes S^n_\alpha k^\alpha_0 +S^n_\beta k^\beta_0\otimes S^n_\alpha \chi)f,g\rangle
\end{align*}
holds for all $f\in K_\alpha^\infty$ and $g\in K_\beta^\infty$.
\end{lemma}
\begin{proof}
Note that if $\varphi=\overline{\chi}+\psi$, where $\chi\in K_\alpha$ and $\psi\in K_\beta$, $\psi(0)=0$, then
  $$P_\alpha(\overline{\varphi})=\chi \qquad \text{and }\qquad S_\beta P_\beta(\overline{z}\varphi)
  =S_\beta P_\beta(\overline{z}\psi)=S_\beta S^*_\beta \psi=\psi.$$
By Lemma \ref{lem_ATTO_rank2_char_aux1},
  $$A^{\alpha,\beta}_\varphi -S_\beta A^{\alpha,\beta}_\varphi S^*_\alpha
   =\psi\otimes k^\alpha_0+k^\beta_0\otimes \chi,$$
and for every $n\geq 0$,
\begin{align*}
  &S^n_\beta A_\varphi^{\alpha,\beta} (S^*_\alpha)^n -S^{n+1}_\beta A_\varphi^{\alpha,\beta} (S^*_\alpha)^{n+1}
  =S^n_\beta \psi\otimes S^n_\alpha k^\alpha_0+S^n_\beta k^\beta_0\otimes S^n_\alpha \chi.
\end{align*}
%holds on $H^2$.
From this
  $$A_\varphi^{\alpha,\beta}
  =\sum_{n=1}^N (S^n_\beta \psi\otimes S^n_\alpha k^\alpha_0
  +S^n_\beta k^\beta_0\otimes S^n_\alpha \chi)
  +S^{N+1}_\beta A_\varphi^{\alpha,\beta} (S^*_\alpha)^{N+1}$$
    for every $N\geq 1$.

    If $f\in K_\alpha^\infty$ and $g\in K_\beta^\infty$, then
\begin{displaymath}
\begin{split}
  \left\langle S^{N+1}_\beta\right.& \left. A_\varphi^{\alpha,\beta}(S^*_\alpha)^{N+1} f,g\right\rangle\\
  &=\left\langle A_{\overline{\chi}}^{\alpha,\beta}(S^*_\alpha)^{N+1} f,
  (S^*_\beta)^{N+1} g\right\rangle
  +\left\langle A_\psi^{\alpha,\beta}(S^*_\alpha)^{N+1} f,(S^*_\beta)^{N+1} g\right\rangle\\
  &=\left\langle T_{\overline{\chi}}(S^*)^{N+1} f,
  (S^*)^{N+1} g\right\rangle
  +\left\langle (S^*)^{N+1} f,T_{\overline{\psi}}(S^*)^{N+1} g\right\rangle\\
  &=\left\langle P(\overline{z}^{N+1}\overline{\chi}f) ,
   (S^*)^{N+1}g\right\rangle
  +\left\langle  (S^*)^{N+1}f, P(\overline{z}^{N+1}\overline{\psi}g)\right\rangle \to 0\quad \text{as } N\to 0
  \end{split}
\end{displaymath}
since $$\|P(\overline{z}^{N+1}\overline{\chi}f)\|_2\leq \|f\|_{\infty}\cdot\|\chi\|_2,\qquad \|P(\overline{z}^{N+1}\overline{\psi}g)\|_2\leq \|g\|_{\infty}\cdot\|\psi\|_2$$ and $(S^*)^N\to 0$ in the strong operator topology. Therefore
  $$\langle A_\varphi^{\alpha,\beta}f,g\rangle
  =\sum_{n=1}^\infty \langle (S^n_\beta \psi\otimes S^n_\alpha k^\alpha_0
  +S^n_\beta k^\beta_0\otimes S^n_\alpha \chi)f,g\rangle$$
  for all $f\in K_{\alpha}^{\infty}$ and $g\in K_{\beta}^{\infty}$.


\end{proof}

\begin{proof}[Proof of Theorem \ref{thm_ATTO_rank_two_char}]
If $A\in\mathscr{T}(\alpha,\beta)$, then $A=A_\varphi^{\alpha,\beta}$ for some $\varphi\in L^2$ and it satisfies \eqref{eq_ATTO_rank_two_char} by Lemma \ref{lem_ATTO_rank2_char_aux1}.

Assume now that $A$ is a bounded linear operator from $K_\alpha$ into $K_\beta$ such that \eqref{eq_ATTO_rank_two_char} holds for $\psi\in K_\beta$ and $\chi\in K_\alpha$. Without any loss of generality we can assume that $\psi(0)=0$. Indeed, if this was not the case we would replace $\psi$ and $\chi$ with $\psi-c k^\beta_0$ and $\chi+\overline{c} k^\alpha_0$, respectively, for $c=\psi(0)/(1-|\beta(0)|^2)$.

Define $\varphi=\overline{\chi}+\psi$. By Lemma \ref{lem_ATTO_rank2_aux}, for every $f\in K_{\alpha}^{\infty}$ and $g\in K_{\beta}^{\infty}$,
  $$\langle A_\varphi^{\alpha,\beta}f,g\rangle
  =\sum_{n=1}^\infty\langle (S^n_\beta \psi\otimes S^n_\alpha k^\alpha_0
  +S^n_\beta k^\beta_0\otimes S^n_\alpha \chi)f,g\rangle.$$
But an argument similar to the one given in the proof of Lemma \ref{lem_ATTO_rank2_char_aux1} shows that
  $$\langle Af,g\rangle=\sum_{n=1}^\infty \langle (S^n_\beta \psi\otimes S^n_\alpha k^\alpha_0
  +S^n_\beta k^\beta_0\otimes S^n_\alpha \chi)f,g\rangle,$$
and so $A=A_\varphi^{\alpha,\beta}\in \mathscr{T}(\alpha,\beta)$.
\end{proof}



\begin{corollary}\label{cor_ATTO_rank2}
Let $A$ be a bounded linear operator form $K_\alpha$ into $K_\beta$. Then $A\in\mathscr{T}(\alpha,\beta)$ if and only if
\begin{equation}\label{eq_ATTO_rank_two_char2}
A-S^*_\beta A S_\alpha=\psi\otimes \widetilde{k}^\alpha_0 +\widetilde{k}^\beta_0\otimes \chi
\end{equation}
for some $\psi\in K_\beta$ and $\chi\in K_\alpha$.
\end{corollary}

\begin{proof}

Recall first that the linear operator $A$ belongs to $\mathscr{T}(\alpha,\beta)$ if and only if $B=C_{\beta}AC_{\alpha}$ belongs to $\mathscr{T}(\alpha,\beta)$ (see \cite[p. 9]{L2}).

If $A\in\mathscr{T}(\alpha,\beta)$, then $C_{\beta}AC_{\alpha}\in\mathscr{T}(\alpha,\beta)$. By Theorem \ref{thm_ATTO_rank_two_char}, there exist functions $\chi_0\in K_{\alpha}$ and $\psi_0\in K_{\beta}$ such that
\begin{equation*}%\label{con4}
C_{\beta}AC_{\alpha}-S_{\beta}C_{\beta}AC_{\alpha} S_{\alpha}^{*}=\psi_0\otimes k_{0}^{\alpha}+k_{0}^{\beta}\otimes\chi_0.
\end{equation*}
Thus (using the symmetry of compressed shifts) we get

\begin{displaymath}
\begin{split}
A-S_{\beta}^{*}A S_{\alpha}&=C_{\beta}^2AC_{\alpha}^2-C_{\beta}S_{\beta}C_{\beta}A C_{\alpha}S_{\alpha}^{*}C_{\alpha}\\
&=C_{\beta}(\psi_0\otimes k_{0}^{\alpha}+k_{0}^{\beta}\otimes\chi_0)C_{\alpha}=\widetilde{\psi}_0\otimes \widetilde{k}_{0}^{\alpha}+\widetilde{k}_{0}^{\beta}\otimes\widetilde{\chi}_0,
\end{split}
\end{displaymath}
and $A$ satisfies \eqref{eq_ATTO_rank_two_char2} with
$$\psi=\widetilde{\psi}_0\quad\mathrm{and}\quad\chi=\widetilde{\chi}_0.$$

The other implication can be proved in a similar way.
\end{proof}

Conditions from Corollary \ref{cor_ATTO_rank2} can also be formulated in terms of modified compressed shifts.

\begin{corollary}\label{cor_ATTO_rank2_MCS}
Let $A$ be a bounded linear operator form $K_\alpha$ into $K_\beta$. Then $A\in\mathscr{T}(\alpha,\beta)$ if and only if one (and all) of the following conditions holds
\begin{enumerate}
  \item[(a)] $A-S_{\beta,b} A S^*_{\alpha,a}=\psi\otimes k^\alpha_0 +k^\beta_0\otimes \chi$;
  \item[(b)] $A-S^*_{\beta,b} A S_{\alpha,a}=\psi\otimes \widetilde{k}^\alpha_0 +\widetilde{k}^\beta_0\otimes \chi$;
\end{enumerate}
for some $a,b\in\mathbb{C}$, $\psi\in K_\beta$ and $\chi\in K_\alpha$ (possibly different for different conditions), where
  $$S_{\alpha,a}=S_\alpha+a (k^\alpha_0\otimes \widetilde{k}^\alpha_0),\qquad
  S_{\beta,b}=S_\beta+ b(k^\beta_0\otimes \widetilde{k}^\beta_0)$$
are the modified compressed shifts.
\end{corollary}

\begin{proof}
The proof uses Corollary \ref{cor_ATTO_rank2} and is analogous to the proof given in \cite[Theorem 7.1]{s}. The details are therefore left to the reader.
\end{proof}

\begin{corollary}
$\mathscr{T}(\alpha,\beta)$ is closed in the weak operator topology.
\end{corollary}
\begin{proof}
See \cite{s}, \cite{ptak} or \cite{L2}.
\end{proof}

Using Corollary \ref{cor_ATTO_rank2} we can now characterize the operators from $\mathscr{T}(\alpha,\beta)$ in terms of shift invariance. The following corollary was first noted in \cite{ptak} (for the case when $\beta$ divides $\alpha$).

\begin{corollary}\label{cor_ATTO_rank2_SI}
Let $A$ be a bounded linear operator form $K_\alpha$ into $K_\beta$. Then $A\in\mathscr{T}(\alpha,\beta)$ if and only if it has the following property
$$\left\langle AS f, S g\right\rangle=\left\langle A f,g \right\rangle$$
 for all $f\in K_\alpha$, $g\in K_\beta$ such that $Sf\in K_\alpha$, $Sg\in K_\beta$.
\end{corollary}
\begin{proof}
See \cite{L2}, \cite{ptak}.
\end{proof}

\begin{thebibliography}{99}
\bibitem{ptak} C. C\^{a}mara, J. Jurasik, K. Kli\'{s}-Garlicka, M. Ptak, \emph{Characterizations of asymmetric truncated Toeplitz operators,} Banach J. Math. Anal. (to appear), arXiv:1607.03342.


\bibitem{part2} M. C. C\^{a}mara, J. R. Partington, \emph{Spectral properties of truncated Toeplitz operators by equivalence after extension,} J. Math. Anal. and Appl. \textbf{433} (2016), no. 2, 762--784 .%arXiv:1504.06446.


\bibitem{part} M. C. C\^{a}mara, J. R. Partington, \emph{Asymmetric truncated Toeplitz operators and Toeplitz operators with matrix symbol,} J. Operator Theory \textbf{77} (2017), no. 2, 455--479 .%arXiv:1504.06446.

\bibitem{gar3} S. R. Garcia, W. T. Ross, \emph{Recent progress on truncated Toeplitz operators,} in: J. Mashreghi, E. Fricain (Eds.), Blaschke products and their applications, Fields Inst. Commun., \textbf{65}, Springer, New York, 2013, 275--319.

\bibitem{blicharz1} J. Jurasik, B. \L anucha, \emph{Asymmetric truncated Toeplitz operators equal to the zero operator,} Ann. Univ. Mariae Curie-Sk\l odowska, Sect. A \textbf{70} (2016), no. 2, 51--62.

\bibitem{blicharz2} J. Jurasik, B. \L anucha, \emph{Asymmetric truncated Toeplitz operators on finite-dimensional spaces,} Operators and Matrices 11 (2017), no. 1, 245--262.


\bibitem{L2} B. {\L}anucha, \emph{On rank-one asymmetric truncated Toeplitz operators on finite-dimensional model spaces}, J. Math. Anal. Appl. %\textbf{}
    (2017), http://dx.doi.org/10.1016/j.jmaa.2017.05.033.
    %430--437.


\bibitem{s} D. Sarason, \emph{Algebraic properties of truncated Toeplitz operators,} Operators and Matrices \textbf{1} (2007), no. 4, 491--526.

\end{thebibliography}

\end{document}
  \bibitem{}, \emph{}, {\bf } (19), --.
"""
    statements_1705 = [{'stmt_id': '1705_10703::s0000', 'type': 'Theorem', 'text': '\\label{thm_ATTO_rank_two_char}\nLet $A$ be a bounded linear operator from $K_\\alpha$ into $K_\\beta$. Then $A\\in\\mathscr{T(\\alpha,\\beta)}$ if and only if there exist $\\psi\\in K_\\beta$ and $\\chi\\in K_\\alpha$ such that\n\\begin{align}\\label{eq_ATTO_rank_two_char}\n  A-S_\\beta A S^*_\\alpha=\\psi\\otimes k^\\alpha_0 +k^\\beta_0\\otimes \\chi.\n\\end{align}'}, {'stmt_id': '1705_10703::s0001', 'type': 'Lemma', 'text': '\\label{lem_ATTO_rank2_char_aux1}\nFor every $\\varphi\\in L^2$ the equality\n\\begin{align*}\n  A^{\\alpha,\\beta}_\\varphi-S_\\beta A^{\\alpha,\\beta}_\\varphi S^*_\\alpha=\\psi\\otimes k^\\alpha_0 +k^\\beta_0\\otimes \\chi\n\\end{align*}\nholds on $H^2$, where\n\\begin{align*}\n  &\\psi=S_\\beta P_\\beta\\left(\\overline{z}\\varphi\\right) \\in K_\\beta,\\\\\n  &\\chi=P_\\alpha\\left(\\overline{\\varphi}\\right) \\in K_\\alpha.\n\\end{align*}'}, {'stmt_id': '1705_10703::s0002', 'type': 'Lemma', 'text': '\\label{lem_ATTO_rank2_aux}\nIf $\\varphi=\\overline{\\chi}+\\psi$, where $\\chi\\in K_\\alpha$ and $\\psi\\in K_\\beta$, $\\psi(0)=0$, then the equality\n\\begin{align*}\n  \\langle A_\\varphi^{\\alpha,\\beta}f,g\\rangle=\\sum_{n=1}^\\infty\\langle(S^n_\\beta \\psi\\otimes S^n_\\alpha k^\\alpha_0 +S^n_\\beta k^\\beta_0\\otimes S^n_\\alpha \\chi)f,g\\rangle\n\\end{align*}\nholds for all $f\\in K_\\alpha^\\infty$ and $g\\in K_\\beta^\\infty$.'}, {'stmt_id': '1705_10703::s0003', 'type': 'Corollary', 'text': '\\label{cor_ATTO_rank2}\nLet $A$ be a bounded linear operator form $K_\\alpha$ into $K_\\beta$. Then $A\\in\\mathscr{T}(\\alpha,\\beta)$ if and only if\n\\begin{equation}\\label{eq_ATTO_rank_two_char2}\nA-S^*_\\beta A S_\\alpha=\\psi\\otimes \\widetilde{k}^\\alpha_0 +\\widetilde{k}^\\beta_0\\otimes \\chi\n\\end{equation}\nfor some $\\psi\\in K_\\beta$ and $\\chi\\in K_\\alpha$.'}, {'stmt_id': '1705_10703::s0004', 'type': 'Corollary', 'text': '\\label{cor_ATTO_rank2_MCS}\nLet $A$ be a bounded linear operator form $K_\\alpha$ into $K_\\beta$. Then $A\\in\\mathscr{T}(\\alpha,\\beta)$ if and only if one (and all) of the following conditions holds\n\\begin{enumerate}\n  \\item[(a)] $A-S_{\\beta,b} A S^*_{\\alpha,a}=\\psi\\otimes k^\\alpha_0 +k^\\beta_0\\otimes \\chi$;\n  \\item[(b)] $A-S^*_{\\beta,b} A S_{\\alpha,a}=\\psi\\otimes \\widetilde{k}^\\alpha_0 +\\widetilde{k}^\\beta_0\\otimes \\chi$;\n\\end{enumerate}\nfor some $a,b\\in\\mathbb{C}$, $\\psi\\in K_\\beta$ and $\\chi\\in K_\\alpha$ (possibly different for different conditions), where\n  $$S_{\\alpha,a}=S_\\alpha+a (k^\\alpha_0\\otimes \\widetilde{k}^\\alpha_0),\\qquad\n  S_{\\beta,b}=S_\\beta+ b(k^\\beta_0\\otimes \\widetilde{k}^\\beta_0)$$\nare the modified compressed shifts.'}, {'stmt_id': '1705_10703::s0005', 'type': 'Corollary', 'text': '$\\mathscr{T}(\\alpha,\\beta)$ is closed in the weak operator topology.'}, {'stmt_id': '1705_10703::s0006', 'type': 'Corollary', 'text': '\\label{cor_ATTO_rank2_SI}\nLet $A$ be a bounded linear operator form $K_\\alpha$ into $K_\\beta$. Then $A\\in\\mathscr{T}(\\alpha,\\beta)$ if and only if it has the following property\n$$\\left\\langle AS f, S g\\right\\rangle=\\left\\langle A f,g \\right\\rangle$$\n for all $f\\in K_\\alpha$, $g\\in K_\\beta$ such that $Sf\\in K_\\alpha$, $Sg\\in K_\\beta$.'}]

    output_1705 = r"""As usual, let $H^2$ denote the classical Hardy space. The space $H^2$ can be seen as a space of functions analytic in the unit disk
    \mathbb{D}=\{z:|z|<1\}$ or as a closed
    subspace of $L^2:=L^2(\partial\mathbb{D})$. In the first case $H^2$ consists of functions analytic in $\mathbb{D}$ with square summable MacLaurin coefficients and in the second it consists of functions from $L^2$ such that their Fourier coefficients with negative indices vanish.
    ===BLOCK_SEPARATOR===
    The unilateral shift $S$ on $H^2$ is the operator of multiplication by the independent variable, that is,
      $$Sf(z)=z\cdot f(z).$$
    The adjoint of $S^*$ of $S$ is called the backward shift. A simple verification shows that
      $$S^*f(z)=\frac{f(z)-f(0)}{z}.$$
    ===BLOCK_SEPARATOR===
    The famous Beurling theorem provides a characterization of all $S^*$-invariant subspaces of $H^2$. Namely, a closed nontrivial subspace of $H^2$ is $S^*$-invariant if and only if it is of the form
      $$K_\alpha=H^2\ominus \alpha H^2,$$
    where $\alpha$ is an inner function, i.e., $\alpha$ belongs to the algebra $H^{\infty}$ of bounded analytic functions and $|\alpha|=1$ a.e. on $\partial
    \mathbb{D}$. The space $K_{\alpha}$ is called the model space associated with $\alpha$.
    ===BLOCK_SEPARATOR===
    Truncated Toeplitz operators are compressions of Toeplitz operators to model spaces. More precisely, a truncated Toeplitz operator $A_{\varphi}^{\alpha}$ with a symbol $\varphi\in L^2$ is defined on the model space $K_{\alpha}$ by
      $$A_{\varphi}^{\alpha}f=P_{\alpha}(\varphi f),$$
    where $P_{\alpha}$ is the orthogonal projection from $L^2$ onto $K_{\alpha}$. In particular, $S_\alpha=A^\alpha_z$ is called the compressed shift. Since $K_\alpha$ is $S^*$-invariant, it easily follows that $S^*_\alpha=S^*_{|K_\alpha}$.
    ===BLOCK_SEPARATOR===
    Let $\alpha$, $\beta$ be two inner functions. An asymmetric
    truncated Toeplitz operator $A_{\varphi}^{\alpha,\beta}$ with a symbol $\varphi\in
    L^2$ is the
    operator from $K_{\alpha}$ into $K_{\beta}$ given by
      $$A_{\varphi}^{\alpha,\beta}f=P_{\beta}(\varphi f).$$
    Clearly, $A_{\varphi}^{\alpha}=A_{\varphi}^{\alpha,\alpha}$.
    Put
      $$\mathscr{T}(\alpha,\beta)=\{A_{\varphi}^{\alpha,\beta}\colon\, \varphi\in
      L^2\ \mathrm{and}\ A_{\varphi}^{\alpha,\beta}\
      \mathrm{is\ bounded}\}.$$
    ===BLOCK_SEPARATOR===
    Recall that  a model space $K_{\alpha}$ is a reproducing kernel Hilbert space. That is to say that for every $f$ in the model space $K_{\alpha}$ and each $w\in\mathbb{D}$,
      $$f(w)=\langle f, k_{w}^{\alpha}\rangle,$$
    where the reproducing kernel function $k_{w}^{\alpha}$ is of the form
      $$k_{w}^{\alpha}(z)=\frac{1-\overline{\alpha(w)}\alpha(z)}{1-\overline{w}z}.$$
      Observe that since $k_{w}^{\alpha}\in H^{\infty}$, the set $K_{\alpha}^{\infty}=K_{\alpha}\cap H^{\infty}$ is dense in $K_{\alpha}$.
    ===BLOCK_SEPARATOR===
    A conjugate kernel is the function
    $\widetilde{k}_{w}^{\alpha}=C_{\alpha}{k}_{w}^{\alpha}$, where $C_{\alpha}:L^2\to L^2$ is given by
      $$C_{\alpha}f(z)=\widetilde{f}(z)=\alpha(z)\overline{z}\overline{f(z)},\quad |z|=1.$$
    ===END_OF_EXTRACTION==="""
    new_messages = [
        {
            "role": "system",
            "content": """You are an expert mathematical information extractor. Your task is to identify and extract foundational definitions and notation from a LaTeX paper that are strictly necessary to understand a specific set of extracted theorems and lemmas.

    INPUT:
    1. **Input Paper**: The full raw LaTeX source of a mathematical paper.
    2. **Captured Statements**: A JSON list of mathematical statements (Theorems, Lemmas, Corollaries) that have already been extracted.

    TASK:
    - Read the "Captured Statements" and identify the custom mathematical terms, spaces, operators, mappings, functions, and symbols they rely upon.
    - Pay special attention to arbitrary maps, functions, or morphisms (e.g., $q_u$, $\phi$, $\Lambda^*$) used in the statements.
    - Scan the "Input Paper" to find the exact paragraphs or text blocks where these dependencies are DEFINED or INTRODUCED for the first time.
    - Extract these defining paragraphs exactly as they appear in the raw LaTeX source.

    CONSTRAINTS:
    - DO NOT paraphrase or summarize. Copy the text EXACTLY as it appears in the source.
    - DO NOT extract any text that is already present inside the "Captured Statements" list.
    - ONLY extract the primary definition. Do not extract every paragraph that mentions a symbol.
    - NEVER output the same paragraph twice.
    - Extract complete logical paragraphs so the mathematical context is preserved.

    OUTPUT FORMAT:
    - Output only the raw LaTeX blocks.
    - You MUST separate each distinct extracted block with the exact string: ===BLOCK_SEPARATOR===
    - You MUST end your entire response with the exact string: ===END_OF_EXTRACTION===
    - Do not output any lists, code blocks, JSON, or conversational filler. 

    Example Output:
    Let $\mathcal{{H}}$ be a Hilbert space...
    ===BLOCK_SEPARATOR===
    A function $f$ is called \emph{{smooth}} if...
    ===END_OF_EXTRACTION==="""
        },
        {
            "role": "user",
            "content": f"""---
    **Input Paper:**
    {paper_1705_10703}

    **Captured Statements:**
    {json.dumps(statements_1705, indent=2)}
    ---
    """
        },
        {
            "role": "assistant",
            "content": output_1705
        },
        {
            "role": "user",
            "content": f"""Excellent. Now, apply the exact same process to the following new statements.

    ---
    **Input Paper:**
    {latex_paper}

    **Captured Statements:**
    {json.dumps(statements, indent=2)}
    ---
    """
        }
    ]

    old_messages = [
    {
        "role": "user",
        "content": f"""You are an expert mathematical information extractor. Your task is to identify and extract foundational definitions and notation from a LaTeX paper that are strictly necessary to understand a specific set of extracted theorems and lemmas.

INPUT:
1. **Input Paper**: The full raw LaTeX source of a mathematical paper.
2. **Captured Statements**: A JSON list of mathematical statements (Theorems, Lemmas, Corollaries) that have already been extracted.

TASK:
- Read the "Captured Statements" and identify the custom mathematical terms, spaces, operators, functions, and symbols they rely upon.
- Scan the "Input Paper" to find the exact paragraphs or text blocks where these dependencies are DEFINED or INTRODUCED for the first time.
- Extract these defining paragraphs exactly as they appear in the raw LaTeX source.

CONSTRAINTS:
- DO NOT paraphrase or summarize. Copy the text EXACTLY as it appears in the source.
- DO NOT extract any text that is already present inside the "Captured Statements" list.
- ONLY extract the primary definition. Do not extract every paragraph that mentions a symbol.
- NEVER output the same paragraph twice.
- Extract complete logical paragraphs so the mathematical context is preserved.

OUTPUT FORMAT:
- Output only the raw LaTeX blocks.
- You MUST separate each distinct extracted block with the exact string: ===BLOCK_SEPARATOR===
- You MUST end your entire response with the exact string: ===END_OF_EXTRACTION===
- Do not output any lists, code blocks, JSON, or conversational filler. 

Example Output:
Let $\mathcal{{H}}$ be a Hilbert space...
===BLOCK_SEPARATOR===
A function $f$ is called \emph{{smooth}} if...
===END_OF_EXTRACTION===
---
**Input Paper:**
{paper_1705_10703}

**Captured Statements:**
{json.dumps(statements_1705, indent=2)}
---
"""
    },
    {
        "role": "assistant",
        "content": output_1705
    },
    {
        "role": "user",
        "content": f"""Excellent, that's the exact correct format. Now, apply the exact same process to the following new statements.

---
**Input Paper:**
{latex_paper}

**Captured Statements:**
{json.dumps(statements, indent=2)}
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
        print("WARNING: Too many tokens in inputs")
        return None
        raise Exception("Token count exceeded 100k limit!")

    print("Generating...")
    end_think_token_id = tokenizer.encode("</think>", add_special_tokens=False)[-1]

    # Initialize the processor. Let's give it 5,000 tokens of "thinking budget".
    # Since max_new_tokens is 8192, this guarantees at least 3,192 tokens remain for the JSON.
    logits_processor = LogitsProcessorList([
        ForceEndThinkLogitsProcessor(max_think_tokens=15000, end_think_token_id=end_think_token_id)
    ])

    # 3. Generate (Notice the ** unpacking operator, this passes both input_ids and attention_mask!)
    outputs = model.generate(
        **inputs,
        max_new_tokens=30384,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.eos_token_id,
        do_sample = False,
        temperature = None,
    )

    # 4. Decode ONLY the newly generated tokens
    result = tokenizer.decode(outputs[0][num_input_tokens:], skip_special_tokens=True)
    #cleaned = clean_and_verify_json(result)
    return result


def get_new_stmts(response_text):
    # Handle thinking block
    if "</think>" in response_text:
        response_text = response_text.split("</think>")[-1].strip()

    # Strip the ending token
    response_text = response_text.replace("===END_OF_EXTRACTION===", "")

    # Split by the exact delimiter
    extracted_definitions = response_text.split("===BLOCK_SEPARATOR===")

    # Clean up whitespace and strictly DEDUPLICATE
    clean_definitions = []
    seen_blocks = set()

    for block in extracted_definitions:
        clean_block = block.strip()
        if clean_block and clean_block not in seen_blocks:
            seen_blocks.add(clean_block)
            clean_definitions.append(clean_block)

    return clean_definitions


NUM_GPUS = 4


def worker(gpu_id, task_queue, result_queue):
    try:
        torch.cuda.set_device(gpu_id)
        model, tokenizer = load_model_and_tokenizer(gpu_id)
    except Exception as e:
        result_queue.put(("FATAL", gpu_id, repr(e)))   # tell the writer, don't just vanish
        return
    result_queue.put(("READY", gpu_id))
    while True:
        item = task_queue.get()
        if item is None:
            break
        paper_id, display = item
        try:
            pap_text = get_paper(paper_id)
            if not pap_text:
                result_queue.put(("SKIP", paper_id)); continue
            twee = extract_definitions(pap_text, display, model, tokenizer)
            if twee is None:
                result_queue.put(("SKIP", paper_id)); continue
            stmt_list = get_new_stmts(twee)
            if not stmt_list:
                result_queue.put(("SKIP", paper_id)); continue
            d2 = [{"stmt_id": f"{paper_id}::a{ind:04d}", "type": "AI Defs", "text": s}
                  for ind, s in enumerate(stmt_list)]
            result_queue.put(("OK", {paper_id: d2 + display}))
        except Exception as e:
            result_queue.put(("ERR", paper_id, str(e)))
    result_queue.put(("DONE", gpu_id))



def main():
    # already-processed papers (resume support)
    processed = set()
    if os.path.exists(EXTRA_STATEMENTS_FILE):
        with open(EXTRA_STATEMENTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    processed.update(json.loads(line).keys())
    print(f"Found {len(processed)} already processed papers. Skipping those...")

    # group statements by paper
    statements = {}
    with open(STATEMENTS_IN, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="loading statements"):
            s = json.loads(line)
            statements.setdefault(s["paper_id"], []).append(s)

    # build the task list
    tasks = []
    for paper_id, x in statements.items():
        if paper_id in processed:
            continue
        display = [{"stmt_id": y["stmt_id"], "type": y["type"], "text": y["text"]} for y in x]
        tasks.append((paper_id, display))
    print(f"{len(tasks)} papers to process across {NUM_GPUS} GPUs.")

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for t in tasks:
        task_queue.put(t)
    for _ in range(NUM_GPUS):
        task_queue.put(None)        # one sentinel per worker

    procs = [mp.Process(target=worker, args=(g, task_queue, result_queue))
             for g in range(NUM_GPUS)]
    for p in procs:
        p.start()

    alive = NUM_GPUS
    with open(EXTRA_STATEMENTS_FILE, "a", encoding="utf-8") as pf, \
         tqdm(total=len(tasks), desc="extracting") as pbar:
        while alive > 0:
            try:
                msg = result_queue.get(timeout=300)
            except pyqueue.Empty:
                if not any(p.is_alive() for p in procs):
                    break
                continue
            tag = msg[0]
            if tag == "OK":
                pf.write(json.dumps(msg[1], ensure_ascii=False) + "\n"); pf.flush(); pbar.update(1)
            elif tag == "SKIP":
                pbar.update(1)
            elif tag == "ERR":
                print(f"\n[ERROR] {msg[1]}: {msg[2]}"); pbar.update(1)
            elif tag == "READY":
                print(f"GPU {msg[1]} ready.")
            elif tag == "FATAL":
                print(f"\n[FATAL] GPU {msg[1]} could not load model: {msg[2]}")
                alive -= 1
            elif tag == "DONE":
                alive -= 1

    task_queue.cancel_join_thread()      # <-- lets the process exit even with a full queue
    result_queue.cancel_join_thread()
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    print("All done.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # required for CUDA in child procs
    main()
