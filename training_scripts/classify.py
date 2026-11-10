"""
classify.py -- run the trained Qwen3.5-4B category classifier on new text.

The checkpoint is a LoRA adapter + a score head on a CUSTOM nn.Module, not an HF
auto class, so we must rebuild the exact same architecture before loading it.
Everything that must match training (label order, max_length, left padding,
last-token pooling) is read from classifier_config.json inside the checkpoint --
not retyped here, so it cannot silently drift.

Usage:
    # single text
    python classify.py --text "Let $X$ be a smooth projective variety over $k$..."

    # a file of texts, one per line
    python classify.py --file chunks.txt

    # a jsonl file with a "text" field -> writes predictions.jsonl
    python classify.py --jsonl chunks.jsonl --out predictions.jsonl

    # pipe from stdin
    cat chunks.txt | python classify.py

Options:
    --model DIR      checkpoint dir (default: qwen35_4b_category6/best_model)
    --batch-size N   default 8
    --probs          show the full probability distribution, not just the winner
    --no-4bit        load in bf16 instead of 4-bit (faster, needs ~9GB VRAM)
"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import PeftModel


# ---------------------------------------------------------------------------
# architecture -- must match train_classifier.py exactly
# ---------------------------------------------------------------------------
class QwenSequenceClassifier(nn.Module):
    def __init__(self, trunk, hidden_size, num_labels):
        super().__init__()
        self.trunk = trunk
        self.score = nn.Linear(hidden_size, num_labels, bias=False, dtype=torch.float32)
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.trunk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        pooled = hidden[:, -1, :].float()      # LEFT padding -> last real token is -1
        logits = self.score(pooled)
        return {"logits": logits}
    @property
    def device(self):
        return next(self.parameters()).device


def load_classifier(model_dir, use_4bit=True):
    cfg_path = os.path.join(model_dir, "classifier_config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(
            f"No classifier_config.json in {model_dir!r}. That file carries the label "
            f"map and pooling convention; without it we'd be guessing which class is "
            f"which. Is this the right checkpoint dir?"
        )
    with open(cfg_path) as f:
        ccfg = json.load(f)

    categories = ccfg["categories"]
    base_model = ccfg["base_model"]
    max_length = ccfg["max_length"]

    if ccfg.get("padding_side") != "left" or ccfg.get("pooling") != "last_token_index_-1":
        raise SystemExit(
            f"Checkpoint was trained with padding_side={ccfg.get('padding_side')} / "
            f"pooling={ccfg.get('pooling')}, which this script does not implement. "
            f"Pooling the wrong position gives confident nonsense, not an error."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"loading {base_model} ({'4-bit' if use_4bit else 'bf16'}) ...", file=sys.stderr)
    causal_lm = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        quantization_config=quant_config,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    trunk = causal_lm.model
    causal_lm.lm_head = None

    hf_cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    hidden_size = getattr(hf_cfg, "text_config", hf_cfg).hidden_size

    model = QwenSequenceClassifier(trunk, hidden_size, len(categories))

    # adapter weights + the `score` head (saved via modules_to_save at train time)
    print(f"loading adapter from {model_dir} ...", file=sys.stderr)
    model = PeftModel.from_pretrained(model, model_dir)
    # trunk went to GPU via device_map, but the fresh score head (and its PEFT
    # wrapper) was created on CPU and from_pretrained doesn't move it
    dev = next(model.parameters()).device
    for name, module in model.named_modules():
        if any(p.device.type == "cpu" for p in module.parameters(recurse=False)):
            module.to(dev)
    model.eval()

    # Loud check: an unloaded score head is all-zeros/random and produces a
    # uniform-ish garbage distribution rather than a crash.
    score_w = model.base_model.model.score.modules_to_save["default"].weight \
        if hasattr(model.base_model.model.score, "modules_to_save") \
        else model.base_model.model.score.weight
    if torch.allclose(score_w, torch.zeros_like(score_w)):
        raise SystemExit("score head loaded as all zeros -- adapter did not restore it.")

    return model, tokenizer, categories, max_length


@torch.no_grad()
def classify(texts, model, tokenizer, categories, max_length, batch_size=8):
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        logits = model(**enc)["logits"]
        probs = torch.softmax(logits, dim=-1).cpu()

        for row in probs:
            idx = int(row.argmax())
            results.append({
                "category": categories[idx],
                "confidence": float(row[idx]),
                "probs": {c: float(row[j]) for j, c in enumerate(categories)},
            })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35_4b_category6/best_model")
    ap.add_argument("--text", help="classify a single string")
    ap.add_argument("--file", help="plain text file, one chunk per line")
    ap.add_argument("--jsonl", help="jsonl file with a 'text' field")
    ap.add_argument("--out", help="write jsonl predictions here (with --jsonl)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--probs", action="store_true", help="show full distribution")
    ap.add_argument("--no-4bit", action="store_true", help="load bf16 instead of 4-bit")
    args = ap.parse_args()

    # ---- gather inputs ----
    rows = None
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    elif args.jsonl:
        rows = [json.loads(ln) for ln in open(args.jsonl) if ln.strip()]
        texts = [r["text"] for r in rows]
    elif not sys.stdin.isatty():
        texts = [ln.strip() for ln in sys.stdin if ln.strip()]
    else:
        raise SystemExit("Give one of --text / --file / --jsonl, or pipe text on stdin.")

    if not texts:
        raise SystemExit("No input texts found.")

    model, tokenizer, categories, max_length = load_classifier(
        args.model, use_4bit=not args.no_4bit
    )
    print(f"classifying {len(texts):,} text(s) ...", file=sys.stderr)
    preds = classify(texts, model, tokenizer, categories, max_length, args.batch_size)

    # ---- output ----
    if args.jsonl and args.out:
        with open(args.out, "w") as f:
            for row, p in zip(rows, preds):
                row.update({"pred_category": p["category"],
                            "pred_confidence": p["confidence"],
                            "pred_probs": p["probs"]})
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(preds):,} predictions -> {args.out}", file=sys.stderr)

        # if the input happened to carry gold labels, score it for free
        gold = [r.get("category") for r in rows]
        if all(g is not None for g in gold):
            correct = sum(g == p["category"] for g, p in zip(gold, preds))
            print(f"accuracy vs 'category' field: {correct / len(preds):.4f}", file=sys.stderr)
    else:
        for text, p in zip(texts, preds):
            preview = text[:70].replace("\n", " ")
            if args.probs:
                dist = "  ".join(f"{c}={p['probs'][c]:.3f}" for c in categories)
                print(f"{p['category']}  ({p['confidence']:.3f})  | {preview}...")
                print(f"    {dist}")
            else:
                print(f"{p['category']}\t{p['confidence']:.3f}\t{preview}...")


if __name__ == "__main__":
    main()
