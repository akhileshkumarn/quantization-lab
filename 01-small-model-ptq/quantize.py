"""Post-training quantization of a small LLM with llm-compressor.

Applies one scheme, measures WikiText-2 perplexity on the quantized model,
saves it in compressed-tensors format and records the on-disk size.

    python quantize.py --scheme w4a16-gptq
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen2.5-0.5B"
SCHEMES = ["baseline", "w8a8-rtn", "w4a16-rtn", "w4a16-gptq"]

SEQ_LEN = 512
NUM_CALIB = 128  # calibration sequences (GPTQ only)
NUM_EVAL = 40  # evaluation windows -> ~20k tokens

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
MODELS_DIR = HERE.parent / "models"
PROMPT = "The key idea behind quantizing a neural network is"


def wikitext_tokens(tokenizer, split):
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(data["text"])
    return tokenizer(text, return_tensors="pt").input_ids[0]


def calibration_dataset(tokenizer):
    """Non-overlapping train windows; eval uses the test split, so no leakage."""
    ids = wikitext_tokens(tokenizer, "train")
    chunks = [ids[i * SEQ_LEN : (i + 1) * SEQ_LEN].tolist() for i in range(NUM_CALIB)]
    return Dataset.from_dict(
        {"input_ids": chunks, "attention_mask": [[1] * SEQ_LEN for _ in chunks]}
    )


@torch.no_grad()
def perplexity(model, tokenizer):
    ids = wikitext_tokens(tokenizer, "test")
    nll, count = 0.0, 0
    for i in range(NUM_EVAL):
        window = ids[i * SEQ_LEN : (i + 1) * SEQ_LEN].unsqueeze(0).to(model.device)
        loss = model(window, labels=window).loss
        nll += loss.item() * (window.shape[1] - 1)
        count += window.shape[1] - 1
    return math.exp(nll / count)


@torch.no_grad()
def sample(model, tokenizer):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def recipe_for(scheme):
    if scheme == "w8a8-rtn":
        # int8 weights (static, per channel) + int8 activations (dynamic, per token)
        return QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])
    if scheme == "w4a16-rtn":
        # int4 weights, group size 128, plain round-to-nearest
        return QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    if scheme == "w4a16-gptq":
        # same format as w4a16-rtn, but GPTQ picks the rounding using calibration data
        return GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    raise ValueError(scheme)


def dir_size_mb(path):
    return sum(f.stat().st_size for f in path.glob("*.safetensors")) / 2**20


def cast_floats_to_bf16(path):
    """Rewrite the fp32 tensors of a saved checkpoint (embedding, norms, biases,
    scales) as bf16, the serving dtype, so sizes compare fairly with the baseline."""
    for file in path.glob("*.safetensors"):
        tensors = {
            name: t.to(torch.bfloat16) if t.dtype == torch.float32 else t
            for name, t in load_file(file).items()
        }
        save_file(tensors, file, metadata={"format": "pt"})
    config_file = path / "config.json"
    config = json.loads(config_file.read_text())
    config["torch_dtype"] = "bfloat16"
    config_file.write_text(json.dumps(config, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=SCHEMES, required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    start = time.time()
    if args.scheme == "w4a16-gptq":
        oneshot(
            model=model,
            dataset=calibration_dataset(tokenizer),
            recipe=recipe_for(args.scheme),
            max_seq_length=SEQ_LEN,
            num_calibration_samples=NUM_CALIB,
        )
    elif args.scheme != "baseline":
        oneshot(model=model, recipe=recipe_for(args.scheme))
    quant_seconds = time.time() - start

    ppl = perplexity(model, tokenizer)
    text = sample(model, tokenizer)

    save_dir = MODELS_DIR / f"{MODEL_ID.split('/')[-1]}-{args.scheme}"
    if args.scheme == "baseline":
        # bf16 is the usual serving dtype, so it is the fair size reference
        model.to(torch.bfloat16).save_pretrained(save_dir)
    else:
        model.save_pretrained(save_dir, save_compressed=True)
        # quantized in fp32, but the unquantized tensors are served in bf16
        cast_floats_to_bf16(save_dir)
    tokenizer.save_pretrained(save_dir)

    result = {
        "model": MODEL_ID,
        "scheme": args.scheme,
        "wikitext2_ppl": round(ppl, 3),
        "size_mb": round(dir_size_mb(save_dir), 1),
        "quant_seconds": round(quant_seconds, 1),
        "eval_tokens": NUM_EVAL * SEQ_LEN,
        "sample": text,
        "torch": torch.__version__,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{args.scheme}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
