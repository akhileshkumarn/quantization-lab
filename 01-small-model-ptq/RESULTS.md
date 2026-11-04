# Experiment 01 results: PTQ of Qwen2.5-0.5B

Qwen/Qwen2.5-0.5B, llm-compressor `oneshot`, Apple M1 8 GB, torch 2.14.1.
WikiText-2 test perplexity, 40 non-overlapping windows of 512 tokens (20,480 tokens), evaluated in memory right
after quantization (fp32 compute). The bf16 baseline ran on CPU. After `oneshot` the quantized models ran on the
M1 GPU (MPS), with weights held in CPU memory and copied to the GPU layer by layer. The effect of this device
difference on perplexity has not been measured yet.
Size = `model.safetensors` with every unquantized tensor in bf16.

| scheme | WikiText-2 PPL | Δ PPL | size (MiB) | vs bf16 | quantize time (s) |
|---|---|---|---|---|---|
| baseline (bf16) | 16.52 | +0.00 | 942 | 1.00x | 0 |
| w8a8-rtn | 16.99 | +0.48 | 602 | 0.64x | 3 |
| w4a16-rtn | 21.72 | +5.20 | 436 | 0.46x | 3 |
| w4a16-gptq | 18.18 | +1.66 | 436 | 0.46x | 11135 |

Cross-check: the saved checkpoints, reloaded from disk in bf16, give 21.718 (w4a16-rtn), 18.183 (w4a16-gptq) and
16.987 (w8a8-rtn). The w4a16-rtn re-run reproduced 21.719 exactly.

## Observations
- Same int4 format, same size: GPTQ removes ~68% of RTN's perplexity increase ((21.72 − 18.18) / 5.20).
- W8A8's cost is almost all activation quantization: +0.46 with int8 activations alone, +0.02 more from int8 weights.
- Size is far from the 4x/2x ideal because the 152k-token embedding (~136M of ~494M params) is not quantized.

## Validity notes
- Numbers before 24 Oct 2026 are invalid: weight fake-quant was silently skipped for offloaded modules after
  `oneshot` (compressed-tensors bug, reported as
  [vllm-project/compressed-tensors#985](https://github.com/vllm-project/compressed-tensors/issues/985)). Pre-fix results are kept in
  `results/pre-fix/` (w4a16-rtn 16.52 = baseline; w8a8-rtn 16.98 = activation quant only).
- Runs use compressed-tensors with the fix (branch `fix-offloaded-weight-qdq`, editable install).
- Earlier sizes were inflated (701/862 MiB) because unquantized tensors were saved in fp32; `quantize.py` now casts
  them to bf16 after saving. The GPTQ checkpoint was converted in place, not re-run (the fp32 copy is kept
  locally, not in this repo).
