"""Print a markdown table comparing every scheme in results/."""

import json
from pathlib import Path

from quantize import RESULTS_DIR, SCHEMES


def main():
    rows = [
        json.loads((RESULTS_DIR / f"{s}.json").read_text())
        for s in SCHEMES
        if (RESULTS_DIR / f"{s}.json").exists()
    ]
    base = next(r for r in rows if r["scheme"] == "baseline")

    print("| scheme | WikiText-2 PPL | Δ PPL | size (MB) | vs bf16 | quantize time (s) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        dppl = r["wikitext2_ppl"] - base["wikitext2_ppl"]
        ratio = r["size_mb"] / base["size_mb"]
        print(
            f"| {r['scheme']} | {r['wikitext2_ppl']:.2f} | {dppl:+.2f} "
            f"| {r['size_mb']:.0f} | {ratio:.2f}x | {r['quant_seconds']:.0f} |"
        )
    print()
    for r in rows:
        print(f"**{r['scheme']}**: {r['sample']}\n")


if __name__ == "__main__":
    main()
