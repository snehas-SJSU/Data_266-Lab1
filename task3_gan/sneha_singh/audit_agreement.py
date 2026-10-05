"""Human audit summary for Part 3: mean scores per rater and inter-rater agreement.

Reads outputs/human_audit/audit_30.csv (two raters, 30 fixed photo->Monet images) and prints,
for style, content, and artifacts: each rater's mean (1-5), percent agreement, and Cohen's kappa.

Usage (from the repo root):
  python task3_gan/sneha_singh/audit_agreement.py
"""

import csv
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

SHEET = Path(__file__).resolve().parent / "outputs" / "human_audit" / "audit_30.csv"
RATERS = ("sneha", "ritika")
CRITERIA = ("style", "content", "artifacts", "mem")


def main():
    rows = list(csv.DictReader(line for line in SHEET.open() if not line.startswith("#")))
    for crit in CRITERIA:
        a = [r[f"rater_{RATERS[0]}_{crit}"].strip() for r in rows]
        b = [r[f"rater_{RATERS[1]}_{crit}"].strip() for r in rows]
        pairs = [(x, y) for x, y in zip(a, b) if x and y]
        if not pairs:
            print(f"{crit}: not rated yet")
            continue
        x, y = zip(*pairs)
        agree = sum(i == j for i, j in pairs) / len(pairs)
        if len(set(x + y)) == 1:   # every answer the same: kappa is undefined
            line = f"{crit}: n={len(pairs)}  agreement={agree:.1%}  kappa=undefined (all '{x[0]}')"
        else:
            line = f"{crit}: n={len(pairs)}  agreement={agree:.1%}  kappa={cohen_kappa_score(x, y):.3f}"
        if crit != "mem":
            line += f"  mean_{RATERS[0]}={sum(map(int, x)) / len(x):.2f}  mean_{RATERS[1]}={sum(map(int, y)) / len(y):.2f}"
        print(line)


if __name__ == "__main__":
    main()
