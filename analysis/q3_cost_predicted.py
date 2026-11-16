"""Q3 cost per additional correct answer - computed from PREDICTED accuracies and ASSUMED token counts/prices (replace with measured values after q3_run.py)."""
import sys; sys.path.insert(0, "src")
from common import write_csv
P_IN, P_OUT = 0.15, 0.60                       # USD per 1M tokens (assumption - cite the provider's price page on the run date)
acc = {"direct": [96, 92, 80, 62, 45], "zero-shot CoT": [94, 93, 88, 78, 66], "few-shot CoT": [96, 95, 92, 85, 76], "self-consistency m=5": [97, 96, 94, 88, 80]}
tok = {"direct": (150, 8), "zero-shot CoT": (160, 150), "few-shot CoT": (270, 160), "self-consistency m=5": (1350, 800)}   # (input, output) tokens per question - assumed
rows = []; base_c = base_a = None
for m, a in acc.items():
    mean = sum(a) / 5; cost = 1000 * (tok[m][0] * P_IN + tok[m][1] * P_OUT) / 1e6
    if base_c is None: base_c, base_a = cost, mean
    extra = (mean - base_a) * 10
    rows.append(dict(method=m, mean_acc_pct=round(mean, 1), in_tok=tok[m][0], out_tok=tok[m][1], out_tok_multiple=round(tok[m][1] / 8, 1), cost_usd_per_1000=round(cost, 4),
                     extra_correct_per_1000=round(extra, 1), usd_per_extra_correct=(round((cost - base_c) / extra, 5) if extra > 0 else "-")))
write_csv("results/q3_cost_per_extra_correct_PREDICTED.csv", rows); [print(r) for r in rows]
