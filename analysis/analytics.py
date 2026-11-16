"""Analytic (no-LLM) results: LoRA parameter counts, self-consistency maths, Wilson CIs, Q2 token budget, weight sensitivity. Run: python analysis/analytics.py"""
import sys, os, math, itertools
sys.path.insert(0, "src")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import beta as B, binom
from common import write_csv, count_tokens, read_csv
# ---- Q5: LoRA parameter counts. Per adapted matrix W (d_out x d_in): LoRA = r*(d_in+d_out); full = d_in*d_out
def lora_total(L, d_in_q, d_out_q, d_in_v, d_out_v, r): return L * r * ((d_in_q + d_out_q) + (d_in_v + d_out_v))
def full_total(L, dq_in, dq_out, dv_in, dv_out): return L * (dq_in * dq_out + dv_in * dv_out)
MODELS = {"Generic 4096-d, 32 layers (square q,v)": (32, 4096, 4096, 4096, 4096), "Qwen2.5-0.5B-like: L=24, d=896, v_proj 896->128 (GQA) [verify config.json]": (24, 896, 896, 896, 128)}
rows = []
for name, (L, qi, qo, vi, vo) in MODELS.items():
    full = full_total(L, qi, qo, vi, vo)
    for r in [2, 4, 8, 16, 64]:
        t = lora_total(L, qi, qo, vi, vo, r); rows.append(dict(model=name, r=r, lora_params=t, full_qv_params=full, pct_of_qv=round(100 * t / full, 3)))
write_csv("results/q5_lora_param_table.csv", rows)
assert 8 * (4096 + 4096) == 65536 and abs(100 * 65536 / 16777216 - 0.390625) < 1e-9     # sanity check from the brief
# ---- Q3: majority of m=5, independent vs correlated (beta-binomial, intraclass corr rho)
def maj_indep(p, m=5): return sum(binom.pmf(j, m, p) for j in range(m // 2 + 1, m + 1))
def maj_corr(p, rho, m=5):
    """Beta-binomial: per-question success prob q ~ Beta(a,b), mean p, intraclass correlation rho = 1/(a+b+1); exact, no grid."""
    if rho <= 1e-9: return maj_indep(p, m)
    from scipy.stats import betabinom
    s = 1 / rho - 1; return float(sum(betabinom.pmf(j, m, p * s, (1 - p) * s) for j in range(m // 2 + 1, m + 1)))
rows = []
for p in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    rows.append(dict(p_single=p, indep=round(maj_indep(p), 4), rho_0_3=round(maj_corr(p, .3), 4), rho_0_6=round(maj_corr(p, .6), 4)))
write_csv("results/q3_majority_vote_table.csv", rows)
ps = np.linspace(0.3, 0.98, 40); plt.figure(figsize=(5.2, 3.6))
plt.plot(ps, ps, "k:", label="single path"); plt.plot(ps, [maj_indep(p) for p in ps], label="independent (rho=0)")
plt.plot(ps, [maj_corr(p, .3) for p in ps], label="rho=0.3"); plt.plot(ps, [maj_corr(p, .6) for p in ps], label="rho=0.6")
plt.xlabel("single-path accuracy p"); plt.ylabel("majority-of-5 accuracy"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("figures/q3_majority_vote.png", dpi=160); plt.close()
# ---- Q6: Wilson 95% CI at n=100, and CI for a difference of two proportions (unpaired, conservative)
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h
rows = [dict(correct=k, n=100, acc=k / 100, ci_low=round(wilson(k, 100)[0], 3), ci_high=round(wilson(k, 100)[1], 3)) for k in [60, 70, 80, 85, 90, 95]]
write_csv("results/q6_wilson_ci_n100.csv", rows)
# ---- Q2: token budget with stated assumptions (heuristic tokenizer unless tiktoken installed)
SYS = ("You are the hostel and campus help-desk assistant of a university. Classify the student's message into exactly one label from this list: "
       "hostel_timing, fee_due, bonafide, library, hallticket, bus_pass, wifi_reset, mess_menu, condonation, id_card, Hostel-Maintenance, Hostel-Mess, Academic-Timetable, "
       "Exam-Results, Fees-Finance, IT-Portal-WiFi. Reply with the label only. If unsure, reply with the closest label. Never invent labels.")
EX = "Message: The fan in room 214 of B block is not working since yesterday.\nLabel: Hostel-Maintenance\n"
Q = "Message: Hall ticket for the CSA6507 exam is not generated yet.\nLabel:"
ts, te, tq = count_tokens(SYS), count_tokens(EX), count_tokens(Q); ans = 300
kmax = (8192 - ans - ts - tq) // te
rows = [dict(option=n, k=k, system_tokens=ts, tokens_per_example=te, query_tokens=tq, input_tokens=ts + k * te + tq) for n, k in [("zero-shot", 0), ("one-shot", 1), ("3-shot", 3), ("few-shot", 5)]]
rows.append(dict(option=f"largest k that fits (8192-{ans} reserved)", k=kmax, system_tokens=ts, tokens_per_example=te, query_tokens=tq, input_tokens=ts + kmax * te + tq))
write_csv("results/q2_token_budget.csv", rows)
# cost per 1,000 queries; PRICES ARE AN ASSUMPTION - replace with the provider's price page value and cite it
P_IN, P_OUT, OUT = 0.15, 0.60, 6      # USD per 1M tokens (assumed), output tokens per query (label only)
for r in rows: r["cost_usd_per_1000_queries"] = round(1000 * (r["input_tokens"] * P_IN + OUT * P_OUT) / 1e6, 4)
write_csv("results/q2_token_budget.csv", rows)
# ---- Q6: weight sensitivity (machinery; run on PREDICTED scores here, re-run on measured scores)
W = dict(accuracy=.30, cost=.15, latency=.10, json_reliability=.15, privacy=.15, context_rate_limits=.15)
S = {"OpenAI-class": dict(accuracy=.85, cost=.6, latency=.7, json_reliability=.95, privacy=.5, context_rate_limits=.8),
     "Gemini-class": dict(accuracy=.83, cost=.8, latency=.8, json_reliability=.9, privacy=.5, context_rate_limits=.85),
     "HF local small": dict(accuracy=.6, cost=.95, latency=.5, json_reliability=.6, privacy=1.0, context_rate_limits=.5)}
def score(w): t = sum(w.values()); return {m: sum(w[k] / t * v[k] for k in w) for m, v in S.items()}
base = score(W); rows = [dict(scenario="base", **{m: round(v, 3) for m, v in base.items()}, winner=max(base, key=base.get))]
changes = 0
for k in W:
    for f in (0.8, 1.2):
        w = dict(W); w[k] *= f; s = score(w); win = max(s, key=s.get); changes += win != max(base, key=base.get)
        rows.append(dict(scenario=f"{k} x{f}", **{m: round(v, 3) for m, v in s.items()}, winner=win))
write_csv("results/q6_sensitivity_on_predicted_scores.csv", rows); print("winner changes in", changes, "of", len(W) * 2, "scenarios (predicted scores)")
print("Q2 token budget:", ts, te, tq, "kmax", kmax, "| tokenizer:", "tiktoken" if "tiktoken" in sys.modules else "heuristic")
# ---- graph: LoRA params vs r
plt.figure(figsize=(5.2, 3.6)); rs = [2, 4, 8, 16, 64]
for name, (L, qi, qo, vi, vo) in MODELS.items(): plt.loglog(rs, [lora_total(L, qi, qo, vi, vo, r) for r in rs], "o-", label=name.split(" [")[0][:34])
plt.xlabel("rank r"); plt.ylabel("trainable LoRA parameters (q,v)"); plt.grid(alpha=.3, which="both"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig("figures/q5_lora_params.png", dpi=160); plt.close()
