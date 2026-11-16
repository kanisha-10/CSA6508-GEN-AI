"""Q6: identical prompts to every candidate, back-off on HTTP 429, full logging. Exact-match scoring (no LLM judge needed).
python src/q6_run.py --candidates openai:<model-id> gemini:<model-id> hf:<model-id>     (keys via OPENAI_API_KEY / GEMINI_API_KEY; never commit keys)"""
import sys, os, argparse, re, math, time
sys.path.insert(0, os.path.dirname(__file__))
from common import LLM, read_csv, write_csv, set_seed
ap = argparse.ArgumentParser(); ap.add_argument("--candidates", nargs="+", default=["mock:mock"]); a = ap.parse_args(); set_seed()
gold = read_csv("data/q6_gold_100.csv"); SYS = "Answer using only the context. Reply with the shortest possible answer (a label, a number or a short phrase) and nothing else."
norm = lambda s: re.sub(r"[^a-z0-9. %]", "", s.lower().replace(",", "")).strip()
def correct(pred, ref): return int(norm(ref) in norm(pred)) if ref.replace(".", "").replace("%", "").isdigit() is False else int(norm(pred).rstrip("0").rstrip(".") == norm(ref).rstrip("0").rstrip(".") or norm(ref) in norm(pred))
def wilson(k, n, z=1.96): p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return round(c - h, 3), round(c + h, 3)
rows, summ = [], []
for cand in a.candidates:
    prov, model = cand.split(":", 1); llm = LLM(prov, model, temperature=0.0, max_tokens=40); lat = []
    for g in gold:
        t0 = time.time(); text, pt, ct = llm.ask(SYS, f"Context: {g['context']}\nQuestion: {g['question']}\nAnswer:", tag=cand); lat.append(time.time() - t0)
        rows.append(dict(candidate=cand, qid=g["qid"], kind=g["kind"], pred=text.strip(), reference=g["reference"], correct=correct(text, g["reference"]), in_tok=pt, out_tok=ct, latency_s=round(lat[-1], 3)))
    c = [r for r in rows if r["candidate"] == cand]; k = sum(r["correct"] for r in c); lat.sort()
    summ.append(dict(candidate=cand, correct=k, n=len(c), acc=k / len(c), ci95=wilson(k, len(c)), p95_latency_s=round(lat[int(.95 * len(lat)) - 1], 3), in_tok=sum(r["in_tok"] for r in c), out_tok=sum(r["out_tok"] for r in c)))
write_csv("results/q6_raw.csv", rows); write_csv("results/q6_summary.csv", summ); print(*summ, sep="\n")
