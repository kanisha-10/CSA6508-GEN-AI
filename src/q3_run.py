"""Q3: direct vs zero-shot CoT vs few-shot CoT vs self-consistency (m=5). python src/q3_run.py --provider openai --model <id>"""
import sys, os, re, argparse, collections
sys.path.insert(0, os.path.dirname(__file__))
from common import LLM, read_csv, write_csv, set_seed
ap = argparse.ArgumentParser(); ap.add_argument("--provider", default="mock"); ap.add_argument("--model", default="mock"); a = ap.parse_args(); set_seed()
RULES = open("data/q3_rules.txt").read(); Q = read_csv("data/q3_questions_50.csv")
SYS = "You are a university fee and eligibility assistant. " + RULES
FEW = ("Example: Q: I have 80% attendance. May I write the exam? A: Rule 1 needs >= 75%. 80 >= 75, so yes. Final answer: yes\n"
       "Example: Q: Fee balance Rs 10000, paid after the due date. Total? A: Late fee = 5% of 10000 = 500. Total = 10000 + 500 = 10500. Final answer: 10500.00\n")
METH = {"direct": "Answer with only 'Final answer: <value>'.", "zs_cot": "Think step by step, then finish with 'Final answer: <value>'.", "fs_cot": FEW + "Now think step by step in the same style and finish with 'Final answer: <value>'."}
def final(t):
    m = re.findall(r"final answer:\s*([^\n]+)", t, re.I); s = (m[-1] if m else t).strip().lower().replace(",", "")
    n = re.findall(r"-?\d+(?:\.\d+)?", s); return ("yes" if "yes" in s else "no" if re.search(r"\bno\b", s) else (f"{float(n[-1]):.2f}" if n else ""))
rows = []
for temp in (0.0, 0.7):
    llm = LLM(a.provider, a.model, temperature=temp, max_tokens=400)
    for q in Q:
        for meth in ("direct", "zs_cot", "fs_cot", "self_consistency"):
            if meth == "self_consistency" and temp == 0.0: continue            # voting needs sampling
            m_runs = 5 if meth == "self_consistency" else 1; ans, itok, otok, texts = [], 0, 0, []
            for j in range(m_runs):
                t, pt, ct = llm.ask(SYS, f"Q: {q['question']}\n" + METH["fs_cot" if meth == "self_consistency" else meth], tag=f"{meth}|T={temp}"); ans.append(final(t)); itok += pt; otok += ct; texts.append(t)
            vote = collections.Counter(ans).most_common(1)[0][0]
            rows.append(dict(qid=q["qid"], steps=q["steps"], method=meth, temp=temp, pred=vote, gold=q["gold"], correct=int(vote == q["gold"]), in_tok=itok, out_tok=otok, paths_agree=ans.count(vote), reasoning_for_manual_check=texts[0][:300].replace("\n", " ")))
write_csv(f"results/{a.provider}/q3_raw.csv", rows)
import pandas as pd; d = pd.DataFrame(rows); print(d.groupby(["method", "temp", "steps"]).correct.mean().unstack())
