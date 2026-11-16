"""Q7 validation of the miss-path with a REAL model: does 1 retrieved example (config E) lose accuracy versus the fixed 5-shot prompt (config A)?
python src/q7_validate_api.py --provider openai --model <id>   (uses the same stream seed as q7_cache_redesign.py; first 300 queries)"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import q7_cache_redesign as q7; from common import LLM, write_csv, set_seed
ap = argparse.ArgumentParser(); ap.add_argument("--provider", default="mock"); ap.add_argument("--model", default="mock"); a = ap.parse_args(); set_seed(); llm = LLM(a.provider, a.model, 0.0, 60)
rows = []
for q, lab in q7.stream(300, q7.SEED)[:300]:
    for cfg in "AE":
        shots = q7.FIXED5 if cfg == "A" else [q7.EXPOOL[int((q7.EXM @ q7.VEC.transform([q]).T).toarray().ravel().argmax())]]
        text, pt, ct = llm.ask("", q7.prompt(shots, q), tag=f"q7|{cfg}")
        rows.append(dict(cfg=cfg, query=q, gold=lab, in_tok=pt, out_tok=ct, correct=int(lab in text or q7.gold_answer(lab).lower()[:25] in text.lower())))
write_csv(f"results/{a.provider}/q7_api_validation.csv", rows)
for cfg in "AE": r = [x for x in rows if x["cfg"] == cfg]; print(cfg, "acc", sum(x["correct"] for x in r) / len(r), "in_tok", sum(x["in_tok"] for x in r))
