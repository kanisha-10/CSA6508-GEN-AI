"""Q1 (prompting arms) and Q2 (k-shot study). Usage: python src/q1_q2_run.py --provider openai --model <exact-model-id> [--study q1|q2]
Writes raw per-query CSVs to results/<provider>/ . Use --provider mock only to dry-run the pipeline."""
import sys, os, argparse, random
sys.path.insert(0, os.path.dirname(__file__))
from common import LLM, read_csv, write_csv, set_seed, count_tokens, SEED
ap = argparse.ArgumentParser(); ap.add_argument("--provider", default="mock"); ap.add_argument("--model", default="mock"); ap.add_argument("--study", default="q1"); a = ap.parse_args(); set_seed()
out = f"results/{a.provider}"; llm = LLM(a.provider, a.model, temperature=0.0, max_tokens=30)
LAB = ["Hostel-Maintenance", "Hostel-Mess", "Academic-Timetable", "Exam-Results", "Fees-Finance", "IT-Portal-WiFi"]
SYS = "Classify the complaint into exactly one label from: " + ", ".join(LAB) + ". Reply with the label only."
def shots_text(ex): return "".join(f"Complaint: {e['text']}\nLabel: {e['label']}\n\n" for e in ex)
def evaluate(rows, ex, tag, labels_ok=LAB):
    res = []
    for r in rows:
        text, pt, ct = llm.ask(SYS, shots_text(ex) + f"Complaint: {r['text']}\nLabel:", tag=tag)
        pred = next((l for l in labels_ok if l.lower() in text.lower()), ""); res.append(dict(qid=r.get("id", r.get("qid")), tag=tag, gold=r["label"], pred=pred, correct=int(pred == r["label"]), in_tok=pt, out_tok=ct))
    return res
if a.study == "q1":
    t = read_csv("data/q1_tickets_300.csv"); train = [r for r in t if r["split"] == "train"]; test = [r for r in t if r["split"] == "test"]
    rnd = random.Random(SEED); five = []
    for l in LAB: five += rnd.sample([r for r in train if r["label"] == l], 1)
    rows = evaluate(test, [], "zero-shot") + evaluate(test, five[:5], "5-shot")      # 5 examples, one per label (first five labels)
    write_csv(f"{out}/q1_prompting_raw.csv", rows)
    for tag in ("zero-shot", "5-shot"):
        r = [x for x in rows if x["tag"] == tag]; print(tag, "acc", sum(x["correct"] for x in r) / len(r), "mean in-tokens", sum(x["in_tok"] for x in r) / len(r))
    print("Fine-tuned arms: run  python src/q1_finetune.py  (needs torch + transformers)")
else:
    ex_pool = read_csv("data/q1_tickets_300.csv"); ex_pool = [r for r in ex_pool if r["split"] == "train"]; rows = []
    for sname in ("faq", "ambiguous", "edge"):
        qs = read_csv(f"data/q2_set_{sname}.csv")
        for order in (0, 1):
            rnd = random.Random(SEED + order); pool = ex_pool[:]; rnd.shuffle(pool)
            for k in (0, 1, 3, 5):
                res = evaluate(qs, pool[:k], f"{sname}|k={k}|order={order}")
                for x in res: x.update(set=sname, k=k, order=order)
                rows += res; print(sname, k, order, sum(x["correct"] for x in res) / len(res))
    write_csv(f"{out}/q2_raw.csv", rows)
