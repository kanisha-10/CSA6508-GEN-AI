"""Q7 prototype: ORIGINAL (5-shot prompt on every query) vs REDESIGN (reuse of answers already generated).
The LLM is replaced by an ORACLE STUB that returns the gold answer on a cache miss, so this experiment counts calls/tokens exactly and
measures the ONLY new error source of the redesign: wrong answers served from the cache. Real-LLM accuracy on misses must be validated with q7_validate_api.py.
Run: python src/q7_cache_redesign.py"""
import sys, os, re, random, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import count_tokens, read_csv, write_csv, SEED
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ANS = {"hostel_timing": "Hostel gate closes at 10 pm; entry after that needs warden permission.", "fee_due": "The semester fee is due on 31 January; a 5% late fee applies after that.",
 "bonafide": "Apply for a bonafide certificate at the office counter or portal; it is issued in 2 working days.", "library": "The library is open 8 am to 8 pm on working days.",
 "hallticket": "Download the hall ticket from the student portal 3 days before the exams.", "bus_pass": "Apply for a bus pass at the transport office (9 am to 4 pm); validity is 6 months.",
 "wifi_reset": "Reset the Wi-Fi password through the IT cell portal using your register number.", "mess_menu": "The weekly mess menu is displayed on the mess notice board and the portal.",
 "condonation": "Submit the condonation form with a medical certificate to your HoD; the fee applies.", "id_card": "Apply at the office for a duplicate ID card; the replacement fee is Rs 200."}
SYS = ("You are the hostel and campus help-desk assistant of a university. Classify the student's message into exactly one label from this list: hostel_timing, fee_due, bonafide, library, hallticket, bus_pass, wifi_reset, mess_menu, condonation, id_card, "
       "Hostel-Maintenance, Hostel-Mess, Academic-Timetable, Exam-Results, Fees-Finance, IT-Portal-WiFi. For the first ten labels also give the standard answer; for the last six reply with the label only.")
faq = read_csv("data/q2_set_faq.csv"); tick = read_csv("data/q1_tickets_300.csv")
FAQ_BY = {}; [FAQ_BY.setdefault(r["label"], []).append(r["text"]) for r in faq]; INTENTS = list(FAQ_BY)
def gold_answer(label): return ANS.get(label, label)
EXPOOL = [(r["text"], r["label"]) for r in faq][::5] + [(r["text"], r["label"]) for r in tick if r["split"] == "train"][:40]   # labelled example store
FIXED5 = [EXPOOL[0], EXPOOL[1], EXPOOL[4], EXPOOL[10], EXPOOL[20]]
def ex_text(t, l): return f"Message: {t}\nAnswer: {gold_answer(l)}\n"
def prompt(shots, q): return SYS + "\n" + "".join(ex_text(t, l) for t, l in shots) + f"Message: {q}\nAnswer:"
def norm(q): return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", q.lower())).strip()
def noise(r, s):
    x = r.random()
    if x < .15: return s.upper() if r.random() < .5 else s.lower()
    if x < .30 and len(s) > 8: i = r.randint(1, len(s) - 3); return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if x < .40: return s.rstrip("?.") + " ??"
    return s
def stream(n, seed, faq_share=.9, zipf=1.1):
    r = random.Random(seed); w = [1 / (i + 1) ** zipf for i in range(len(INTENTS))]; out = []
    for _ in range(n):
        if r.random() < faq_share: it = r.choices(INTENTS, w)[0]; out.append((noise(r, r.choice(FAQ_BY[it])), it))
        else: t = r.choice(tick); out.append((noise(r, t["text"]) + f" (ref {r.randint(100000, 999999)})", t["label"]))   # complaints are novel: unique reference number
    return out
VEC = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit([r["text"] for r in faq] + [t["text"] for t in tick if t["split"] == "train"])
EXM = VEC.transform([t for t, _ in EXPOOL])
class Cache:
    def __init__(s, tau, mode): s.tau, s.mode, s.exact, s.keys, s.ans, s.faq, s.M = tau, mode, {}, [], [], [], None
    def get(s, q):
        k = norm(q)
        if k in s.exact: return s.exact[k], "exact"
        if s.mode in ("sem_all", "sem_faq") and s.keys:
            sims = (s.M @ VEC.transform([q]).T).toarray().ravel()
            ok = np.ones_like(sims, bool) if s.mode == "sem_all" else np.array(s.faq)        # sem_faq: only entries whose answer is a reusable KB answer
            sims = np.where(ok, sims, -1); i = int(sims.argmax())
            if sims[i] >= s.tau: return s.ans[i], "semantic"
        return None, "miss"
    def put(s, q, a, is_faq):
        s.exact[norm(q)] = a
        if s.mode in ("sem_all", "sem_faq"):
            s.keys.append(q); s.ans.append(a); s.faq.append(is_faq)
            import scipy.sparse as sp; v = VEC.transform([q]); s.M = v if s.M is None else sp.vstack([s.M, v])
def run(cfg, qs, tau=0.8):
    c = Cache(tau, {"A": "none", "B": "exact", "C": "sem_all", "D": "sem_faq", "E": "sem_faq"}[cfg]); st = dict(calls=0, in_tok=0, out_tok=0, hits_exact=0, hits_sem=0, wrong=0)
    for q, lab in qs:
        g = gold_answer(lab)
        if cfg != "A":
            a, how = c.get(q)
            if a is not None:
                st["hits_exact" if how == "exact" else "hits_sem"] += 1; st["wrong"] += (a != g); continue
        shots = FIXED5
        if cfg == "E":                                                   # dynamic k=1: nearest labelled example instead of 5 fixed ones
            j = int((EXM @ VEC.transform([q]).T).toarray().ravel().argmax()); shots = [EXPOOL[j]]
        p = prompt(shots, q); st["calls"] += 1; st["in_tok"] += count_tokens(p); st["out_tok"] += count_tokens(g)   # oracle stub answer
        if cfg != "A": c.put(q, g, lab in ANS)
    n = len(qs); st.update(n=n, total_tok=st["in_tok"] + st["out_tok"], wrong_pct=round(100 * st["wrong"] / n, 3), calls_pct=round(100 * st["calls"] / n, 2)); return st
if __name__ == "__main__":
    N = 3000; rows = []; names = {"A": "A original: 5-shot every query", "B": "B exact (normalised) cache", "C": "C exact + semantic cache (all entries)", "D": "D exact + semantic (KB-answer entries only)", "E": "E = D + 1 retrieved example on misses"}
    for cfg in "ABCDE":
        runs = [run(cfg, stream(N, SEED + s)) for s in range(5)]
        m = lambda k: float(np.mean([r[k] for r in runs])); sd = lambda k: float(np.std([r[k] for r in runs]))
        rows.append(dict(config=names[cfg], tau=0.8 if cfg in "CDE" else "-", queries=N, llm_calls=round(m("calls"), 1), input_tokens=round(m("in_tok")), output_tokens=round(m("out_tok")), total_tokens=round(m("total_tok")),
                         total_tokens_sd=round(sd("total_tok")), exact_hits=round(m("hits_exact"), 1), semantic_hits=round(m("hits_sem"), 1), wrong_served=round(m("wrong"), 2), wrong_pct=round(m("wrong_pct"), 3)))
    base = rows[0]["total_tokens"]
    for r in rows: r["token_saving_pct"] = round(100 * (1 - r["total_tokens"] / base), 1); r["call_saving_pct"] = round(100 * (1 - r["llm_calls"] / rows[0]["llm_calls"]), 1)
    write_csv("results/q7_results.csv", rows)
    sw = []
    for tau in [.5, .6, .7, .8, .9, .95]:
        for cfg in "CD":
            rs = [run(cfg, stream(N, SEED + s), tau) for s in range(5)]; sw.append(dict(config=cfg, tau=tau, llm_calls=round(np.mean([r["calls"] for r in rs]), 1), total_tokens=round(np.mean([r["total_tok"] for r in rs])), wrong_pct=round(np.mean([r["wrong_pct"] for r in rs]), 3)))
    write_csv("results/q7_tau_sweep.csv", sw)
    sc = []
    for n in [250, 500, 1000, 2000, 3000, 5000, 10000]:
        for cfg in "AD": r = run(cfg, stream(n, SEED)); sc.append(dict(config=cfg, queries=n, llm_calls=r["calls"], total_tokens=r["total_tok"], tokens_per_query=round(r["total_tok"] / n, 1), wrong_pct=r["wrong_pct"]))
    write_csv("results/q7_scaling.csv", sc)
    plt.figure(figsize=(5.2, 3.6))
    for cfg, lab in [("A", "Original (5-shot)"), ("D", "Redesign D")]:
        d = [r for r in sc if r["config"] == cfg]; plt.plot([r["queries"] for r in d], [r["total_tokens"] for r in d], "o-", label=lab)
    plt.xlabel("queries per day"); plt.ylabel("total tokens (in+out)"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("figures/q7_scaling.png", dpi=160); plt.close()
    plt.figure(figsize=(5.2, 3.6))
    for cfg, lab in [("C", "C: semantic cache, all entries"), ("D", "D: semantic cache, KB entries only")]:
        d = [r for r in sw if r["config"] == cfg]; plt.plot([r["tau"] for r in d], [r["wrong_pct"] for r in d], "o-", label=lab)
    plt.xlabel("similarity threshold tau"); plt.ylabel("wrong answers served (% of queries)"); plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("figures/q7_tau_wrong.png", dpi=160); plt.close()
    rs = []
    for share in (0.3, 0.5, 0.7, 0.9):
        runs = [run("D", stream(N, SEED + s, faq_share=share)) for s in range(5)]; ra = run("A", stream(N, SEED, faq_share=share))
        rs.append(dict(faq_share=share, calls_A=ra["calls"], calls_D=round(np.mean([r["calls"] for r in runs]), 1), tokens_A=ra["total_tok"], tokens_D=round(np.mean([r["total_tok"] for r in runs])),
                       token_saving_pct=round(100 * (1 - np.mean([r["total_tok"] for r in runs]) / ra["total_tok"]), 1), wrong_pct=round(np.mean([r["wrong_pct"] for r in runs]), 3)))
    write_csv("results/q7_repeat_share_sensitivity.csv", rs)
    qs = stream(N, SEED); sysT = count_tokens(SYS); exT = sum(count_tokens(ex_text(t, l)) for t, l in FIXED5); qT = float(np.mean([count_tokens(x) for x, _ in qs])) + count_tokens("Message: \nAnswer:")
    write_csv("results/q7_token_breakdown.csv", [dict(system_prompt=sysT, five_examples=exT, query_with_wrapper=round(qT, 1), static_share_pct=round(100 * (sysT + exT) / (sysT + exT + qT), 1),
        unique_normalised_query_pct=round(100 * len({norm(x) for x, _ in qs}) / N, 1))])
    for r in rows + rs: print(r)
