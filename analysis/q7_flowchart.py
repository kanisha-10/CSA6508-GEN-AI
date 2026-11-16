import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(7.2, 4.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
def box(x, y, w, h, t, c="#EAF1FB"): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#2F5597", lw=1)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5)
def arrow(a, b, t=""): ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1)); ax.text((a[0] + b[0]) / 2 + .1, (a[1] + b[1]) / 2, t, fontsize=7, color="#C00000")
box(0.2, 5.7, 2.4, 0.9, "Student query"); box(3.2, 5.7, 2.6, 0.9, "Personal / function-call\nintent? (fee, timetable)", "#FFF2CC"); box(6.6, 5.7, 3.1, 0.9, "Function-calling path\n(no cache, app validates)", "#FCE4D6")
box(0.2, 4.1, 2.8, 0.9, "normalise + exact cache\nlookup"); box(3.6, 4.1, 3.0, 0.9, "semantic lookup (TF-IDF cosine)\nKB-answer entries only"); box(7.2, 4.1, 2.5, 0.9, "serve cached answer\n(0 LLM calls) -> return", "#E2F0D9")
box(0.2, 2.4, 3.0, 0.9, "retrieve 1 nearest\nlabelled example (config E)"); box(3.8, 2.4, 2.8, 0.9, "LLM call: system prompt\n+ example + query"); box(7.2, 2.4, 2.5, 0.9, "validate answer,\nstore (version, TTL)", "#E2F0D9")
box(3.8, 0.7, 2.8, 0.9, "return answer to student", "#E2F0D9")
arrow((2.6, 6.15), (3.2, 6.15)); arrow((5.8, 6.15), (6.6, 6.15), "yes"); arrow((4.5, 5.7), (1.6, 5.0), "no"); arrow((3.0, 4.55), (3.6, 4.55), "miss"); arrow((6.6, 4.55), (7.2, 4.55), ">= tau")
arrow((5.1, 4.1), (1.8, 3.3), "< tau"); arrow((3.2, 2.85), (3.8, 2.85)); arrow((6.6, 2.85), (7.2, 2.85)); arrow((8.4, 2.4), (6.6, 1.3))
arrow((5.1, 2.4), (5.1, 1.6)); plt.tight_layout(); plt.savefig("figures/q7_flowchart.png", dpi=170)
