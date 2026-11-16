# Predictions (written BEFORE any experiment was run)
Commit this file to git before running any run_*.py script; the commit timestamp is the evidence.

## Q1
- Task (b): zero-shot ~70%, 5-shot ~82%; fine-tuned small model 20 ex ~55%, 100 ex ~85%, 200 ex ~90%.
- Fine-tuning overtakes 5-shot prompting at ~60-100 labelled examples (about 10-17 per class).
- Limitation exposed by task (e): frozen knowledge (regulation published yesterday) -> not fixed by fine-tuning; needs retrieval.

## Q2
- FAQ look-ups: k=0 94%, k=1 95%, k=3 96%, k=5 96% -> best accuracy per token at k=0.
- Ambiguous complaints: k=0 62%, k=1 68%, k=3 74%, k=5 76% -> gain flattens after k=3; order changes accuracy by 2-5 points.
- 70/30 mix: best accuracy per token at k=1 (or k=0 if routed).
- Edge set (typos / Tamil-English): k=0 55%, k=5 70% (examples help most here).

## Q3 (accuracy %, steps 1..5)
- Direct: 96 92 80 62 45 | ZS-CoT: 94 93 88 78 66 | FS-CoT: 96 95 92 85 76 | SC(m=5, FS-CoT, T=0.7): 97 96 94 88 80
- Output-token multiple vs direct: ZS-CoT ~15x, FS-CoT ~20x, SC ~5 x FS-CoT (~100x).
- CoT starts paying for itself at 3 steps.

## Q4
- Valid JSON: plain 80%, schema-in-prompt 94%, native function calling 99%.
- Correct function: 85 / 92 / 96 %. Correct args: 75 / 88 / 93 %.
- Higher temperature lowers all three by 2-6 points. Model-level injections succeeding: 4/12, 2/12, 1/12; with application-side session check: 0 executed.

## Q5
- Accuracy r=4 < r=8 ~ r=16 (differences within noise at 500 examples). Peak GPU memory almost flat in r. Forgetting on 50 general prompts: small, grows slightly with r.

## Q6
- Predicted ranking (accuracy): hosted API model A ~ hosted API model B > small local HF model. Privacy/cost: local HF best. Overall weighted winner: a hosted API model with local HF as fallback.

## Q7
- Chosen: few-shot (k=5) prompt on a 90%-repeated FAQ stream.
- Prediction: exact+semantic(FAQ-only) cache cuts LLM calls by >=80% and total tokens by >=80% at N=3000 queries; wrong cached answers <=1% of queries; saving grows with N (sub-linear tokens).
