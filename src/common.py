"""Shared helpers: seeds, token counting, LLM client with exponential back-off, API logging. NO API KEYS here: set env vars."""
import os, time, random, csv, math, json
SEED = 42
def set_seed(s=SEED):
    random.seed(s)
    try:
        import numpy as np; np.random.seed(s)
    except ImportError: pass

def count_tokens(text: str) -> int:
    """tiktoken cl100k_base if installed, else heuristic max(1.3*words, chars/4) (state which one you used)."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, math.ceil(max(1.3 * len(text.split()), len(text) / 4)))

LOG = "logs/api_log.csv"
def log_row(row: dict):
    os.makedirs("logs", exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new: w.writeheader()
        w.writerow(row)

class LLM:
    """provider: openai | gemini | hf | mock.  mock is only a dry-run to test the pipeline - never report its numbers."""
    def __init__(self, provider, model, temperature=0.0, max_tokens=300, seed=SEED):
        self.p, self.model, self.t, self.mt, self.seed = provider, model, temperature, max_tokens, seed
        if provider == "openai":
            from openai import OpenAI; self.c = OpenAI()          # reads OPENAI_API_KEY
        elif provider == "gemini":
            from google import genai; self.c = genai.Client()     # reads GEMINI_API_KEY
        elif provider == "hf":
            from transformers import pipeline; self.c = pipeline("text-generation", model=model)

    def _once(self, system, user):
        if self.p == "mock":
            rnd = random.Random(len(system) * 31 + len(user)); return rnd.choice(["A", "B", "C"]), count_tokens(system + user), 1
        if self.p == "openai":
            r = self.c.chat.completions.create(model=self.model, temperature=self.t, max_tokens=self.mt, seed=self.seed,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens
        if self.p == "gemini":
            from google.genai import types
            r = self.c.models.generate_content(model=self.model, contents=user, config=types.GenerateContentConfig(
                system_instruction=system, temperature=self.t, max_output_tokens=self.mt, seed=self.seed))
            u = r.usage_metadata; return r.text, u.prompt_token_count, u.candidates_token_count
        if self.p == "hf":
            prompt = system + "\n\n" + user
            out = self.c(prompt, max_new_tokens=self.mt, do_sample=self.t > 0, temperature=max(self.t, 1e-5), return_full_text=False)[0]["generated_text"]
            return out, count_tokens(prompt), count_tokens(out)

    def ask(self, system, user, tag=""):
        retries, t0 = 0, time.time()
        while True:
            try:
                text, pt, ct = self._once(system, user)
                log_row(dict(ts=time.strftime("%F %T"), provider=self.p, model=self.model, tag=tag, temp=self.t, prompt_tokens=pt,
                             completion_tokens=ct, latency_s=round(time.time() - t0, 3), retries=retries, error=""))
                return text, pt, ct
            except Exception as e:
                is429 = "429" in str(e) or getattr(e, "status_code", None) == 429 or "rate" in str(e).lower()
                if not is429 or retries >= 6:
                    log_row(dict(ts=time.strftime("%F %T"), provider=self.p, model=self.model, tag=tag, temp=self.t, prompt_tokens=0,
                                 completion_tokens=0, latency_s=round(time.time() - t0, 3), retries=retries, error=str(e)[:80]))
                    return "", 0, 0
                time.sleep(min(60, 2 ** retries) + random.random()); retries += 1   # exponential back-off + jitter

def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def read_csv(path):
    with open(path, encoding="utf-8") as f: return list(csv.DictReader(f))
