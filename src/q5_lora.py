"""Q5: LoRA (PEFT) on a <=1B open causal LM, r in {4,8,16}, seeds fixed. Needs GPU + pip install torch transformers peft.
python src/q5_lora.py --model Qwen/Qwen2.5-0.5B   (verify the model card/licence; any <=1B causal LM works)"""
import sys, os, argparse, random, time
sys.path.insert(0, os.path.dirname(__file__))
import torch; from transformers import AutoTokenizer, AutoModelForCausalLM; from peft import LoraConfig, get_peft_model
from common import read_csv, write_csv, SEED
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B"); ap.add_argument("--epochs", type=int, default=3); a = ap.parse_args()
LAB = ["Hostel-Maintenance", "Hostel-Mess", "Academic-Timetable", "Exam-Results", "Fees-Finance", "IT-Portal-WiFi"]
tick = read_csv("data/q5_tickets_600.csv"); gen = read_csv("data/q5_general_50.csv")
train = [r for r in tick if r["split"] == "train"]; test = [r for r in tick if r["split"] == "test"]      # 500 train / 100 test
tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token; dev = "cuda" if torch.cuda.is_available() else "cpu"
P = lambda t: f"Complaint: {t}\nLabel:"
def gen_answer(m, prompt, n=8):
    ids = tok(prompt, return_tensors="pt").to(dev); out = m.generate(**ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id); return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
def general_acc(m): return sum(r["reference"].lower() in gen_answer(m, f"Q: {r['question']}\nA:", 10).lower() for r in gen) / len(gen)
rows = []
for r_ in (4, 8, 16):
    torch.manual_seed(SEED); random.seed(SEED); torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    g_before = general_acc(base)
    m = get_peft_model(base, LoraConfig(r=r_, lora_alpha=2 * r_, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad); tot = sum(p.numel() for p in m.parameters()); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    t0 = time.time(); m.train()
    for ep in range(a.epochs):
        random.Random(SEED + ep).shuffle(train)
        for r in train:
            ids = tok(P(r["text"]) + " " + r["label"] + tok.eos_token, return_tensors="pt").to(dev); lab = ids.input_ids.clone(); lab[:, :len(tok(P(r["text"])).input_ids)] = -100
            m(**ids, labels=lab).loss.backward(); opt.step(); opt.zero_grad()
    m.eval(); acc = sum(next((l for l in LAB if l.lower() in gen_answer(m, P(r["text"]), 10).lower()), "") == r["label"] for r in test) / len(test)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else float("nan")
    rows.append(dict(r=r_, trainable_params=tr, total_params=tot, trainable_pct=round(100 * tr / tot, 4), peak_gpu_gb=round(peak, 2), test_acc=round(acc, 4), general_acc_before=round(g_before, 3), general_acc_after=round(general_acc(m), 3), train_s=round(time.time() - t0)))
    print(rows[-1]); del m, base
write_csv("results/q5_lora_measured.csv", rows)
