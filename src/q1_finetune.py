"""Q1 fine-tuning arms: small pre-trained encoder fine-tuned on 20/100/200 training tickets. Needs: pip install torch transformers. Seeds fixed."""
import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))
import torch; from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common import read_csv, write_csv, SEED
MODEL = "distilbert-base-uncased"; LAB = ["Hostel-Maintenance", "Hostel-Mess", "Academic-Timetable", "Exam-Results", "Fees-Finance", "IT-Portal-WiFi"]
t = read_csv("data/q1_tickets_300.csv"); train = [r for r in t if r["split"] == "train"]; test = [r for r in t if r["split"] == "test"]; rows = []
for n in (20, 100, 200):
    random.seed(SEED); torch.manual_seed(SEED); sub = random.Random(SEED).sample(train, n)
    tok = AutoTokenizer.from_pretrained(MODEL); m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=6); opt = torch.optim.AdamW(m.parameters(), lr=5e-5)
    enc = lambda rs: tok([r["text"] for r in rs], padding=True, truncation=True, max_length=64, return_tensors="pt"); y = lambda rs: torch.tensor([LAB.index(r["label"]) for r in rs])
    m.train()
    for ep in range(8):
        random.Random(SEED + ep).shuffle(sub)
        for i in range(0, n, 16):
            b = sub[i:i + 16]; loss = m(**enc(b), labels=y(b)).loss; loss.backward(); opt.step(); opt.zero_grad()
    m.eval()
    with torch.no_grad(): pred = m(**enc(test)).logits.argmax(-1)
    acc = (pred == y(test)).float().mean().item(); trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    rows.append(dict(arm=f"fine-tune n={n}", labelled_examples=n, test_acc=round(acc, 4), trainable_params=trainable)); print(rows[-1])
write_csv("results/q1_finetune.csv", rows)
