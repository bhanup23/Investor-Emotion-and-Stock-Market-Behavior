# ============================================================
# STEP 3: EMOTION EXTRACTION (SAFE, FIXED VERSION)
# ============================================================

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- SETTINGS ----------------
INPUT_DIR = r"E:/pre_market_messages_csv"
OUTPUT_DIR = r"E:/emotion_messages_csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
BATCH_SIZE = 32
MAX_LEN = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------

LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise"
]

print(f"Using device: {DEVICE}")

# ---------------- LOAD TOKENIZER ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---------------- LOAD MODEL (SAFE TENSORS) ----------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    trust_remote_code=False,
    use_safetensors=True   # <<< THIS FIXES THE ERROR
)

model.to(DEVICE)
model.eval()  # Freeze model

# ---------------- INFERENCE FUNCTION ----------------
def infer_emotions(texts):
    all_probs = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu())

    return torch.cat(all_probs).numpy()

# ---------------- PROCESS FILES ----------------
for file in os.listdir(INPUT_DIR):
    if not file.endswith(".csv"):
        continue

    print(f"\nProcessing {file}")

    df = pd.read_csv(
        os.path.join(INPUT_DIR, file),
        encoding="utf-8",
        low_memory=False
    )

    probs = infer_emotions(df["text"].astype(str).tolist())

    for i, label in enumerate(LABELS):
        df[f"p_{label}"] = probs[:, i]

    df.to_csv(
        os.path.join(OUTPUT_DIR, file),
        index=False
    )

    print(f"Saved → {file}")

print("\nSTEP 3 COMPLETED SUCCESSFULLY.")
