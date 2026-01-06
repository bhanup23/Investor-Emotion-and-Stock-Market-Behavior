# ============================================================
# STEP 4: AGGREGATE MESSAGE-LEVEL EMOTIONS TO FIRM-DAY SIGNALS
# (CORRECTED & HARDENED VERSION)
# ============================================================

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------------- SETTINGS ----------------
INPUT_DIR = r"E:/emotion_messages_csv"
OUTPUT_DIR = r"E:/firm_day_emotions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMOTION_COLS = [
    "p_anger",
    "p_disgust",
    "p_fear",
    "p_joy",
    "p_neutral",
    "p_sadness",
    "p_surprise"
]
# -----------------------------------------


def aggregate_file(path):
    df = pd.read_csv(path, low_memory=False)

    # --------------------------------------------------------
    # 1. Normalize trading_date (remove timezone safely)
    # --------------------------------------------------------
    df["trading_date"] = pd.to_datetime(
        df["trading_date"],
        utc=True,
        errors="coerce"
    ).dt.date

    df = df.dropna(subset=["trading_date"])

    # --------------------------------------------------------
    # 2. Clean followers and compute influence weights
    # --------------------------------------------------------
    df["followers"] = (
        df["followers"]
        .fillna(0)
        .astype(float)
        .clip(lower=0)
    )

    # Paper-specified influence weight
    df["weight"] = np.log1p(df["followers"])

    # --------------------------------------------------------
    # 3. Apply weights to emotion probabilities
    # --------------------------------------------------------
    for col in EMOTION_COLS:
        df[col] = df[col] * df["weight"]

    # --------------------------------------------------------
    # 4. Aggregate by (ticker, trading_date)
    # --------------------------------------------------------
    grouped = df.groupby(["ticker", "trading_date"])

    emotion_sum = grouped[EMOTION_COLS].sum()
    weight_sum = grouped["weight"].sum()

    # Avoid division by zero
    weight_sum = weight_sum.replace(0, np.nan)

    for col in EMOTION_COLS:
        emotion_sum[col] = emotion_sum[col] / weight_sum

    emotion_sum = emotion_sum.reset_index()

    return emotion_sum


# ---------------- RUN STEP 4 ----------------
all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
all_results = []

for file in tqdm(all_files):
    print(f"Aggregating {file}")
    path = os.path.join(INPUT_DIR, file)
    agg_df = aggregate_file(path)
    all_results.append(agg_df)

final_df = pd.concat(all_results, ignore_index=True)

# --------------------------------------------------------
# 5. Final cleanup
# --------------------------------------------------------
final_df = final_df.dropna(subset=EMOTION_COLS)

final_path = os.path.join(OUTPUT_DIR, "firm_day_emotions.csv")
final_df.to_csv(final_path, index=False)

print("\nSTEP 4 COMPLETED SUCCESSFULLY.")
print(f"Saved → {final_path}")
