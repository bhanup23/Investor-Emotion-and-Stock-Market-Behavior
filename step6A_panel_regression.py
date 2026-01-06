# ============================================================
# STEP 6A: ENHANCED PANEL REGRESSION (ADAPTED FOR YOUR DATA)
# Replicates Table 5 from Vamossy (2024)
# ============================================================

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from scipy import stats

# ---------------- FILE PATHS ----------------
EMOTION_FILE = r"C:\Users\Vaibhav singh\Desktop\socialmedia\emotions.csv"
RETURNS_FILE = r"C:\Users\Vaibhav singh\Desktop\socialmedia\price_returns.csv"
# --------------------------------------------

# ---------------- LOAD DATA ----------------
print("Loading data...")
emo = pd.read_csv(EMOTION_FILE)
rets = pd.read_csv(RETURNS_FILE)

print(f"Raw emotion data shape: {emo.shape}")
print(f"Raw returns data shape: {rets.shape}")
print(f"Emotion columns: {emo.columns.tolist()}")
print(f"Returns columns: {rets.columns.tolist()}")

# Check for and remove any header rows that snuck into the data
print("\nCleaning data...")
emo = emo[emo["trading_date"] != "trading_date"]
rets = rets[rets["trading_date"] != "trading_date"]

# Ensure correct types and handle timezone info
emo["trading_date"] = pd.to_datetime(emo["trading_date"], format="mixed", utc=True, errors="coerce").dt.tz_localize(None)
rets["trading_date"] = pd.to_datetime(rets["trading_date"], format="mixed", utc=True, errors="coerce").dt.tz_localize(None)

# Drop any rows where date parsing failed
emo = emo.dropna(subset=["trading_date"])
rets = rets.dropna(subset=["trading_date"])

print(f"After cleaning - Emotion data: {emo.shape}")
print(f"After cleaning - Returns data: {rets.shape}")

# Normalize to date only
emo["trading_date"] = pd.to_datetime(emo["trading_date"].dt.date)
rets["trading_date"] = pd.to_datetime(rets["trading_date"].dt.date)

# Standardize ticker column
emo["ticker"] = emo["ticker"].astype(str).str.upper().str.strip()
rets["Ticker"] = rets["Ticker"].astype(str).str.upper().str.strip()
rets = rets.rename(columns={"Ticker": "ticker"})

# Convert numeric columns to proper types
rets["Open"] = pd.to_numeric(rets["Open"], errors="coerce")
rets["Close"] = pd.to_numeric(rets["Close"], errors="coerce")
rets = rets.dropna(subset=["Open", "Close"])

emotion_cols = ["p_anger", "p_disgust", "p_fear", "p_joy", "p_neutral", "p_sadness", "p_surprise"]
for col in emotion_cols:
    emo[col] = pd.to_numeric(emo[col], errors="coerce")
emo = emo.dropna(subset=emotion_cols)

print(f"Emotion data: {len(emo)} observations")
print(f"Returns data: {len(rets)} observations")
print(f"Emotion tickers: {emo['ticker'].nunique()}")
print(f"Returns tickers: {rets['ticker'].nunique()}")

# ---------------- CALCULATE RETURNS FROM PRICE DATA ----------------
print("\n" + "="*60)
print("CALCULATING RETURNS")
print("="*60)

# Sort by ticker and date
rets = rets.sort_values(["ticker", "trading_date"])

# Calculate open-to-close return (intraday return)
rets["ret_oc"] = (rets["Close"] - rets["Open"]) / rets["Open"]

# Calculate close-to-open return (overnight return)
rets["prev_close"] = rets.groupby("ticker")["Close"].shift(1)
rets["ret_co"] = (rets["Open"] - rets["prev_close"]) / rets["prev_close"]

# Drop rows without previous close (first observation per ticker)
rets = rets.dropna(subset=["prev_close"])

print(f"Calculated returns for {len(rets)} observations")
print(f"Return range: {rets['trading_date'].min()} to {rets['trading_date'].max()}")

# ---------------- TEMPORAL ALIGNMENT ----------------
print("\n" + "="*60)
print("TEMPORAL STRUCTURE")
print("="*60)
print("   Emotions (4PM t-1 to 9:29AM t) → Returns (9:30AM t to 4PM t)")

# ---------------- HANDLE NON-TRADING DAYS ----------------
print("\n" + "="*60)
print("HANDLING NON-TRADING DAYS")
print("="*60)

# Get all unique trading dates from returns data
trading_days = pd.Series(rets["trading_date"].unique()).sort_values().reset_index(drop=True)
trading_days_set = set(trading_days)

def find_next_trading_day(date, max_gap=5):
    """Find the next trading day within max_gap days."""
    for i in range(1, max_gap + 1):
        next_date = date + pd.Timedelta(days=i)
        if next_date in trading_days_set:
            return next_date
    return None

# Create mapping for emotion dates to trading dates
emo["original_date"] = emo["trading_date"]
emo["days_to_next_trade"] = 0

non_trading_mask = ~emo["trading_date"].isin(trading_days_set)
non_trading_count = non_trading_mask.sum()

print(f"Emotions on non-trading days: {non_trading_count:,}")

if non_trading_count > 0:
    print("Mapping to next available trading day (max 5 days gap)...")
    
    emo.loc[non_trading_mask, "trading_date"] = emo.loc[non_trading_mask, "original_date"].apply(
        lambda x: find_next_trading_day(x, max_gap=5)
    )
    
    emo.loc[non_trading_mask, "days_to_next_trade"] = (
        emo.loc[non_trading_mask, "trading_date"] - emo.loc[non_trading_mask, "original_date"]
    ).dt.days
    
    unmapped = emo["trading_date"].isna().sum()
    if unmapped > 0:
        print(f"⚠️  Dropping {unmapped:,} emotions >5 days from next trading day")
        emo = emo.dropna(subset=["trading_date"])
    
    gap_stats = emo[emo["days_to_next_trade"] > 0].groupby("days_to_next_trade").size()
    if len(gap_stats) > 0:
        print("\nGap distribution (days to next trading day):")
        for days, count in gap_stats.items():
            print(f"  {int(days)} days: {count:,} observations")

print(f"Final emotion observations: {len(emo):,}")

# ---------------- MERGE DATASETS ----------------
print("\n" + "="*60)
print("MERGING EMOTIONS WITH RETURNS")
print("="*60)

df = emo.merge(
    rets[["ticker", "trading_date", "Open", "Close", "ret_oc", "ret_co"]],
    on=["ticker", "trading_date"],
    how="inner"
)
print(f"Merged data: {len(df):,} observations")

mapped_count = (df["days_to_next_trade"] > 0).sum()
if mapped_count > 0:
    print(f"Including {mapped_count:,} observations from non-trading days")

# ---------------- CONSTRUCT CONTROL VARIABLES ----------------
print("\n" + "="*60)
print("CONSTRUCTING CONTROL VARIABLES")
print("="*60)

df = df.sort_values(["ticker", "trading_date"])

# Lagged open-close return (t-1)
df["ret_oc_lag1"] = df.groupby("ticker")["ret_oc"].shift(1)

# Past 20-day return
df["ret_past20"] = (
    df.groupby("ticker")["ret_oc"]
    .rolling(window=20, min_periods=10)
    .sum()
    .reset_index(level=0, drop=True)
)

# Past 183-day volatility (half-year)
df["volatility_183"] = (
    df.groupby("ticker")["ret_oc"]
    .rolling(window=183, min_periods=90)
    .std()
    .reset_index(level=0, drop=True)
)

# ---------------- WINSORIZE RETURNS ----------------
def winsorize(series, limits=(0.001, 0.001)):
    """Winsorize at 0.1% and 99.9%"""
    lower = series.quantile(limits[0])
    upper = series.quantile(1 - limits[1])
    return series.clip(lower, upper)

df["ret_oc"] = winsorize(df["ret_oc"])
df["ret_oc_lag1"] = winsorize(df["ret_oc_lag1"])
df["ret_co"] = winsorize(df["ret_co"])
df["ret_past20"] = winsorize(df["ret_past20"])

# ---------------- DROP MISSING VALUES ----------------
df = df.dropna(subset=[
    "ret_oc", "ret_oc_lag1", "ret_co", 
    "ret_past20", "volatility_183"
])

print(f"Final estimation sample: {len(df):,} observations")
print(f"Number of firms: {df['ticker'].nunique()}")
print(f"Date range: {df['trading_date'].min()} to {df['trading_date'].max()}")

# ---------------- DESCRIPTIVE STATISTICS ----------------
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)

desc_vars = ["ret_oc", "ret_oc_lag1", "ret_co", "ret_past20", "volatility_183",
             "p_joy", "p_sadness", "p_fear", "p_anger", "p_disgust", 
             "p_surprise", "p_neutral"]

desc_stats = df[desc_vars].describe().T
desc_stats["within_std"] = df.groupby("ticker")[desc_vars].std().mean()

print(desc_stats[["mean", "std", "within_std", "min", "max"]].round(6))

# ---------------- SET PANEL STRUCTURE ----------------
df = df.set_index(["ticker", "trading_date"])

# ---------------- MODEL SPECIFICATIONS ----------------

# Dependent variable
Y = df["ret_oc"]

# Emotion variables (neutral is baseline, omitted)
emotion_vars = ["p_joy", "p_sadness", "p_fear", "p_anger", "p_disgust", "p_surprise"]

# Control variables
control_vars = ["ret_oc_lag1", "ret_co", "ret_past20", "volatility_183"]

# ---------------- MODEL 1: CONTROLS ONLY ----------------
print("\n" + "="*60)
print("MODEL 1: CONTROLS ONLY (BASELINE)")
print("="*60)

X1 = df[control_vars]

model1 = PanelOLS(
    Y,
    X1,
    entity_effects=True,
    time_effects=True
)

results1 = model1.fit(
    cov_type="clustered",
    cluster_entity=True
)

print(results1.summary)

# ---------------- MODEL 2: EMOTIONS + CONTROLS ----------------
print("\n" + "="*60)
print("MODEL 2: EMOTIONS + CONTROLS (MAIN SPECIFICATION)")
print("="*60)

X2 = df[emotion_vars + control_vars]

model2 = PanelOLS(
    Y,
    X2,
    entity_effects=True,
    time_effects=True
)

results2 = model2.fit(
    cov_type="clustered",
    cluster_entity=True
)

print(results2.summary)

# ---------------- STANDARDIZED EFFECTS ----------------
print("\n" + "="*60)
print("STANDARDIZED EFFECTS (β × σ_X / σ_Y)")
print("="*60)

y_std = df.groupby(level="ticker")["ret_oc"].apply(lambda x: x - x.mean()).std()

standardized_effects = {}
for var in emotion_vars:
    beta = results2.params[var]
    x_std = df.groupby(level="ticker")[var].apply(lambda x: x - x.mean()).std()
    std_effect = (beta * x_std / y_std) * 100
    standardized_effects[var] = std_effect
    print(f"{var:15s}: {std_effect:6.2f}% of σ(ret_oc)")

# ---------------- R² IMPROVEMENT ----------------
r2_baseline = results1.rsquared
r2_full = results2.rsquared
r2_improvement = (r2_full - r2_baseline) * 10000

print(f"\nR² Baseline:    {r2_baseline:.6f}")
print(f"R² Full Model:  {r2_full:.6f}")
print(f"R² Improvement: {r2_improvement:.2f} basis points")

# ---------------- EXPORT RESULTS ----------------
results_table = pd.DataFrame({
    "Variable": emotion_vars + control_vars,
    "Coefficient": results2.params[emotion_vars + control_vars],
    "Std_Error": results2.std_errors[emotion_vars + control_vars],
    "T_Stat": results2.tstats[emotion_vars + control_vars],
    "P_Value": results2.pvalues[emotion_vars + control_vars]
})

def add_stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    else:
        return ""

results_table["Significance"] = results_table["P_Value"].apply(add_stars)

print("\n" + "="*60)
print("REGRESSION RESULTS TABLE")
print("="*60)
print(results_table.to_string(index=False))

results_table.to_csv(r"C:\Users\Vaibhav singh\Desktop\socialmedia\regression_results_table5.csv", index=False)
print(f"\nResults saved to: C:\\Users\\Vaibhav singh\\Desktop\\socialmedia\\regression_results_table5.csv")

# ---------------- KEY FINDINGS SUMMARY ----------------
print("\n" + "="*60)
print("KEY FINDINGS (REPLICATING TABLE 5)")
print("="*60)

print(f"""
1. Sample Size: {len(df):,} firm-day observations
2. Number of Firms: {df.reset_index()['ticker'].nunique()}
3. Time Period: {df.reset_index()['trading_date'].min().date()} to {df.reset_index()['trading_date'].max().date()}

4. Main Results:
   - Joy coefficient:     {results2.params['p_joy']:.6f} (t = {results2.tstats['p_joy']:.2f})
   - Sadness coefficient: {results2.params['p_sadness']:.6f} (t = {results2.tstats['p_sadness']:.2f})
   - Fear coefficient:    {results2.params['p_fear']:.6f} (t = {results2.tstats['p_fear']:.2f})

5. Model Fit:
   - R² improvement: {r2_improvement:.2f} basis points
   - Within R²: {results2.rsquared_within:.4f}
   - Overall R²: {results2.rsquared:.4f}

6. Interpretation:
   - A 1 std. dev. increase in Joy → {standardized_effects['p_joy']:.2f}% std. dev. change in returns
   - A 1 std. dev. increase in Sadness → {standardized_effects['p_sadness']:.2f}% std. dev. change in returns
   - A 1 std. dev. increase in Fear → {standardized_effects['p_fear']:.2f}% std. dev. change in returns
""")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)