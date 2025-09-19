import pandas as pd
import numpy as np

# --- User-specific column mapping ---
# 1) status: 'unsafe' where 0=safe, 1=unsafe
# 2) lab type: 'lab type' (biology, chemistry, ee)
# 3) category: 'category' (ppe, sop, wo)
# 4) certainty: 'certainty' (exclude certainty==0 from TEST)
STATUS_COL = "unsafe"
LAB_COL = "lab type"
CAT_COL = "category"
CERTAINTY_COL = "certainty"

# --- Parameters ---
INPUT_PATH = "/mnt/data/Image Collection - good quality.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TRAIN_OUT = "/mnt/data/train_split.csv"
TEST_OUT = "/mnt/data/test_split.csv"

# --- Load ---
df = pd.read_csv(INPUT_PATH)

# --- Build stratification key (combined labels) ---
# Convert unsafe {0,1} -> {safe, unsafe} text for clearer strata
df["_status"] = df[STATUS_COL].apply(lambda x: "unsafe" if x == 1 else "safe")
df["_lab"] = df[LAB_COL].astype(str).str.strip().str.lower()
df["_cat"] = df[CAT_COL].astype(str).str.strip().str.lower()
df["_stratum"] = df["_status"] + " | " + df["_lab"] + " | " + df["_cat"]

# --- Stratified sampling with certainty==0 excluded from TEST ---
rng = np.random.default_rng(RANDOM_STATE)
test_indices = []

for stratum, group in df.groupby("_stratum"):
    # Only allow rows with certainty != 0 to go to the test set
    eligible_for_test = group[group[CERTAINTY_COL] != 0]
    n = len(group)
    # target test size per stratum (rounded)
    n_test = int(round(TEST_SIZE * n))
    # ensure it's feasible given the number of eligible rows
    n_test = max(1, min(n_test, len(eligible_for_test))) if n > 0 else 0

    if n_test > 0 and len(eligible_for_test) >= n_test:
        chosen = rng.choice(eligible_for_test.index, size=n_test, replace=False)
        test_indices.extend(chosen.tolist())

# --- Split & keep ALL original columns ---
test_mask = df.index.isin(test_indices)
train_df = df[~test_mask].drop(columns=["_status", "_lab", "_cat", "_stratum"])
test_df = df[test_mask].drop(columns=["_status", "_lab", "_cat", "_stratum"])

# --- Save ---
train_df.to_csv(TRAIN_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)

print(f"Wrote {TRAIN_OUT} and {TEST_OUT}")
