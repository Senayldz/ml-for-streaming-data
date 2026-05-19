import joblib
import sys
sys.path.append(".")
import pandas as pd
import numpy as np
from pathlib import Path
LGBM_PATH = Path("artifacts/lgbm_detector.joblib")
SCALER_PATH = Path("artifacts/scaler.joblib")

print("Loading artifacts...")
scaler = joblib.load(SCALER_PATH)
model = joblib.load(LGBM_PATH)

mini_csv = Path("dataset/merged_mini.csv")
print("Reading header...")
df = pd.read_csv(mini_csv, nrows=5)

# Same logic as dashboard
feat_cols = [c.strip() for c in df.columns if c.strip() not in {"Timestamp", "Normal/Attack"} and df[c].dtype != object]
print(f"Total features: {len(feat_cols)}")

idx_mv = [i for i, c in enumerate(feat_cols) if "MV" in c.upper()]
idx_p = [i for i, c in enumerate(feat_cols) if "P" in c.upper() and c.upper().startswith("P")]
idx_lit = [i for i, c in enumerate(feat_cols) if "LIT" in c.upper()]

print("MV idx:", idx_mv)
print("P idx:", idx_p)
print("LIT idx:", idx_lit)

# Let's test the model prediction
X_raw = df[feat_cols].fillna(0).values.astype(np.float32)
X_scaled = scaler.transform(X_raw).astype(np.float32)

preds_normal = model.predict(X_scaled)
print("Normal preds:", preds_normal)

# Inject attack
X_attack = X_scaled.copy()
if idx_mv:
    X_attack[:, idx_mv[0]] = 15.0

preds_attack = model.predict(X_attack)
print("Attack preds (MV=15):", preds_attack)

if idx_mv:
    X_attack[:, idx_mv] = 50.0  # try extreme on all MV
print("Attack preds (All MV=50):", model.predict(X_attack))

if idx_lit:
    X_attack[:, idx_lit] = -50.0
print("Attack preds (All LIT=-50):", model.predict(X_attack))

