import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.extract import load_segmented_dataset

# =========================
# CONFIG
# =========================

PROCESSED_ROOT = "data/processed"
TIME_WINDOWS_MS = [100, 200, 500, 800, 1000, 1200, 1500]
DEVICES = ["oppo", "phone", "um"]

FMIN = 0
FMAX = 8000
NBINS = 20

OUTPUT_CSV = "results/benchmark_results.csv"

# =========================
# MODELS
# =========================

MODELS = {
    "RF": RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=42),
    "GB": GradientBoostingRegressor(random_state=42),
    "SVR": SVR()
}

# =========================
# RUN
# =========================

all_results = []

for time_ms in TIME_WINDOWS_MS:
    for device in DEVICES:

        base_dir = os.path.join(
            PROCESSED_ROOT,
            f"processed_{time_ms}ms",
            device
        )

        if not os.path.exists(base_dir):
            print(f"[SKIP] {base_dir}")
            continue

        print(f"[LOAD] {device} | {time_ms} ms")

        X, y = load_segmented_dataset(
            base_dir=base_dir,
            fmin=FMIN,
            fmax=FMAX,
            nbins=NBINS
        )

        print(f"  Samples: {X.shape[0]} | Features: {X.shape[1]}")

        for model_name, model in MODELS.items():
            print(f"  → {model_name}")

            preds = cross_val_predict(model, X, y, cv=5, n_jobs=-1)

            all_results.append({
                "device": device,
                "time_ms": time_ms,
                "model": model_name,
                "MAE": mean_absolute_error(y, preds),
                "MSE": mean_squared_error(y, preds),
                "R2": r2_score(y, preds),
                "n_samples": len(y)
            })

# =========================
# SAVE
# =========================

df = pd.DataFrame(all_results)
os.makedirs("results", exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved → {OUTPUT_CSV}")
