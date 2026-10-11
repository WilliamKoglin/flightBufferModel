"""
calibrate_buffer.py — turn point predictions into a P90 safety buffer.

Idea: the model predicts a TYPICAL delay, but travelers need to survive a
BAD day. So we measure how wrong the model tends to be (residuals), and add
a margin big enough that 90% of real flights come in UNDER our estimate.

Flow:
  load artifacts -> predict on validation set -> residuals = actual - pred
  -> P90 of residuals = safety margin M -> save M into metadata.json
"""
import json
import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data_cleaning import load_and_clean   # adjust to your function name
from features import prepare_features, CATEGORICAL_COLS, NUMERICAL_COLS, TARGET
from dataset import FlightDelayDataset
from model import DelayNet
from torch.utils.data import DataLoader

ARTIFACTS_DIR = "artifacts"
PERCENTILE = 90   # P90 = "90% of flights come in under our buffered estimate"
BATCH_SIZE = 1024


def main():
    # 1) load metadata + encoders (same ones training produced)
    with open(f"{ARTIFACTS_DIR}/metadata.json") as f:
        meta = json.load(f)
    with open(f"{ARTIFACTS_DIR}/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    scaler = meta["scaler"]

    # 2) rebuild + load the trained model
    model = DelayNet(meta["vocab_sizes"], meta["num_numeric"], meta["emb_dim"])
    model.load_state_dict(torch.load(f"{ARTIFACTS_DIR}/model.pt", map_location="cpu"))
    model.eval()

    # 3) recreate the SAME validation split used in training (same seed!)
    df = load_and_clean()
    _, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # reuse the TRAINED encoders/scaler — do NOT rebuild them
    val_df, _, _ = prepare_features(val_df, encoders=encoders, scaler=scaler)
    val_ds = FlightDelayDataset(val_df)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 4) collect predictions and actuals across the whole val set
    preds, actuals = [], []
    with torch.no_grad():
        for cat, num, target in val_loader:
            pred = model(cat, num)
            preds.append(pred.numpy())
            actuals.append(target.numpy())

    preds = np.concatenate(preds).ravel()
    actuals = np.concatenate(actuals).ravel()

    # 5) residuals: how much WORSE reality was than our prediction
    residuals = actuals - preds

    # 6) the safety margin: P90 of residuals
    margin = float(np.percentile(residuals, PERCENTILE))

    # --- diagnostics so you understand what you got ---
    print(f"residual mean:   {residuals.mean():7.2f} min")
    print(f"residual median: {np.median(residuals):7.2f} min")
    print(f"P90 margin (M):  {margin:7.2f} min")

    # sanity: what fraction of flights land UNDER pred + margin?
    covered = np.mean(actuals <= preds + margin)
    print(f"coverage with margin: {covered*100:5.1f}%  (target ~{PERCENTILE}%)")

    # 7) save the margin back into metadata.json
    meta["buffer_margin"] = margin
    meta["buffer_percentile"] = PERCENTILE
    with open(f"{ARTIFACTS_DIR}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n Saved buffer_margin={margin:.2f} to metadata.json")


if __name__ == "__main__":
    main()