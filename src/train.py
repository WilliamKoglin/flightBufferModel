"""
train.py — THE CONDUCTOR. Run this file to train and export the model.

Flow:
  data_cleaning -> features -> dataset -> model -> training loop -> artifacts/
Produces: artifacts/model.pt, artifacts/encoders.pkl, artifacts/metadata.json
"""
import json
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data_cleaning import load_and_clean      # <-- adjust to your function name
from features import prepare_features, vocab_sizes, CATEGORICAL_COLS, NUMERICAL_COLS, TARGET
from dataset import FlightDelayDataset
from model import DelayNet

# ---- config ----
ARTIFACTS_DIR = "artifacts"
EPOCHS = 5
BATCH_SIZE = 256
LR = 1e-3
EMB_DIM = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    

    # 1) DATA ---------------------------------------------------------------
    df = load_and_clean()  # your data_cleaning output (the columns you showed)

   

    # split BEFORE building encoders/scaler so stats come only from train data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 2) FEATURES -----------------------------------------------------------
    # build encoders/scaler on TRAIN, then REUSE them on VAL (consistency!)
    train_df, encoders, scaler = prepare_features(train_df)
    val_df, _, _ = prepare_features(val_df, encoders=encoders, scaler=scaler)

    # 3) DATASETS / LOADERS -------------------------------------------------
    train_ds = FlightDelayDataset(train_df)
    val_ds = FlightDelayDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 4) MODEL --------------------------------------------------------------
    vsizes = vocab_sizes(encoders)
    model = DelayNet(vsizes, num_numeric=len(NUMERICAL_COLS), emb_dim=EMB_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()  # MAE — in minutes, easy to interpret

    # 5) TRAIN LOOP ---------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for cat, num, target in train_loader:
            cat, num, target = cat.to(DEVICE), num.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            pred = model(cat, num)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target)
        train_loss /= len(train_ds)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cat, num, target in val_loader:
                cat, num, target = cat.to(DEVICE), num.to(DEVICE), target.to(DEVICE)
                pred = model(cat, num)
                val_loss += loss_fn(pred, target).item() * len(target)
        val_loss /= len(val_ds)

        # are any feature columns suspiciously correlated with the target?
        print(f"epoch {epoch:2d} | train MAE {train_loss:6.2f} min | val MAE {val_loss:6.2f} min")

    # 6) EXPORT THE 3 ARTIFACTS --------------------------------------------
    torch.save(model.state_dict(), os.path.join(ARTIFACTS_DIR, "model.pt"))

    with open(os.path.join(ARTIFACTS_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    metadata = {
        "cat_cols": CATEGORICAL_COLS,
        "num_cols": NUMERICAL_COLS,
        "target": TARGET,
        "vocab_sizes": vsizes,
        "emb_dim": EMB_DIM,
        "num_numeric": len(NUMERICAL_COLS),
        "scaler": scaler,
    }
    with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n Exported model.pt, encoders.pkl, metadata.json to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()