import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import json
from sklearn.metrics import confusion_matrix
import os 

from utils import create_dataframe
from dataset import SpectrogramDataset, collate_fn
from model_2 import ConvPatchAudioTransformer
from train import train_one_epoch, evaluate

DATA_DIR = "features/spectrogram"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 25
N_SPLITS = 5


def main():

    df, class_to_idx = create_dataframe(DATA_DIR)
    labels = df["label"].values

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42
    )

    fold_results = []
    results = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
         
        fold_key = f"fold_{fold+1}"
        results[fold_key] = {} 
        
        print(f"\n========== Fold {fold+1} ==========")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_dataset = SpectrogramDataset(train_df)
        val_dataset = SpectrogramDataset(val_df)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )

        model = ConvPatchAudioTransformer(
            num_classes=len(class_to_idx)
        ).to(DEVICE)

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )

        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )

        for epoch in range(EPOCHS):

            train_loss, train_acc, train_f1 = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE
            )

            val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
                model, val_loader, criterion, DEVICE
            )

            print(
                f"Epoch {epoch+1} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            results[fold_key][f"epoch_{epoch+1}"] = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1": val_f1
            }

        os.makedirs("fold_CM", exist_ok=True)
        cm = confusion_matrix(val_labels, val_preds)
        np.save(f"fold_CM/fold_{fold}_confusion.npy", cm)
        np.savetxt(f"fold_CM/fold_{fold}_confusion.csv", cm, delimiter=",")
        

    # Save JSON after all folds
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nFinal CV Accuracy:", np.mean(fold_results))


if __name__ == "__main__":
    main()