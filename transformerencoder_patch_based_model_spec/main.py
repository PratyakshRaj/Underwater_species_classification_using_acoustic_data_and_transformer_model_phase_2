import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import json
from sklearn.metrics import confusion_matrix
import os 
import math
from utils import create_dataframe
from dataset import SpectrogramDataset, collate_fn
from model_3 import ConvPatchAudioTransformer
from train import train_one_epoch, evaluate
import time

DATA_DIR = "features/spectrogram"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 25
N_SPLITS = 5

PARAMS = [1,2,3,4]

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
    
    for nlayers in PARAMS:
        
        patch_key = f"num_layers_{nlayers}"
        results[patch_key] = {}
        
        patch_time=8
        max_time = 5000
        max_patches = math.ceil(max_time / patch_time)

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            
            if fold>0:
                break

            fold_key = f"fold_{fold+1}"
            results[patch_key][fold_key] = {} 
            
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

            max_time = 5000
            max_patches = math.ceil(max_time / patch_time)

            model = ConvPatchAudioTransformer(
                num_classes=len(class_to_idx),
                d_model=64,
                num_layers=nlayers,
                dim_feedforward=2 * 64,
                patch_freq=128,
                stride_freq=128,
                patch_time=patch_time,
                stride_time=patch_time,
                max_patches=max_patches
            ).to(DEVICE)

            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"num_layers={nlayers}, Params={param_count:,}")

            results[patch_key][fold_key]["params"] = param_count
            
            
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
                if DEVICE == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()

                val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
                    model, val_loader, criterion, DEVICE
                )

                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()
                if DEVICE == "cuda":
                    memory = torch.cuda.max_memory_allocated() / 1024**2
                    print(f"Max GPU Memory: {memory:.2f} MB")
                
                print(
                    f"Epoch {epoch+1} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )
                
                total_time = end_time - start_time
                total_samples = len(val_dataset)

                print(f"Evaluation Time: {total_time:.2f} sec")
                print(f"Total Validation Samples: {total_samples}")
                print(f"Samples/sec: {total_samples / total_time:.2f}")

                results[patch_key][fold_key][f"epoch_{epoch+1}"] = {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1
                }

            os.makedirs("fold_CM", exist_ok=True)
            cm = confusion_matrix(val_labels, val_preds)
            np.save(f"fold_CM/num_layers_{nlayers}_fold_{fold+1}_confusion.npy", cm)
            np.savetxt(
                f"fold_CM/num_layers_{nlayers}_fold_{fold+1}_confusion.csv",cm,delimiter=",")
            

        # Save JSON after all folds
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)

        print("\nFinal CV Accuracy:", np.mean(fold_results))


if __name__ == "__main__":
    main()