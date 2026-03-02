import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from data_loading import AudioFeatureDataset
from model_building import TransformerModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
import os
from torch.nn.utils.rnn import pad_sequence
import torch
from collections import Counter
import math



def collate_fn(batch):
    
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    specs, labels = zip(*batch)

    # ---- Spectrogram ----
    specs = [s.squeeze(0) if s.dim() == 3 else s for s in specs]  # make sure [mel, T]
    specs = [s.permute(1, 0) for s in specs]                      # → [T, mel]
    spec_lengths = [s.shape[0] for s in specs]
    specs_padded = pad_sequence(specs, batch_first=True)          # [B, T_max, mel]
    specs_padded = specs_padded.permute(0, 2, 1).unsqueeze(1)     # [B, 1, mel, T_max]
    spec_mask = torch.arange(specs_padded.shape[-1])[None, :] < torch.tensor(spec_lengths)[:, None]

    
    labels = torch.tensor(labels)

    return specs_padded, labels, spec_mask




class trainer:
    def __init__(self,path_spec_folder,label_map_path,n_mels,d_model, num_classes,n_heads,n_layers, fold,train_subset=None, val_subset=None):
        with open(label_map_path, "r") as f:
            self.label_map = json.load(f)

        # If subsets are provided (K-Fold), wrap them in DataLoaders
        if train_subset is not None and val_subset is not None:
            self.train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, collate_fn=collate_fn)
            self.val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, collate_fn=collate_fn)    
        else:
            # Otherwise just use the full dataset (your original code)
            full_dataset = AudioFeatureDataset(path_spec_folder, self.label_map)
            self.train_loader = DataLoader(full_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
            self.val_loader = None  # no validation in this case
        
 


        # path_spectrogram_folder="features/spectrogram" 
        #self.train_dataset = AudioFeatureDataset(path_spec_folder, self.label_map)
        #self.train_loader = DataLoader(self.train_dataset, batch_size=4, shuffle=True,collate_fn=collate_fn)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # d_model=128, num_classes=25
        self.model = TransformerModel(n_mels=n_mels,d_model=d_model, num_classes=num_classes,n_heads=n_heads,n_layers=n_layers).to(self.device)
        self.fold=fold
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=3e-4)
        self.train_losses = []
        self.val_losses = []  
        self.best_val_loss = float("inf")
        self.patience = 20
        self.patience_counter = 0
        self.train_res=[]
        self.val_res=[]



    def train(self,epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_idx,batch in enumerate(self.train_loader):
                if batch is None:     # ← skip empty batches
                    continue
                spec, labels, spec_mask = batch
                spec, labels = spec.to(self.device), labels.to(self.device)
                spec_mask = spec_mask.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(spec,spec_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if batch_idx % 5 == 0:  # print every 5 batches
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")


            print(f"Epoch {epoch+1}: Loss = {total_loss/len(self.train_loader):.4f}")
            
            if self.val_loader is not None:
                val_loss = self.validation_loss()
                print(f"Epoch {epoch+1}: Val loss = {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print("🛑 Early stopping triggered")
                        break
            if self.fold==0:
                self.train_losses.append(round(total_loss/len(self.train_loader),5))
                self.val_losses.append(val_loss)


             
            # ---- Evaluate both train and validation ----
        self.train_res.append(self.evaluate_loader(self.train_loader, name="Train",base_dir="check_train"))
        if self.val_loader:
            self.val_res.append(self.evaluate_loader(self.val_loader, name="Val",base_dir="check_train"))            
    
        return self.model

    def evaluate_loader(self,loader,name="Val",base_dir="train_test_all"):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                spec, labels, spec_mask = batch
                spec, labels = spec.to(self.device), labels.to(self.device)
                spec_mask= spec_mask.to(self.device)

                outputs = self.model(spec,spec_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
              
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        avg_loss = total_loss / len(loader)

        print(f"{name} Loss = {avg_loss:.4f}, {name} Acc = {acc:.4f}, {name} F1 = {f1:.4f}")

        results = {
                "name": name,
                "loss": avg_loss,
                "accuracy": acc,
                "f1": f1
            }

        if name == "Val":  # confusion matrix only for validation
          
            cm = confusion_matrix(all_labels, all_preds).tolist()
            results["confusion_matrix"] = cm
            print(f"[Fold {self.fold}] Validation Confusion Matrix:\n", cm)
        
        return results
        
    

    def validation_loss(self):
        self.model.eval()
        total_loss = 0
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue
                spec, labels, spec_mask = batch
                spec, labels = spec.to(self.device), labels.to(self.device)
                spec_mask = spec_mask.to(self.device)

                outputs = self.model(spec,spec_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
              
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        #f1 = f1_score(all_labels, all_preds, average="macro")
                

        return total_loss/len(self.val_loader)      




