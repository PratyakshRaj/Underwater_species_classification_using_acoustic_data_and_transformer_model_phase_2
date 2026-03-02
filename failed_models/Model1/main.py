import torch
from model_training import trainer
import sklearn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from model_training import collate_fn
from model_building import TransformerModel
from sklearn.metrics import accuracy_score, confusion_matrix
from data_loading import AudioFeatureDataset
import json
import os

with open("label_map_aug.json", "r") as f:
    label_map = json.load(f)


# # Number of folds
# sk_folds = 5

# # Prepare KFold splitter (shuffling recommended)
# skf = StratifiedKFold(n_splits=sk_folds, shuffle=True, random_state=42)

        
# # path_spectrogram_folder="features/spectrogram" path_mfcc_folder="features/mfcc"
# dataset = AudioFeatureDataset("features/spectrogram","features/mfcc", label_map)

# valid_indices = []
# labels = []
# for i in range(len(dataset)):
#     item = dataset[i]
#     if item is None:
#         continue
#     _, _, label = item
#     valid_indices.append(i)
#     labels.append(label)

# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()

# for fold, (train_idx, val_idx) in enumerate(skf.split(valid_indices,labels)):
#         print(f"\n========== Fold {fold+1}/{sk_folds} ==========")
        
#         train_subset = Subset(dataset, [valid_indices[i] for i in train_idx])
#         val_subset = Subset(dataset, [valid_indices[i] for i in val_idx])
        
#         train_ins = trainer("features/spectrogram","features/mfcc","label_map.json",36,512,25,fold+1,train_subset=train_subset, val_subset=val_subset)

#         trained_model=train_ins.train(epochs=20)

#         # model_path = f"check_model.pth"
#         # torch.save(trained_model.state_dict(), model_path)
#         # print(f"Model saved: {model_path}")

## Training on full dataset- first comment out bove code
# train_full_ins=trainer("features/spectrogram","features/mfcc","label_map.json",36,512,25,0,train_subset=None,val_subset=None)
# trained_model_full= train_full_ins.train(epochs=25)
# final_model_path="fully_trained_model.pth"
# torch.save(trained_model_full.state_dict(),final_model_path)
# print("fully trained model saved")    

train_dataset = AudioFeatureDataset("features/spectrogram", "features/mfcc", label_map)
test_dataset = AudioFeatureDataset("features/test_spectrogram", "features/test_mfcc", label_map)

for x, y in [(2,2),(2,4),(4,2),(4,4),(8,4)]: 
    train_ins=trainer("features/spectrogram","features/mfcc","label_map_aug.json",36,72,25,n_heads=x,n_layers=y,fold=0,train_subset=train_dataset,val_subset=test_dataset)
    trained_model= train_ins.train(epochs=20)
    #final_model_path="train_test_all_model.pth"
    #torch.save(trained_model.state_dict(),final_model_path)
    #print("new trained_tested model saved") 
    
    # along epoches
    train_losses = train_ins.train_losses
    val_losses   = train_ins.val_losses
    
    # final results
    train_results=train_ins.train_res
    val_results=train_ins.val_res

    
    dir=f"nheads_{x}_nlayers_{y}_results"
    
    epo_dir = "epoch_metrics"
    folder_1 = os.path.join(dir,epo_dir)
    os.makedirs(folder_1, exist_ok=True)

    record = {
        "fold": train_ins.fold,
        "epochs_run": len(train_ins.train_losses),
        "train_losses": train_ins.train_losses,
        "val_loss": train_ins.val_losses
    }

    out_file_1 = os.path.join(folder_1, "epoch_metrics.jsonl")
    with open(out_file_1, "a") as f:
        f.write(json.dumps(record) + "\n")

    res_dir="results"
    folder_2=os.path.join(dir,res_dir)
    
    os.makedirs(folder_2,exist_ok=True)
    out_file_2=os.path.join(folder_2,"train_results.jsonl")
    with open(out_file_2,"a") as f:
        f.write(json.dumps(train_results)+"\n")

    os.makedirs(folder_2,exist_ok=True)
    out_file_3=os.path.join(folder_2,"val_results.jsonl")
    with open(out_file_3,"a") as f:
        f.write(json.dumps(val_results)+"\n")    
    