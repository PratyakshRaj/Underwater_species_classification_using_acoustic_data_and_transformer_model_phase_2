import torch
from tqdm import tqdm
from sklearn.metrics import f1_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    for mfccs, padding_mask, labels in tqdm(loader):
        mfccs = mfccs.to(device)
        padding_mask = padding_mask.to(device)
        labels = labels.to(device)

        outputs = model(mfccs, padding_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, accuracy, f1


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for mfccs, padding_mask, labels in loader:
            mfccs = mfccs.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)

            outputs = model(mfccs, padding_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    f1 = f1_score(all_labels, all_preds, average="weighted")
    
    return avg_loss, accuracy, f1, all_preds, all_labels