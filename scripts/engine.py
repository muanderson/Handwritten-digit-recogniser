import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
from tqdm import tqdm

def train_epoch(model, train_loader, val_loader, config, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    best_f1 = float('-inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        total_loss = 0
        # train loop
        all_preds_train = []
        all_labels_train = []

        for image, label in tqdm(train_loader):
            image = image.to(device)
            label = label.to(device).long()

            prediction = model(image)
            loss = criterion(prediction,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_preds_train.append(prediction.argmax(dim=1).cpu().numpy())
            all_labels_train.append(label.cpu().numpy())

            total_loss += loss.item()

        all_preds_train = np.concatenate(all_preds_train)
        all_labels_train = np.concatenate(all_labels_train)

        train_acc = accuracy_score(all_preds_train, all_labels_train)
        train_f1 = f1_score(all_preds_train, all_labels_train, average='weighted')
        train_loss = total_loss / len(train_loader)
        
        #val loop
        all_preds_val = []
        all_labels_val = []

        model.eval()
        with torch.no_grad():
            for image, label in tqdm(val_loader):
                image = image.to(device)
                label = label.to(device).long()

                prediction = model(image)
                loss = criterion(prediction,label)
                all_preds_val.append(prediction.argmax(dim=1).cpu().numpy())
                all_labels_val.append(label.cpu().numpy())
                
        all_preds_val = np.concatenate(all_preds_val)
        all_labels_val = np.concatenate(all_labels_val)

        val_acc = accuracy_score(all_preds_val, all_labels_val)
        val_f1 = f1_score(all_preds_val, all_labels_val, average='weighted')

        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f}")
        print(f"Train F1: {train_f1:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        print(f"Val F1: {val_f1:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")

        if val_f1 > best_f1:
            best_f1 = val_f1
            model_path = os.path.join(
                config['output_dir'],
                f"best_model_fold_{config['fold'] + 1}.pt"
            )
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter +=1
            print(f"No improvement. Patience {patience_counter}/{config['patience']}")

        if patience_counter >= config['patience']:
            print("Early stopping triggered.")
            break

    return train_acc, train_f1, val_acc, val_f1