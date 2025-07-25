import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau # 1. Import the scheduler

def train_epoch(model, train_loader, val_loader, config, device):
    # --- Define optimizer, loss, and scheduler once ---
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # 2. Initialize the scheduler to monitor the validation F1 score
    # It will reduce LR if val_f1 doesn't improve for 'scheduler_patience' epochs.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max', # 'max' because a higher F1 score is better
        factor=config.get('scheduler_factor', 0.1), # Factor to reduce LR by
        patience=config.get('scheduler_patience', 5), # Epochs to wait for improvement
        verbose=True
    )
    
    best_f1 = float('-inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train() # Set model to training mode
        total_loss = 0
        all_preds_train = []
        all_labels_train = []

        # --- Train Loop ---
        for image, label in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            image = image.to(device)
            label = label.to(device).long()

            prediction = model(image)
            loss = criterion(prediction, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            all_preds_train.append(prediction.argmax(dim=1).cpu().numpy())
            all_labels_train.append(label.cpu().numpy())
            total_loss += loss.item()

        # --- Calculate Training Metrics ---
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(np.concatenate(all_labels_train), np.concatenate(all_preds_train))
        train_f1 = f1_score(np.concatenate(all_labels_train), np.concatenate(all_preds_train), average='weighted')
        
        # --- Validation Loop ---
        model.eval() # Set model to evaluation mode
        all_preds_val = []
        all_labels_val = []

        with torch.no_grad():
            for image, label in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                image = image.to(device)
                label = label.to(device).long()

                prediction = model(image)
                all_preds_val.append(prediction.argmax(dim=1).cpu().numpy())
                all_labels_val.append(label.cpu().numpy())
                
        # --- Calculate Validation Metrics ---
        val_acc = accuracy_score(np.concatenate(all_labels_val), np.concatenate(all_preds_val))
        val_f1 = f1_score(np.concatenate(all_labels_val), np.concatenate(all_preds_val), average='weighted')

        # 3. Step the scheduler with the validation F1 score
        scheduler.step(val_f1)

        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")

        # --- Early Stopping & Model Checkpointing ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0 # Reset patience
            model_path = os.path.join(
                config['output_dir'],
                f"best_model_fold_{config['fold'] + 1}.pt"
            )
            torch.save(model.state_dict(), model_path)
            print(f"Validation F1 improved to {best_f1:.4f}. Saving model.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{config['patience']}")

        if patience_counter >= config['patience']:
            print("Early stopping triggered.")
            break

    return train_acc, train_f1, val_acc, val_f1