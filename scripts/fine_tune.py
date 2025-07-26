import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from model import CNN
from data_loader import MNISTDataset, transforms

def seed_everything(seed=42):
    """
    Set seeds for reproducibility across numpy, torch, and python hash.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """
    Main function to run the fine-tuning process with K-Fold cross-validation.
    """
    seed_everything()

    # --- Configuration for Fine-Tuning ---
    config = {
        'data_dir': r'C:\Users\Matthew\Documents\PhD\MNIST\MNIST\my_drawings',
        'model_path': r'C:\Users\Matthew\Documents\PhD\MNIST\models\best_model_fold_2.pt',
        'output_dir': r'C:\Users\Matthew\Documents\PhD\MNIST\models',
        'learning_rate': 1e-4,  
        'epochs': 25,          
        'batch_size': 16,
        'n_splits': 5,         
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    }
    print(f"Using device: {config['device']}")

    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)

    # --- Load Data Paths and Labels ---
    image_paths = []
    labels = []
    for label_folder in os.listdir(config['data_dir']):
        label_path = os.path.join(config['data_dir'], label_folder)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                if image_file.endswith('.png'):
                    image_paths.append(os.path.join(label_path, image_file))
                    labels.append(int(label_folder))
    
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    print(f"Found {len(image_paths)} images for fine-tuning.")

    # --- K-Fold Cross-Validation Loop ---
    kf = KFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths, labels)):
        print(f"\n===== Starting Fold {fold+1}/{config['n_splits']} =====")
        
        # --- Create Datasets and Dataloaders for the current fold ---
        train_images, train_labels = image_paths[train_idx], labels[train_idx]
        val_images, val_labels = image_paths[val_idx], labels[val_idx]

        train_dataset = MNISTDataset(train_images, train_labels, transform=transforms(is_training=True))
        val_dataset = MNISTDataset(val_images, val_labels, transform=transforms(is_training=False))

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        # --- Load Model, Freeze Layers, and Setup Optimizer for each fold ---
        model = CNN().to(config['device'])
        model.load_state_dict(torch.load(config['model_path']))
        
        # Freeze convolutional layers
        for name, param in model.named_parameters():
            if 'conv' in name:
                param.requires_grad = False
        
        # Optimizer will only update the weights of unfrozen layers
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0

        # --- Training & Validation Loop for the current fold ---
        for epoch in range(config['epochs']):
            model.train()
            for images, epoch_labels in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1} Train"):
                images, epoch_labels = images.to(config['device']), epoch_labels.to(config['device']).long()
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, epoch_labels)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, epoch_labels in val_loader:
                    images, epoch_labels = images.to(config['device']), epoch_labels.to(config['device']).long()
                    outputs = model(images)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(epoch_labels.cpu().numpy())
            
            val_acc = accuracy_score(all_labels, all_preds)
            print(f"Fold {fold+1} Epoch {epoch+1} - Val Accuracy: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(config['output_dir'], f'fine_tuned_best_model_fold_{fold+1}.pt')
                torch.save(model.state_dict(), save_path)
                print(f"Best validation accuracy improved to {best_val_acc:.4f}. Saving model to {save_path}")
        
        fold_results.append(best_val_acc)

    # --- Final Results ---
    if fold_results:
        avg_acc = np.mean(fold_results)
        print('\n=== Cross-Validation Fine-Tuning Results ===')
        for i, acc in enumerate(fold_results):
            print(f'Fold {i+1} Best Accuracy: {acc:.4f}')
        print(f'\nAverage Best Accuracy: {avg_acc:.4f}')
    else:
        print("No folds were completed successfully.")

if __name__ == "__main__":
    main()
