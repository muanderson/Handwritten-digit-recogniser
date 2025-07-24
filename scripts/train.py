import os
import numpy as np
from torch.utils.data import DataLoader
from model import CNN
from data_loader import MNISTDataset, transforms
from sklearn.model_selection import KFold
import torch
from engine import train_epoch

NO_ALBUMENTATIONS_UPDATE = 1

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
    seed_everything()

    # establish config parameters
    config = {
        'output_dir': r'C:\Users\Matthew\Documents\PhD\MNIST\models',
        'train_img_dir': r'C:\Users\Matthew\Documents\PhD\MNIST\data\train',
        'learning_rate': 1e-4,
        'epochs': 100,
        'patience': 10,
        'device': torch.device("cuda:0"),
        'batch_size': 32,
        'n_splits': 5,
    }

    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)

    # Extract data paths and corresponding labels
    image_paths = []
    labels = []

    for root, dirs, files in os.walk(config['train_img_dir']):
        for images in files:
            if images.endswith('.png'):
                image_path = os.path.join(root, images)
            else:
                print('no MNIST images present')
                break

            image_paths.append(image_path)
            label = os.path.basename(root)
            label = int(label)
            labels.append(label)
    
    # Kfold split
    KF = KFold(n_splits=config['n_splits'], shuffle=True)

    image_paths = np.array(image_paths)
    labels = np.array(labels)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(KF.split(image_paths, labels)):
        config['fold'] = fold
        # Use current fold idx
        train_images, train_labels = image_paths[train_idx], labels[train_idx]
        val_images, val_labels = image_paths[val_idx], labels[val_idx]

        # Create datasets
        train_dataset = MNISTDataset(train_images, train_labels, transform=transforms(is_training=True))
        val_dataset = MNISTDataset(val_images, val_labels, transform=transforms(is_training=False))

        # Create dataloaders
        train_loader = DataLoader(train_dataset,batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())        
        val_loader = DataLoader(val_dataset,batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())  

        # Create model
        model = CNN()b 
        model.to(config['device'])

        train_acc, train_f1, val_acc, val_f1 = train_epoch(model, train_loader, val_loader, config, config['device'])

        print(f"Finished Fold: {config['fold']:.4f}")
        print(f"Best Train Acc: {train_acc:.4f}")
        print(f"Best Train F1: {train_f1:.4f}")
        print(f"Best Val Acc: {val_acc:.4f}")
        print(f"Best Val F1: {val_f1:.4f}")

        fold_results.append((val_acc, val_f1))

    if fold_results:
        avg_acc = sum(r[0] for r in fold_results) / len(fold_results)
        avg_f1 = sum(r[1] for r in fold_results) / len(fold_results)

        print('\n=== Cross-Validation Results ===')
        for i, (r1, r5) in enumerate(fold_results, 1):
            print(f'Fold {i}: Acc={r1:.4f}, F1={r5:.4f}')
        print(f'Average Acc: {avg_acc:.4f}, Average F1: {avg_f1:.4f}')
    else:
        print("No folds were completed successfully.")

if __name__ == "__main__":
    main()
