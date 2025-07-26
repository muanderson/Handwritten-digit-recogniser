import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import mlflow
import mlflow.pytorch
from model import CNN
from data_loader import MNISTDataset, transforms
from engine import train_epoch 
from sklearn.model_selection import KFold

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

    # --- Configuration ---
    config = {
        'output_dir': r'models', 
        'train_img_dir': r'C:\Users\Matthew\Documents\PhD\MNIST\data\train', 
        'learning_rate': 1e-4,
        'epochs': 100,
        'patience': 10,
        'scheduler_patience': 5, 
        'scheduler_factor': 0.1,  
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'batch_size': 32,
        'n_splits': 5,
    }

    # --- MLflow Setup ---
    # Set an experiment name. MLflow will create it if it doesn't exist.
    mlflow.set_experiment("MNIST Base Model Training")

    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)

    # --- Data Loading ---
    image_paths = []
    labels = []
    for root, _, files in os.walk(config['train_img_dir']):
        for image_file in files:
            if image_file.endswith('.png'):
                image_paths.append(os.path.join(root, image_file))
                labels.append(int(os.path.basename(root)))
    
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # --- K-Fold Cross-Validation ---
    kf = KFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths, labels)):
        config['fold'] = fold + 1
        
        # --- MLflow Run for each Fold ---
        with mlflow.start_run(run_name=f"Fold_{config['fold']}"):
            print(f"\n===== Starting Fold {config['fold']}/{config['n_splits']} =====")
            
            # 1. Log parameters for this run
            mlflow.log_params(config)

            # --- Dataloaders ---
            train_images, train_labels = image_paths[train_idx], labels[train_idx]
            val_images, val_labels = image_paths[val_idx], labels[val_idx]
            train_dataset = MNISTDataset(train_images, train_labels, transform=transforms(is_training=True))
            val_dataset = MNISTDataset(val_images, val_labels, transform=transforms(is_training=False))
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

            # --- Model ---
            model = CNN().to(config['device'])

            # 2. Run the training epoch, which will log metrics per-epoch
            # The engine now returns the best metrics and the path to the best model
            best_val_acc, best_val_f1, best_model_path = train_epoch(model, train_loader, val_loader, config, config['device'])

            # 3. Log the best model from this fold as an artifact
            if best_model_path and os.path.exists(best_model_path):
                # We log the model state dict, not the whole model object
                mlflow.pytorch.log_model(model, "model", registered_model_name=f"mnist-cnn-fold-{config['fold']}")
            
            # 4. Log the final summary metrics for this fold
            mlflow.log_metric("best_validation_accuracy", best_val_acc)
            mlflow.log_metric("best_validation_f1", best_val_f1)

            print(f"Finished Fold: {config['fold']}")
            print(f"Best Val Acc: {best_val_acc:.4f}")
            print(f"Best Val F1: {best_val_f1:.4f}")
            fold_results.append((best_val_acc, best_val_f1))

    # --- Final Results ---
    if fold_results:
        avg_acc = np.mean([r[0] for r in fold_results])
        avg_f1 = np.mean([r[1] for r in fold_results])
        print('\n=== Cross-Validation Results ===')
        for i, (acc, f1) in enumerate(fold_results, 1):
            print(f'Fold {i}: Best Val Acc={acc:.4f}, Best Val F1={f1:.4f}')
        print(f'Average Best Val Acc: {avg_acc:.4f}, Average Best Val F1: {avg_f1:.4f}')
    else:
        print("No folds were completed successfully.")

if __name__ == "__main__":
    main()