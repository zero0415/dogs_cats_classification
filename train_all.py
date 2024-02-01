import torch
import torch.nn as nn
import sys, os
import numpy as np
from tqdm import tqdm
import signal

from config import MLConfig
from dataset import CatDogDataset
from torch.utils.data import DataLoader

from early_stopping import EarlyStopping

import timm 

def signal_handler(signal, frame, model, record, roc_data, conf_data):
    print('Pressed Ctrl+C! Saving logs and model...', file=logfile)
    torch.save(model.state_dict(), f'{MLConfig.MODEL_DIR}/interrupt_model.pth')
    np.savez(f'{MLConfig.MODEL_DIR}/interrupt_logs.npz', record=record, roc_data=roc_data, conf_data=conf_data, allow_pickle=True)

    sys.exit(0)


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    train_files = os.listdir(MLConfig.DATASET_DIR)
    datasets = CatDogDataset(train_files, MLConfig.DATASET_DIR, transform = MLConfig.TRAIN_TRANSFORM)

    # Model
    model = timm.create_model('efficientnet_b5', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)

    model.to(device)

    # Training Setting
    setting = MLConfig.SETTING
    print(f"Training Setting: {setting}", file=logfile)
    np.savez(f'{MLConfig.MODEL_DIR}/training_setting.npz', setting=setting, allow_pickle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=setting['learning_rate_initial'], amsgrad=True)

    # train all datasets use MultiStepLR
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=setting['scheduler_milestones'], gamma=setting['scheduler_gamma'])
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    # Ctrl+C Signal
    signal.signal(signal.SIGINT, lambda signal, frame: signal_handler(signal, frame, model, record, roc_data, conf_data))

    num_epochs = setting['num_epochs']
    batch_size = setting['batch_size']

    best_loss = float('inf')
    best_epoch = -1

    record = {}
    conf_data = {}
    roc_data = {}

    train_loader = DataLoader(datasets, batch_size = batch_size, shuffle=True)
    fold = 0

    record[fold] = {'training_loss': [], 'training_accuracy': [], 'validation_loss': [], 'validation_accuracy': [], 'learning_rate': []}
    conf_data[fold] = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'precision': [], 'recall': [], 'f1-score': []}
    roc_data[fold] = {'fpr': [], 'tpr': [], 'thresholds': [], 'auc': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels, filenames in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += labels.size(0)

            probs, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

        # Calculate Training Loss, accuracy
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}, Learning_Rate: {optimizer.param_groups[0]['lr']}", file=logfile)
       
        record[fold]['training_loss'].append(avg_loss)
        record[fold]['training_accuracy'].append(accuracy)
        record[fold]['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            print(f"Fold {fold} Best Model save at epoch {best_epoch} and loss {best_loss}", file=logfile)
            torch.save(model.state_dict(), f'{MLConfig.MODEL_DIR}/best_model.pth')
            
        early_stopping(avg_loss, model)
        scheduler.step() # MultiStepLR
        
        if early_stopping.early_stop:
            print("Early stopping", file=logfile)
            break

        torch.save(model.state_dict(), f'{MLConfig.MODEL_DIR}/epoch_{epoch}_checkpoint.pth')
    torch.save(model.state_dict(), f'{MLConfig.MODEL_DIR}/last_checkpoint.pth')
    np.savez(f'{MLConfig.MODEL_DIR}/record.npz', record=record, roc_data=roc_data, conf_data=conf_data, allow_pickle=True)

if __name__ == "__main__":
    if not os.path.exists(MLConfig.MODEL_DIR):
        os.makedirs(MLConfig.MODEL_DIR)
    logfile = open(f'{MLConfig.MODEL_DIR}/log.txt', 'a+', buffering = 1)
    main()
    logfile.close()