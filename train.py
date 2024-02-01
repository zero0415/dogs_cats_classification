import torch
import torch.nn as nn
import sys, os
import numpy as np
from tqdm import tqdm
import signal

from config import MLConfig
from dataset import CatDogDataset
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
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

    # Training setting    
    setting = MLConfig.SETTING
    print(f"Training Setting: {setting}", file=logfile)
    np.savez(f'{MLConfig.MODEL_DIR}/training_setting.npz', setting=setting, allow_pickle=True)    
    num_epochs = setting['num_epochs']
    k_folds = setting['k_folds']
    batch_size = setting['batch_size']
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)


    record = {}
    conf_data = {}
    roc_data = {}
    for fold, (train_index, valid_index) in enumerate(skf.split(datasets.files, datasets.labels)):
        print(f"Fold {fold + 1}/{k_folds}: Training: {len(train_index)}, Validation: {len(valid_index)}", file=logfile)

        # Model
        model = timm.create_model('efficientnet_b5', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)

        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=setting['learning_rate_initial'], amsgrad=True)

        # train cross validation datasets use ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

        # Ctrl+C Signal
        signal.signal(signal.SIGINT, lambda signal, frame: signal_handler(signal, frame, model, record, roc_data, conf_data))
        
        train_dataset = torch.utils.data.Subset(datasets, train_index)
        valid_dataset = torch.utils.data.Subset(datasets, valid_index)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        # Reset earlystop
        early_stopping = EarlyStopping(patience=4, delta=0.001, checkpoint_path=f"{MLConfig.MODEL_DIR}/{fold}_early_stop_model.pth", log_file=logfile)

        record[fold] = {'training_loss': [], 'training_accuracy': [], 'validation_loss': [], 'validation_accuracy': [], 'learning_rate': []}
        conf_data[fold] = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'precision': [], 'recall': [], 'f1-score': []}
        roc_data[fold] = {'fpr': [], 'tpr': [], 'thresholds': [], 'auc': []}

        best_loss = float('inf')
        best_epoch = -1

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


            # Validation
            model.eval()
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            all_probs = np.array([])
            all_preds = np.array([])
            all_labels = np.array([])

            with torch.no_grad():
                for images, labels, filenames in tqdm(valid_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    total_loss += loss.item()
                    total_samples += labels.size(0)

                    probs, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == labels).sum().item()

                    tp_mask = (labels == 1) & (predicted == 1)
                    fn_mask = (labels == 1) & (predicted == 0)
                    fp_mask = (labels == 0) & (predicted == 1)
                    tn_mask = (labels == 0) & (predicted == 0)

                    conf_data[fold]['tp'] += np.array(filenames)[tp_mask.cpu().numpy()].tolist()
                    conf_data[fold]['fn'] += np.array(filenames)[fn_mask.cpu().numpy()].tolist()
                    conf_data[fold]['fp'] += np.array(filenames)[fp_mask.cpu().numpy()].tolist()
                    conf_data[fold]['tn'] += np.array(filenames)[tn_mask.cpu().numpy()].tolist()

                    all_preds = np.concatenate([all_preds, predicted.cpu().numpy()])
                    all_probs = np.concatenate([all_probs, probs.detach().cpu().numpy()])
                    all_labels = np.concatenate([all_labels, labels.cpu().numpy()])

            confusion_matrix = np.zeros((2, 2))
            for i in range(len(all_labels)):
                predicted_label = 1 if all_probs[i] > 0.5 else 0
                confusion_matrix[int(all_labels[i]), predicted_label] += 1

            # Calculate Precision, Recall, F1-score
            tps = len(conf_data[fold]['tp'])
            tns = len(conf_data[fold]['tn'])
            fps = len(conf_data[fold]['fp'])
            fns = len(conf_data[fold]['fn'])

            precision = tps/(tps+fps+1E-10)
            recall = tps/(tps+fns+1E-10)
            f1_score = 2*recall*precision/(recall+precision+1E-10)
            
            conf_data[fold]['precision'].append(precision)
            conf_data[fold]['recall'].append(recall)
            conf_data[fold]['f1-score'].append(f1_score)
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            roc_auc = auc(fpr, tpr)

            roc_data[fold]['fpr'].append(fpr)
            roc_data[fold]['tpr'].append(tpr)
            roc_data[fold]['thresholds'].append(thresholds)
            roc_data[fold]['auc'].append(roc_auc)

            # Calculate Validation loss, accuracy
            avg_loss = total_loss / len(valid_loader)
            accuracy = correct_predictions / total_samples
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}, Learning_Rate: {optimizer.param_groups[0]['lr']}", file=logfile)

            record[fold]['validation_loss'].append(avg_loss)
            record[fold]['validation_accuracy'].append(accuracy)
            record[fold]['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # Save the best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                print(f"Fold {fold} Best Model save at epoch {best_epoch} and loss {best_loss}", file=logfile)
                torch.save(model.state_dict(), f'{MLConfig.MODEL_DIR}/{fold}_best_model.pth')

            early_stopping(avg_loss, model)
            scheduler.step(avg_loss) # ReduceLROnPlateau scheduler need avg_loss to update
            
            if early_stopping.early_stop:
                print("Early stopping", file=logfile)
                break

        torch.save(model.state_dict(), f'{MLConfig.MODEL_DIR}/{fold}_last_checkpoint.pth')
    np.savez(f'{MLConfig.MODEL_DIR}/record.npz', record=record, roc_data=roc_data, conf_data=conf_data, allow_pickle=True)

if __name__ == "__main__":
    if not os.path.exists(MLConfig.MODEL_DIR):
        os.makedirs(MLConfig.MODEL_DIR)
    logfile = open(f'{MLConfig.MODEL_DIR}/log.txt', 'a+', buffering = 1)
    main()
    logfile.close()