
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

from config import MLConfig

def main():
    parser = argparse.ArgumentParser(description='Dog vs Cat Classification Evaluation')
    parser.add_argument('--npz_path', default=f'{MLConfig.MODEL_DIR}/record.npz', type=str, 
                    help=f'The model path used for prediction (default: {MLConfig.MODEL_DIR}/best_model.pth)')
    parser.add_argument('--plot', default='all',
                        choices=["cm", "roc", "lc", "all"],
                        help='cm (confusion matrix), roc (roc curve), lc (learning curve[loss, accuracy]), all(all of them), (default: all)')
    parser.add_argument('--folds', default=5, type=int,
                        help='k-fold cross-validation for training (default: 5)')

    args = parser.parse_args()

    images_path = os.path.dirname(args.npz_path)

    data = np.load(args.npz_path, allow_pickle=True)
    roc_data = data['roc_data'].item()
    conf_data = data['conf_data'].item()
    record = data['record'].item()

    folds = args.folds

    if (args.plot == "roc" or args.plot == "all"):
        # ROC curve data
        fprs = [[] for i in range(folds)]
        tprs = [[] for i in range(folds)]
        thresholds = [[] for i in range(folds)]
        roc_aucs = [[] for i in range(folds)]

        for fold in range(folds):
            fprs[fold] = roc_data[fold]['fpr']
            tprs[fold] = roc_data[fold]['tpr']
            thresholds[fold] = roc_data[fold]['thresholds']
            roc_aucs[fold] = roc_data[fold]['auc']    

        # 繪製 ROC 曲線
        plt.figure(figsize=(8, 8))
        colors = ['#FF4500', '#FF8C00', '#D2B48C', '#A52A2A', '#8B4513']
        for fold in range(folds):
            plt.plot(fprs[fold][0], tprs[fold][0], color=colors[fold], lw=2, label=f'{fold}_ROC curve (AUC = {roc_aucs[fold][0]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f'{images_path}/{folds}-folds_ROC.png')
        print(f'{images_path}/{folds}-folds_ROC.png is generated.')
        # plt.show()

    if (args.plot == "cm" or args.plot == "all"):
        # confusion matrix data
        precision = [[] for i in range(folds)]
        recall = [[] for i in range(folds)]
        f1_score = [[] for i in range(folds)]
        validation_accuracy = [[] for i in range(folds)]
        tps = [[] for i in range(folds)]
        tns = [[] for i in range(folds)]
        fps = [[] for i in range(folds)]
        fns = [[] for i in range(folds)]
        confusion_matrix = [[] for i in range(folds)]
        sum_acc = 0.0
        sum_precision = 0.0
        sum_recall = 0.0
        sum_f1_score = 0.0

        for fold in range(folds):
            precision[fold] = np.array(conf_data[fold]['precision'])
            recall[fold] = np.array(conf_data[fold]['recall'])
            f1_score[fold] = np.array(conf_data[fold]['f1-score'])
            validation_accuracy[fold] = np.array(record[fold]['validation_accuracy'])

            tps[fold] = len(conf_data[fold]['tp'])
            tns[fold] = len(conf_data[fold]['tn'])
            fps[fold] = len(conf_data[fold]['fp'])
            fns[fold] = len(conf_data[fold]['fn'])
            
            fold_acc = (tps[fold]+tns[fold])/(tps[fold]+tns[fold]+fps[fold]+fns[fold]+1E-10)
            fold_precision = tps[fold]/(tps[fold]+fps[fold]+1E-10)
            fold_recall = tps[fold]/(tps[fold]+fns[fold]+1E-10)
            fold_f1_score = 2*fold_recall*fold_precision/(fold_recall+fold_precision+1E-10)
            sum_acc += fold_acc
            sum_precision += fold_precision
            sum_recall += fold_recall
            sum_f1_score += fold_f1_score

            print(f"Fold: {fold}, Val Acc:{fold_acc:.4f}, Precision: {fold_precision:.4f}, Recall: {fold_recall:.4f}, F1-Score: {fold_f1_score:.4f}")
            confusion_matrix[fold] = [[tps[fold], fps[fold]], [fns[fold], tns[fold]]]
        print(f"{folds}-Fold Avg: Val Acc:{sum_acc/folds:.4f}, Precision: {sum_precision/folds:.4f}, Recall: {sum_recall/folds:.4f}, F1-Score: {sum_f1_score/folds:.4f}")
        plt.figure(figsize=(20, 12))
        
        row = 2
        col = -(-folds // row) # ceil(fold//row)
        for fold in range(folds):
            # 繪製 Confusion Matrix
            plt.subplot(row, col, fold+1)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            sns.heatmap(confusion_matrix[fold], annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0 (cat)', 'Predicted 1 (dog)'], yticklabels=['Actual 0 (cat)', 'Actual 1 (dog)'])
            plt.text(0.5, 1.8, 'False Negatives', color='#808080', fontsize=10, ha='center', va='center')
            plt.text(0.5, 0.8, 'True Negatives', color='#808080', fontsize=10, ha='center', va='center')
            plt.text(1.5, 0.8, 'False Positives', color='#808080', fontsize=10, ha='center', va='center')
            plt.text(1.5, 1.8, 'True Positives', color='#808080', fontsize=10, ha='center', va='center')
            plt.text(0.2, 2.2, f'Acc: {validation_accuracy[fold][-1]:.4f}, Recall: {recall[fold][-1]:.4f}, \nPrec: {precision[fold][-1]:.4f}, F1-Score: {f1_score[fold][-1]:.4f}', fontsize=6, ha='center', va='center')

            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Fold {fold}: Last Epoch Confusion Matrix')
        plt.savefig(f'{images_path}/Last_epoch_confusion_matrix.png')
        print(f'{images_path}/Last_epoch_confusion_matrix.png is generated.')
        # plt.show()

    if (args.plot == "lc" or args.plot == "all"):

        training_loss = [[] for i in range(folds)]
        training_accuracy = [[] for i in range(folds)]
        validation_loss = [[] for i in range(folds)]
        validation_accuracy = [[] for i in range(folds)]

        for fold in range(folds):
            training_loss[fold] = record[fold]['training_loss']
            training_accuracy[fold] = record[fold]['training_accuracy']
            validation_loss[fold] = record[fold]['validation_loss']
            validation_accuracy[fold] = record[fold]['validation_accuracy']

        plt.figure(figsize=(10, 15))

        for fold in range(folds):
            epochs = np.arange(1, len(training_loss[fold]) + 1)

            # 繪製 Loss vs Epoch
            plt.subplot(folds, 2, 2*fold+1)
            plt.plot(epochs, training_loss[fold], label='Training Loss', color='#20B2AA')
            plt.plot(epochs, validation_loss[fold], label='Validation Loss', color='orange')
            plt.title(f'Fold {fold}: Training/Validation Loss vs Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # 繪製 Accuracy vs Epoch
            plt.subplot(folds, 2, 2*fold+2)
            plt.plot(epochs, training_accuracy[fold], label='Training Accuracy', color='#20B2AA')
            plt.plot(epochs, validation_accuracy[fold], label='Validation Accuracy', color='orange')
            plt.title(f'Fold {fold}: Training/Validation Accuracy vs Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'{images_path}/{folds}-folds_Learning_curves.png')
        print(f'{images_path}/{folds}-folds_Learning_curves.png is generated.')
        # plt.show()

if __name__ == "__main__":
    main()