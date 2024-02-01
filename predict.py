import torch
import torch.nn as nn
import os
import pandas as pd
from tqdm import tqdm
import argparse

from config import MLConfig
from dataset import CatDogDataset
from torch.utils.data import DataLoader

import timm

def main():
    parser = argparse.ArgumentParser(description='Dog vs Cat Classification Predict')
    parser.add_argument('--checkpoint_path', default=f'{MLConfig.MODEL_DIR}/best_model.pth', type=str, 
                    help=f'The model path used for prediction (default: {MLConfig.MODEL_DIR}/best_model.pth)')
    parser.add_argument('--csv_name', type=str, 
                    help=f'submission csv filename (default: args.checkpoint_path/preds.csv)')

    args = parser.parse_args()

    if (not args.csv_name):
        args.csv_name = f'{os.path.dirname(args.checkpoint_path)}/preds.csv'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_files = os.listdir(MLConfig.TESTSET_DIR)
    testset = CatDogDataset(test_files, MLConfig.TESTSET_DIR, mode='test', transform = MLConfig.TEST_TRANSFORM)
    test_loader = DataLoader(testset, batch_size = 64, shuffle=False)

    ## models
    model = timm.create_model('efficientnet_b5', pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(f'{args.checkpoint_path}', map_location=device))
    print(f"Load Model from {args.checkpoint_path}...")

    model.to(device)
    model.eval()

    # predict
    fn_list = []
    pred_list = []
    for images, idxs in tqdm(test_loader, desc='Predict testsets'):
        with torch.no_grad():
            images = images.to(device)
            output = model(images)
            pred = torch.argmax(output, dim=1)
            fn_list += [idx.item() for idx in idxs]
            pred_list += [p.item() for p in pred]

    print("Generating Submission File...")
    submission = pd.DataFrame({"id":fn_list, "label":pred_list})
    submission['id'] = pd.to_numeric(submission['id'])
    submission_sorted = submission.sort_values(by=submission.columns[0])
    submission_sorted.to_csv(f'{args.csv_name}', index=False)
    print(f"Submission file has been generated at {args.csv_name}")

if __name__ == "__main__":
    main()