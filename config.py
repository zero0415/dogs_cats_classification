from pathlib import Path
from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class MLConfig:
    MODEL_DIR: Path = Path("checkpoints/")
    DATASET_DIR: Path = Path("data/train")
    TESTSET_DIR: Path = Path("data/test/test1")

    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((456, 456)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize((456, 456)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    SETTING = {'learning_rate_initial': 0.002, 'scheduler_milestones': [5, 8, 10], 'scheduler_gamma': 0.5, 
    'num_epochs': 10, 'k_folds': 5, 'batch_size': 64, 'comment': 'freezed timm efficientnet_b5 model'}
