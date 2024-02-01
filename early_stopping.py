import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0, checkpoint_path='early_stop_checkpoint.pth', log_file=None):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path
        self.log_file = log_file

    def __call__(self, val_loss, model):
        print(f"EarlyStopping val_loss: {val_loss} vs best_score: {self.best_score}", file=self.log_file)
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        print(f'Model saved! Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f})', file=self.log_file)
