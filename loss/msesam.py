from torch import nn
from metrics.sam import SAMScore


class MSESAMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.sam = SAMScore()
        self.weight = 1.0
        self.max_epochs = None

    def set_max_epochs(self, max_epochs):
        print(f"Setting max_epochs in MSESAMLoss: {max_epochs}")
        self.max_epochs = max_epochs

    def forward(self, pred, target):
        return self.weight*self.mse(pred, target) + (1-self.weight)*self.sam(pred + 1e-5, target + 1e-5)
    
    def epoch_step(self):
        self.weight = self.weight - 1/self.max_epochs
        print(f"Current MSESAMLoss MSE weight: {self.weight}")

    def __str__(self):
        return f"MSESAMLoss with max epochs: {self.max_epochs}"
