import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dropout_prob=0.1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout_prob)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=2*dropout_prob),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 10, 3)
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)        
    
    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)