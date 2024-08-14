import torch.nn as nn

class my_cnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels= 8,
                      kernel_size= (5,5), stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = 2),
            nn.Conv2d(in_channels = 8, out_channels= 16,
                      kernel_size= (5,5), stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = 2),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.model(x)