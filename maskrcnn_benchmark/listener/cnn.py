from abc import ABC 
import torch
import torch.nn as nn


class Cnn(ABC, nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
    def forward(self):
        pass

class SimpleCNN(Cnn):
    def __init__(self, H, W, C, output_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(((H//8)* (W//8))*64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


CNN_ARCHITECHTURE = { 'SimpleCNN' : SimpleCNN }

def build_cnn(cfg):
    cnn = CNN_ARCHITECHTURE[cfg.LISTENER.CNN](cfg.LISTENER.IMAGE_SIZE, cfg.LISTENER.IMAGE_SIZE, 3, cfg.LISTENER.CNN_OUTPUT)
    return cnn

if __name__ == '__main__':
    cnn = SimpleCNN(24, 24, 3, 4)
    x = torch.zeros((1, 3, 24, 24))
    output = cnn(x)
    print(output)        