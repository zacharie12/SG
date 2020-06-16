from abc import ABC 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
class Cnn(ABC, nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
    def forward(self):
        pass

class StdConv2d(nn.Conv2d):

  def forward(self, x):
    w = self.weight
    v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    w = (w - m) / torch.sqrt(v + 1e-10)
    return F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)

class SimpleCNN(Cnn):
    def __init__(self, H, W, C, output_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(((H//8)* (W//8))*64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

class GroupCNN(Cnn):
    def __init__(self, H, W, C, output_size):
        super(GroupCNN, self).__init__()
        self.conv1 = nn.Sequential(
            StdConv2d(C, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            StdConv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(16, 32),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            StdConv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
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
        #print('In CNN:', x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

class ResNetBased(Cnn):
    def __init__(self, H, W, C, output_size):
        super(ResNetBased, self).__init__()
        model = models.resnet152(pretrained=True)
        n_inputs = model.fc.in_features
        # add more layers as required
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(n_inputs, output_size))
        ]))

        model.fc = classifier

        self.resnet = model
    def forward(self, x):
        x = self.resnet(x)
        return x

CNN_ARCHITECHTURE = { 'SimpleCNN' : SimpleCNN, 'GroupCNN' : GroupCNN, 'ResNet': ResNetBased }

def build_cnn(cfg):
    cnn = CNN_ARCHITECHTURE[cfg.LISTENER.CNN](cfg.LISTENER.IMAGE_SIZE, cfg.LISTENER.IMAGE_SIZE, 3, cfg.LISTENER.CNN_OUTPUT)
    return cnn

if __name__ == '__main__':
    cnn = GroupCNN(24, 24, 3, 4)
    x = torch.zeros((1, 3, 24, 24))
    output = cnn(x)
    print(output)        