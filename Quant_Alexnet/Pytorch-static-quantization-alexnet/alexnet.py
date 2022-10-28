import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):  
        super().__init__()
          
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4)  
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  

        self.conv2 = nn.Conv2d(64, 192, 5, padding=2) 
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)  

        self.conv3 = nn.Conv2d(192, 384, 3, padding=1) 
        self.relu3 = nn.ReLU(inplace=False)

        self.conv4 = nn.Conv2d(384, 256, 3, padding=1) 
        self.relu4 = nn.ReLU(inplace=False)

        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)  
        self.relu5 = nn.ReLU(inplace=False)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)  
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.drop1   = nn.Dropout(p=0.5, inplace=False)
        self.linear1 = nn.Linear(in_features=(9216), out_features=4096)
        self.relu6   = nn.ReLU(inplace=False)
        self.drop2   = nn.Dropout(p=0.5, inplace=False)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu7   = nn.ReLU(inplace=False)
        self.linear3 = nn.Linear(in_features=4096, out_features=num_classes)
        
        self.init_bias()  

    def init_bias(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.conv2.bias, 1)
        nn.init.constant_(self.conv4.bias, 1)
        nn.init.constant_(self.conv5.bias, 1)

    def forward(self, x):
      
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)
        
        x = torch.flatten(x, 1)  # reduce the dimensions for linear layer input

        x = self.drop1(x)
        x = self.linear1(x)
        x = self.relu6(x)
        x = self.drop2(x)
        x = self.linear2(x)
        x = self.relu7(x)
        x = self.linear3(x)
        
        return x