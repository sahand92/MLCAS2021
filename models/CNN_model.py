import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(7, 14, (1, 5))
        self.pool = nn.MaxPool2d((1, 2), 2)
        self.conv2 = nn.Conv2d(14, 28, (1, 5))
        self.fc1 = nn.Linear(2828, 120)
        self.fc2 = nn.Linear(120, 1)

    def forward(self, x):
        #print(x.size())
        
        x = self.conv1(x)
        #print(x.size())
        
        x = F.relu(self.pool(x))
        #print(x.size())
        
        x = F.relu(self.conv2(x))
        #print(x.size())
        
        x = F.relu(x.view(x.size()[0], -1))
        #print(x.size())
        
        x = F.relu(self.fc1(x))
        #print(x.size())
        
        x = self.fc2(x)
        #print(x.size())
        
        x = x.view(-1)
        return x


net = Net()