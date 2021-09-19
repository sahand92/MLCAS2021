import torch.optim as optim
from models.CNN_model import Net
import torch.nn as nn
import numpy as np
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

# load training dataset -------------------------------------------------------
train_weather = np.load('.\Dataset_Competition\Training\inputs_weather_train.npy')
train_others = np.load('.\Dataset_Competition\Training\inputs_others_train.npy')
train_yield = np.load('.\Dataset_Competition\Training\yield_train.npy')
cluster_ID = np.load('.\Dataset_Competition\clusterID_genotype.npy')


# weather data contains 7 variables, across 214 days:     
#    0: Average Direct Normal Irradiance (ADNI)
#    1: Average Precipitation (AP)
#    2: Average Relative Humidity (ARH)
#    3: Maximum Direct Normal Irradiance (MDNI)
#    4: Maximum Surface Temperature (MaxSur)
#    5: Minimum Surface Temperature (MinSur)
#    6: Average Surface Temperature (AvgSur)

# 'train_others' contains 5 additional variables:
#  Maturity Group (MG), Genotype ID, State, Year, 
#  and Location for each performance record.

# create dataset
dataset = BasicDataset(train_weather, train_yield, 'True')
# load dataset
trainloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)
# -----------------------------------------------------------------------------

net = Net()

# Define the loss function and optimizer
loss_function = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

num_epochs = 2
for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:
            print('Loss after mini-batch %5d: %.3f' %
                  (i + 1, running_loss / 500))
            current_loss = 0.0

print('Finished Training')