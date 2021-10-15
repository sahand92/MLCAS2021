import torch.optim as optim
from models.CNN_model import Net
import torch.nn as nn
import random
import gc
import pickle
import numpy as np
import pandas as pd
import torch
from MLCAS2021.utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

# load training dataset -------------------------------------------------------
train_weather = np.load('.\Dataset_Competition\Training\inputs_weather_train.npy')
train_others = np.load('.\Dataset_Competition\Training\inputs_others_train.npy')
train_yield = np.load('.\Dataset_Competition\Training\yield_train.npy')
cluster_ID = np.load('.\Dataset_Competition\clusterID_genotype.npy')

test_weather = np.load('.\Dataset_Competition\Test_inputs\inputs_weather_test.npy')
test_others = np.load('.\Dataset_Competition\Test_Inputs\inputs_others_test.npy')

#tr_W = train_weather[:, 91:214,:]
#tr_W_R = np.reshape(tr_W, (93028, 861))
#tr_W = train_weather[:, :,:]
tr_W_R = np.reshape(train_weather, (93028, 1498))

TR = np.concatenate([tr_W_R, np.delete(train_others, 2, 1)], axis=1)
TR1 = np.concatenate([TR, train_yield[:, None]], axis=1)
random.shuffle(TR1)

del train_weather, train_others, tr_W_R, cluster_ID
gc.collect()
#TR = np.random.shuffle(np.concatenate([tr_W_R, np.delete(train_others, 2, 1)], axis=1))
val_indices = np.random.choice(93028, size=9300, replace=False)
val = TR1[val_indices, :]
tr = np.delete(TR1,  val_indices, 0)

#ts_W = test_weather[:, 91:214,:]
#ts_W_R = np.reshape(ts_W, (10337, 861))
#ts_W = test_weather[:, :,:]
ts_W_R = np.reshape(test_weather, (10337, 1498))
TS = np.concatenate([ts_W_R, np.delete(test_others, 2, 1)], axis=1)

del ts_W, ts_W_R, TR1
gc.collect()

################################################
from sklearn.linear_model import LinearRegression
from lineartree import LinearTreeRegressor
from sklearn.datasets import make_regression
MT = LinearTreeRegressor(base_estimator=LinearRegression())
MT.fit(tr[:, :-1], pd.to_numeric(tr[:, -1]))

pred_MT = MT.predict(val[:,:-1])
MAE_MT = round(np.mean(abs(pred_MT - pd.to_numeric(val[:, 865]))), 2)
RMSE_MT = round(np.sqrt(((pred_MT - pd.to_numeric(val[:, 865])) **2).mean()), 2)
print('MAE:', MAE_MT, 'RMSE', RMSE_MT)

pred_test = MT.predict(TS)
np.savetxt('pred1_mt.csv', pred_test, delimiter=",")
np.save('pred2_MLCAS2021_AgVic.csv', pred_test)
pickle.dump(MT, open('MT_1_model.sav', 'wb'))



#######RF
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_estimators = 80)
RF.fit(tr[:, :-1], tr[:, -1])

pred = RF.predict(val[:,:-1])
MAE_RF = round(np.mean(abs(pred - pd.to_numeric(val[:, 865]))), 2)
RMSE_RF = round(np.sqrt(((pred - pd.to_numeric(val[:, 865])) **2).mean()), 2)
print('MAE:', MAE_RF, 'RMSE', RMSE_RF)

pred_test = RF.predict(TS)
np.savetxt('pred4_80_rf.csv', pred_test, delimiter=",")
np.save('pred4_80_MLCAS2021_AgVic.npy', pred_test)
pickle.dump(RF, open('RF_4_80_model.sav', 'wb'))



############################
from sklearn import neighbors
KNN = neighbors.KNeighborsRegressor(n_neighbors=5)
KNN.fit(tr[:, :-1], tr[:, -1])

pred = RF.predict(val[:,:-1])
MAE_RF = round(np.mean(abs(pred - pd.to_numeric(val[:, 865]))), 2)
RMSE_RF = round(np.sqrt(((pred - pd.to_numeric(val[:, 865])) **2).mean()), 2)
print('MAE:', MAE_RF, 'RMSE', RMSE_RF)

pred_test = RF.predict(TS)
np.savetxt('pred4_80_rf.csv', pred_test, delimiter=",")
np.save('pred4_80_MLCAS2021_AgVic.npy', pred_test)
pickle.dump(RF, open('RF_4_80_model.sav', 'wb'))

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
#
# net = Net()
#
# # Define the loss function and optimizer
# loss_function = nn.L1Loss()
# optimizer = optim.Adam(net.parameters(), lr=1e-4)
#
# num_epochs = 2
# for epoch in range(num_epochs):  # loop over the dataset multiple times
#
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, targets = data
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = loss_function(outputs, targets)
#         loss.backward()
#         optimizer.step()

        # print statistics
#         running_loss += loss.item()
#         if i % 10 == 0:
#             print('Loss after mini-batch %5d: %.3f' %
#                   (i + 1, running_loss / 500))
#             current_loss = 0.0
#
# print('Finished Training')