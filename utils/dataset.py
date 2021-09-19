import torch
import numpy as np
from torch.utils.data import Dataset


def preprocess_data(weather_data, yield_data):
    # scale weather data between -1 and 1 for each variable. Used in Shook et al. pp. 5?
    # note: sklearn.preprocessing.MinMaxScalar does the same job
    # other scalings to try later: https://en.wikipedia.org/wiki/Feature_scaling
    xdata = np.zeros(np.shape(weather_data))
    for i in range(6):
        xdata[:, :, i] = -1 + 2*((weather_data[:, :, i] - np.min(weather_data[:, :, i]))
                                /(np.max(weather_data[:, :, i])-np.min(weather_data[:, :, i])))
    ydata = -1 + 2*(yield_data - np.min(yield_data))/(np.max(yield_data)-np.min(yield_data))
    return xdata, ydata

class BasicDataset(Dataset):

  def __init__(self, X, y, scale_data=True):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      if scale_data:
          X, y = preprocess_data(X, y)
      self.X = torch.from_numpy(X.astype(np.float32)).permute(0,2,1).unsqueeze(-2)
      self.y = torch.from_numpy(y.astype(np.float32))

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]
