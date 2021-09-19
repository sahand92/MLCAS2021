import matplotlib.pyplot as plt


X_train = torch.from_numpy(train_weather.astype(np.float32)).permute(0,2,1).unsqueeze(-2)
y_train = torch.from_numpy(train_yield.astype(np.float32))
y_train_np = y_train.detach().numpy()

y_pred = net(X_train)
y_pred_np = y_pred.detach().numpy()

plt.scatter(y_train_np,y_pred_np)