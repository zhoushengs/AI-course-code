import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import torch
from sklearn.datasets import load_boston


boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
boston.isnull().sum()
X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']
X = np.array(X,dtype=float)
Y = np.array(Y, dtype=float)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=5)


train_columns = X_train.shape[1]
train_rows = X_train.shape[0]
test_columns = X_test.shape[1]
test_rows = X_test.shape[0]

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_train = torch.unsqueeze(Y_train, dim=1)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)
Y_test = torch.unsqueeze(Y_test, dim=1)


D_in, H1, H2, D_out = train_columns, 16, 32, 1

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1, H2),
    torch.nn.Tanh(),
    torch.nn.Linear(H2, D_out),

)


loss_fn = torch.nn.MSELoss()
#optimizer =torch.optim.Adadelta(model.parameters(), lr=2, rho=0.9, eps=1e-06, weight_decay=0)
#optimizer = torch.optim.Adagrad(model.parameters(), lr=5, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0045, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0025)
for t in range(30001):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, Y_train)
    if t % 1000 == 0:
        print(loss)
    model.zero_grad()
    loss.backward()
    optimizer.step()

t_pre = model(X_test)
t_loss = loss_fn(t_pre, Y_test)
print("The MSE of training data is :")
#print(y_pred)
print(loss)

print("The MSE of test data is :")
print(t_loss)
