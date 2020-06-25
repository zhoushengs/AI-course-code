import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn as sk
import torch
import torch.nn.functional as F


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)

sex1 = pd.get_dummies(train['Sex'],drop_first=True)
embark1 = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex1,embark1],axis=1)



colname = train.columns.tolist()
colname.insert(0, 'one')
train['one']= 1
train = train.reindex(columns = colname)
X = train.drop('Survived',axis=1)

#X = train.drop('one',axis=1)
Y = train['Survived']
X = np.array(X,dtype=float)
Y = np.array(Y,dtype=float)



X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20,random_state=101)

train_columns = X_train.shape[1]
train_rows = X_train.shape[0]
test_columns = X_test.shape[1]
test_rows = X_test.shape[0]

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
Y_train = torch.unsqueeze(Y_train, dim=1)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)
Y_test = torch.unsqueeze(Y_test, dim=1)

D_in, H1, H2, D_out = train_columns, 5, 3, 2
iteration = 10000
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1, H2),
    torch.nn.Tanh(),
    torch.nn.Linear(H2, D_out),

)


criterion = torch.nn.CrossEntropyLoss()
#optimizer =torch.optim.Adadelta(model.parameters(), lr=2, rho=0.9, eps=1e-06, weight_decay=0)
#optimizer = torch.optim.Adagrad(model.parameters(), lr=5, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0045, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0025)



for i in range(iteration):
    y_pre = torch.sigmoid(model(X_train))
    loss = criterion(y_pre, Y_train.long().squeeze(1))
    model.zero_grad()
    loss.backward()
    optimizer.step()

Pre_train = model(X_train)
Pre_train = torch.softmax(Pre_train,dim=1)
Pre_train = torch.max(Pre_train, 1)
Pre_train = Pre_train[1].data.numpy()
Y_true = Y_train.data.numpy()


print(classification_report(Pre_train, Y_true))
print("Accuracy for training data")
print( sk.metrics.accuracy_score(Pre_train,Y_true))

Pre_test = model(X_test)
Pre_test = torch.softmax(Pre_test,dim=1)
Pre_test = torch.max(Pre_test, 1)
Pre_test = Pre_test[1].data.numpy()
Test_true = Y_test.data.numpy()


print(classification_report(Test_true, Pre_test))
print("Accuracy for test data")
print( sk.metrics.accuracy_score(Test_true, Pre_test))
