import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)

sex1 = pd.get_dummies(train['Sex'],drop_first=True)
embark1 = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex1,embark1],axis=1)

sex2 = pd.get_dummies(test['Sex'],drop_first=True)
embark2 = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex2,embark2],axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size=0.20,random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)

predictions = logmodel.predict(X_test)
print(classification_report(Y_test, predictions))

print("accuracy:", accuracy_score(Y_test, predictions))