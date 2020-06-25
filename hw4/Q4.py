import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train = pd.read_csv('pima-indians-diabetes.csv')







X_train, X_test, Y_train, Y_test = train_test_split(train.drop('Class',axis=1),train['Class'], test_size=0.20,random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)

predictions = logmodel.predict(X_test)

print(classification_report(Y_test, predictions))
print("accuracy:", accuracy_score(Y_test, predictions))