import numpy as np
from sklearn.lda import LDA
from sklearn import  metrics
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import (LinearRegression, Ridge,Lasso, RandomizedLasso)
from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import csv
data=[]
mark=[]
with open('/Users/hhy/Desktop/Data100.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
X_train, X_test, y_train, y_test = train_test_split(
    data, mark, test_size=0.1,random_state=1)
clf = RandomForestClassifier(random_state=1113,n_jobs=4,max_depth=17,n_estimators=1000,max_features=10)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_train)
test_auc = metrics.roc_auc_score(y_train, y_predict)
print('训练集AUC:', test_auc)

y_predict = clf.predict(X_test)
test_auc = metrics.roc_auc_score(y_test, y_predict)
print('测试集AUC:', test_auc)
