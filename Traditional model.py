import numpy as np
from sklearn.lda import LDA
from sklearn import cross_validation,metrics
from sklearn.linear_model import (LinearRegression, Ridge,Lasso, RandomizedLasso)
from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import csv
data=[]
mark=[]
with open('/Users/hhy/Desktop/1/test.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
#number=[38, 54,32,25,53,80,0,10,22,28,29]
res=[]
train=[]
for i in range(10):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    data, mark, test_size=0.1, random_state=i)
    clf = MLPClassifier(random_state=1113)
    #clf = svm.SVC(kernel='linear', C=2)   #0.789888355024
    #clf = RandomForestClassifier(random_state=1113,n_jobs=4,max_depth=12,n_estimators=2100,min_samples_leaf=5,min_samples_split=14,max_features=13) #== 0.759574468085   0.758865248227
    #clf = LogisticRegression(C=4.7)   #0.79027075505
    clf.fit(X_train, y_train)
    #print('准确率:',clf.score(X_test, y_test))
    '''
    y_predict = clf.predict(X_test)
    test_auc = metrics.roc_auc_score(y_test, y_predict)  # 验证集上的auc值
    print('AUC:', test_auc)
    res.append(test_auc)'''
    #print('F1值:',metrics.f1_score(y_test, y_predict))
    res.append(clf.score(X_test, y_test))
    train.append(clf.score(X_train, y_train))
print("==",sum(res)/len(res))
print("==",sum(train)/len(train))

''' 
y_predict=clf.predict(X_test)
test_auc = metrics.roc_auc_score(y_test, y_predict)  # 验证集上的auc值
print('AUC:',test_auc)0.75808   0.7593 '''