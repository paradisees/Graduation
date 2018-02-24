from sklearn import cross_validation
import numpy as np
import csv
from sklearn.externals import joblib
sclf=joblib.load('sclf.model')
res=[]
data=[]
mark=[]
train=[]
with open('/Users/hhy/Desktop/1/test.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
    data=np.array(data)
for i in range(10):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    data, mark, test_size=0.1, random_state=i)
    sclf.fit(X_train, y_train)
    #print('准确率:',sclf.score(X_test, y_test))
    res.append(sclf.score(X_test, y_test))
    train.append(sclf.score(X_train, y_train))
print("==", sum(res) / len(res))
print("==", sum(train) / len(train))
