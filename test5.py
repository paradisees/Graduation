from sklearn import datasets
import csv
from sklearn import svm
from sklearn import cross_validation
import numpy as np
data=[]
mark=[]

with open('/Users/hhy/Desktop/test.csv','r',encoding='utf-8_sig') as f1:
    csv_reader=csv.reader(f1)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(list(map(float,x[-1])))
X=np.array(data)
#y=mark
y=[]
[y.extend(i) for i in mark]
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB

import numpy as np
#clf1 = svm.SVC(kernel='linear', C=2)
#clf2 = LogisticRegression(C=4.7)
clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]
clf1=RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini')
clf2=RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy')
clf3=ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini')
clf4=ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy')
clf5=GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)
rf=RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini')
sclf = StackingClassifier(classifiers=clfs,
                          meta_classifier=rf)

res=[]
for i in range(10):
    for clf, label in zip([clf1,clf2,clf3,clf4,clf5,sclf],
                          ['RandomForestClassifier',
                           'RandomForestClassifier2',
                           'ExtraTreesClassifier',
                           'ExtraTreesClassifier2',
                           'GradientBoostingClassifier',
                           'StackingClassifier']):
        scores = model_selection.cross_val_score(clf, X, y,cv=10, scoring='accuracy')
        #new = clf.fit(X_train, y_train)
        #print(label,':',new.score(X_test, y_test))
        print("Accuracy: %0.4f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        res.append(scores.mean())
print(sum(res)/len(res))
