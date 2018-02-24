from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.grid_search import GridSearchCV

data=[]
mark=[]
with open('/Users/hhy/Desktop/Data122.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
#parameters = {'kernel': ('linear', 'sigmoid'), 'C': [1, 2],'gamma': [0.001, 0.002, 0.004, 0.006, 0.01, 0.02, 0.04, 0.1]}
#parameters = {'kernel': ('linear', 'sigmoid'), 'C': [1,2],'gamma':[0.001,0.002,0.004,0.006,0.01,0.02,0.04,0.1]}
#param_test1= {'C':[i for i in np.arange(0.5,5,0.1)]}
#param_test1= {'n_estimators':[x for x in range(1100,2500,100)]} # 83396
#param_test2= {'max_depth':[i for i in range(3,21,2)], 'min_samples_split':[j for j in range(5,60,5)]}
#param_test3= {'min_samples_split':[x for x in range(2,20,2)], 'min_samples_leaf':[y for y in range(5,40,5)]}
param_test4= {'max_features':[x for x in range(7,16,1)]}
gsearch1= GridSearchCV(estimator = RandomForestClassifier(random_state=1113,n_jobs=4,max_depth=19,n_estimators=2100,min_samples_leaf=5,min_samples_split=14,max_features=13),
                       param_grid =param_test4, scoring='roc_auc',cv=5)
gsearch1.fit(data,mark)
print(gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_)

#,n_estimators=900,max_depth=15,min_samples_split=15,min_samples_leaf=10

'''random_state=1113,n_jobs=4,max_depth=17,n_estimators=1000,max_features=10'''

'''n_estimators=2100,'max_depth': 19, 'min_samples_split': 5'''