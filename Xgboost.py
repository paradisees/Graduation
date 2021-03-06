import xgboost as xgb
from sklearn import cross_validation

import csv
data=[]
mark=[]
res=[]
with open('/Users/hhy/Desktop/Data100.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
for i in range(10):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data, mark, test_size=0.1,random_state=i)
    dtrain=xgb.DMatrix(X_train,label=y_train)
    dtest=xgb.DMatrix(X_test)
    #watchlist = [(dtrain,'train')]
    params={'booster':'gbtree',
            'objective':'binary:logistic',
            'learning_rate':0.1,
            'max_depth':4,
            'n_estimators':400,
            'min_child_weight':1,
            'gamma':0,
            'subsample':0.8,
            'colsample_bytree':0.8,
            'nthread':4,
            'seed':1113,
            'silent': 0}
    model=xgb.train(params,dtrain,num_boost_round=30)
    ypred=model.predict(dtest)
    y_pred=(ypred>=0.5)*1
    from sklearn import metrics
    print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred))
    res.append(metrics.accuracy_score(y_test,y_pred))
    #print ('AUC: %.4f' % metrics.roc_auc_score(y_test,ypred))
    #print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred))
    #print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred))
print(sum(res)/len(res))