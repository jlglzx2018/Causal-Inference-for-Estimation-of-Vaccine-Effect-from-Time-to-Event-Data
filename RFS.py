import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
#load data
train = pd.read_csv("./data/train.csv")
##control
data = train.copy()
data = data.loc[data["a"]==0,]
X_train = data[['x0_0', 'x0_1', 'x1']]
data['tr'] = np.where(data['event']==1, True, False)
data_y = data[['tr', 'Z']].to_numpy()
data_y
data_y.shape
dt=[('fstat', '?'), ('lenfol', '<f8')]
aux = [(e1,e2) for e1,e2 in data_y]
y_train = np.array(aux, dtype=dt)
y_train.shape
estimator = RandomSurvivalForest().fit(X_train, y_train)
#hazard function for test dataset
test = pd.read_csv("./data/test.csv", index_col = False)
test['ID'] = range(1, len(test) + 1)
data = test.copy()
data = data.loc[data["a"]==0,]
X_test = data[['x0_0', 'x0_1', 'x1']]
data['tr'] = np.where(data['event']==1, True, False)
data_y = data[['tr', 'Z']].to_numpy()
data_y
data_y.shape
dt=[('fstat', '?'), ('lenfol', '<f8')]
aux = [(e1,e2) for e1,e2 in data_y]
y_test = np.array(aux, dtype=dt)
surv_fns = estimator.predict_survival_function(X_test)
hzrd_fns = estimator.predict_cumulative_hazard_function(X_test)
####extract survival probability
result = [0 for a in range(203)]
for i in range(surv_fns.shape[0]):
  df = surv_fns[i](surv_fns[i].x)
  result = np.vstack([result, df])
#result = np.transpose(result)
result = pd.DataFrame(result)
result = result.iloc[1:,]
result['ID'] = data['ID'].values
result.to_csv("./RSF/RFS_surv_con.csv")

####extract hazard probability
result1 = [0 for a in range(203)]
for i in range(hzrd_fns.shape[0]):
  df = hzrd_fns[i](hzrd_fns[i].x)
  result1 = np.vstack([result1, df])
#result = np.transpose(result)
result1 = pd.DataFrame(result1)
result1 = result1.iloc[1:,]
result1['ID'] = data['ID'].values
result1.to_csv("./RSF/RFS_hzrd_con.csv")




##case
data = train.copy()
data = data.loc[data["a"]==1,]
X_train = data[['x0_0', 'x0_1', 'x1']]
data['tr'] = np.where(data['event']==1, True, False)
data_y = data[['tr', 'Z']].to_numpy()
data_y
data_y.shape
dt=[('fstat', '?'), ('lenfol', '<f8')]
aux = [(e1,e2) for e1,e2 in data_y]
y_train = np.array(aux, dtype=dt)
y_train.shape
estimator = RandomSurvivalForest().fit(X_train, y_train)
#hazard function for test dataset
data = test.copy()
data = data.loc[data["a"]==1,]
X_test = data[['x0_0', 'x0_1', 'x1']]
data['tr'] = np.where(data['event']==1, True, False)
data_y = data[['tr', 'Z']].to_numpy()
data_y
data_y.shape
dt=[('fstat', '?'), ('lenfol', '<f8')]
aux = [(e1,e2) for e1,e2 in data_y]
y_test = np.array(aux, dtype=dt)
surv_fns = estimator.predict_survival_function(X_test)
hzrd_fns = estimator.predict_cumulative_hazard_function(X_test)
####extract survival probability
result = [0 for a in range(202)]
for i in range(surv_fns.shape[0]):
  df = surv_fns[i](surv_fns[i].x)
  result = np.vstack([result, df])
#result = np.transpose(result)
result = pd.DataFrame(result)
result = result.iloc[1:,]
result['ID'] = data['ID'].values
result.to_csv("./RSF/RFS_surv_case.csv")

####extract hazard probability
result1 = [0 for a in range(202)]
for i in range(hzrd_fns.shape[0]):
  df = hzrd_fns[i](hzrd_fns[i].x)
  result1 = np.vstack([result1, df])
#result = np.transpose(result)
result1 = pd.DataFrame(result1)
result1 = result1.iloc[1:,]
result1['ID'] = data['ID'].values
result1.to_csv("./RSF/RFS_hzrd_case.csv")
