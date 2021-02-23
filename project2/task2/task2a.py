import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier


world_df=pd.read_csv('world.csv',encoding = 'ISO-8859-1')
life_df=pd.read_csv('life.csv',encoding = 'ISO-8859-1')
#combine dataframes
merge_df = pd.merge(world_df,life_df, on='Country Code')

feature_dict = {'feature':[],'median':[],'mean':[],'variance':[]}



# get the features
data=merge_df.iloc[:,3:23]



# get class label
classlabel = merge_df.iloc[:,-1]

# peform train test split
X_train, X_test, y_train, y_test = train_test_split(data,classlabel, train_size=2/3,random_state=100)


X_train.replace("\.{2}",np.nan,inplace=True,regex=True)
X_test.replace("\.{2}",np.nan,inplace=True,regex=True)

# impute empty values by median of  column
medians = []
for feature in data:
    median = X_train[feature].median(skipna=True)
    medians.append(median)

X_train.fillna(X_train.median(),inplace=True)    
X_test.fillna(X_train.median(),inplace=True)    
# remove the mean and scale to unit variance
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
means = list(scaler.mean_)
variances = list(scaler.var_)
i=0
for feature in data:
    feature_dict["feature"].append(feature)
    feature_dict["median"].append(medians[i])
    feature_dict["mean"].append(means[i])
    feature_dict["variance"].append(variances[i])
    i+=1
    
feature_df = pd.DataFrame(feature_dict)
feature_df = feature_df.to_csv("task2a.csv",index=False)




#perform knn with 5 neighbours
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
k5_pred=knn.predict(X_test)
k5Score = accuracy_score(y_test, k5_pred)

#perform knn with 10 neighbours
knn = neighbors.KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
k10_pred=knn.predict(X_test)
k10Score = accuracy_score(y_test, k10_pred)

#perform decision tree classifier 
dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, y_train)
dt_pred=dt.predict(X_test)
dtScore = accuracy_score(y_test,dt_pred)

print("Accuracy of decision tree:",round(dtScore,3))
print("Accuracy of k-nn (k=5):",round(k5Score,3))
print("Accuracy of k-nn (k=10):",round(k10Score,3))



