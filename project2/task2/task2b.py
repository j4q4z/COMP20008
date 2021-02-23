import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import math
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from sklearn.cluster import KMeans
from itertools import combinations 
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif




world_df=pd.read_csv('world.csv',encoding = 'ISO-8859-1',na_values="..")
life_df=pd.read_csv('life.csv',encoding = 'ISO-8859-1')
merge_df = pd.merge(world_df,life_df, on='Country Code')


# vat visualisation used for clustering
def VAT(R):
    """
    VAT algorithm adapted from matlab version:
    http://www.ece.mtu.edu/~thavens/code/VAT.m

    Args:
        R (n*n double): Dissimilarity data input
        R (n*D double): vector input (R is converted to sq. Euclidean distance)
    Returns:
        RV (n*n double): VAT-reordered dissimilarity data
        C (n int): Connection indexes of MST in [0,n)
        I (n int): Reordered indexes of R, the input data in [0,n)
    """
        
    R = np.array(R)
    N, M = R.shape
    if N != M:
        R = squareform(pdist(R))
        
    J = list(range(0, N))
    
    y = np.max(R, axis=0)
    i = np.argmax(R, axis=0)
    j = np.argmax(y)
    y = np.max(y)


    I = i[j]
    del J[I]

    y = np.min(R[I,J], axis=0)
    j = np.argmin(R[I,J], axis=0)
    
    I = [I, J[j]]
    J = [e for e in J if e != J[j]]
    
    C = [1,1]
    for r in range(2, N-1):   
        y = np.min(R[I,:][:,J], axis=0)
        i = np.argmin(R[I,:][:,J], axis=0)
        j = np.argmin(y)        
        y = np.min(y)      
        I.extend([J[j]])
        J = [e for e in J if e != J[j]]
        C.extend([i[j]])
    
    y = np.min(R[I,:][:,J], axis=0)
    i = np.argmin(R[I,:][:,J], axis=0)
    
    I.extend(J)
    C.extend(i)
    
    RI = list(range(N))
    for idx, val in enumerate(I):
        RI[val] = idx

    RV = R[I,:][:,I]
    
    return RV.tolist(), C, I

def knn_4features(merge_df):

    # get the first four features
    first4=merge_df.iloc[:,3:8]
    # get class label
    classlabel = merge_df.iloc[:,-1]

    # peform train test split with first 4 features
    X_train, X_test, y_train, y_test = train_test_split(first4,classlabel, train_size=0.67,random_state=100)

        
    # impute empty values by median of column, imp
    
    X_train.fillna(X_train.median(),inplace=True)    
    X_test.fillna(X_train.median(),inplace=True) 

    # remove the mean and scale to unit variance
    scaler = preprocessing.StandardScaler().fit(X_train)

    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)

    #perform knn with 5 neighbours
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    k5_pred=knn.predict(X_test)
    k5Score = accuracy_score(y_test, k5_pred)
    return round(k5Score,3)


def featureeng(merge_df):
    
    # get all the features
    data=merge_df.iloc[:,3:23]
    # get class label
    classlabel = merge_df.iloc[:,-1]

    # peform train test split
    X_train, X_test, y_train, y_test = train_test_split(data,classlabel, train_size=0.67,random_state=100)

   
    #impute data
    X_train.fillna(X_train.median(),inplace=True)    
    X_test.fillna(X_train.median(),inplace=True) 
    
    X_train_copy = X_train.copy(deep=True)
    X_test_copy = X_test.copy(deep=True)
    
     # get interaction pair features
    col_list = list(combinations(data.columns,2))
    for com in col_list:
        col1 = com[0]
        col2 = com[1]
        com_name = com[0]+" * "+com[1]
        X_train_copy[com_name] = X_train[col1] * X_train[col2]
        X_train_copy.iscopy=None
        X_test_copy[com_name] = X_test[col1] * X_test[col2]
        X_test_copy.iscopy=None

        
        
     # scale data without interacion pairs
    scaler = preprocessing.StandardScaler().fit(X_train)

    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
   
    # visualise for clustering
    RV, R, I = VAT(X_train)
    x = sns.heatmap(RV, cmap='viridis', xticklabels=False, yticklabels=False)
    x.set(xlabel='Objects', ylabel='Objects')
    plt.savefig('task2bgraph1.png')

    
    # get cluster labels with 3 clusters
    km = KMeans(n_clusters=3).fit(X_train)
    cluster_labels = km.labels_
    closest = km.cluster_centers_

    # give all the test data points an assigned cluster
    clusters = []
    for feature in X_test:
        smallest_distance = math.inf
        closest_cluster = None
        # 3 possible clusters
        for i in range(0,3):
            # get respective centroid
            centroid = closest[i]
            distance = np.linalg.norm(feature-centroid)
            # new assigned cluster
            if distance < smallest_distance:
                smallest_distance = distance
                closest_cluster= i
        clusters.append(closest_cluster)
    
    X_train_copy["ClusterLabel"] = cluster_labels
    X_test_copy["ClusterLabel"] = clusters
    
    
    # using mutual information, obtain the best 4 features
    X = SelectKBest(mutual_info_classif,k=4).fit(X_train_copy,y_train)
    
    
    select_df = pd.DataFrame({"feature":list(X_train_copy.columns),'score':X.scores_})
    # order in terms of mutual information
    select_df = select_df.sort_values(by='score',ascending=False)
    fourfeat = select_df.head(4)
    fourfeat = fourfeat.set_index("feature")
    fourfeat = list(fourfeat.index)
    # convert training and testing data into the four features
    X_train = X_train_copy[fourfeat]
    X_test = X_test_copy[fourfeat]
    
    # scale data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    
    #perform knn with 5 neighbours
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    k5_pred=knn.predict(X_test)
    k5Score = accuracy_score(y_test, k5_pred)
    return round(k5Score,3)
    
def pcaknn(df):
    
    # get the features
    data=merge_df.iloc[:,3:23]
    
    # get class label
    classlabel = merge_df.iloc[:,-1]
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(data,classlabel, train_size=0.67,random_state=100)
    
    
    #impute data
    X_train.fillna(X_train.median(),inplace=True)
    X_test.fillna(X_train.median(),inplace=True)
    
    # standardise the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    
    # get the first 4 principal components
    pca = PCA(n_components=4).fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    #perform knn with 5 neighbours
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    k5_pred=knn.predict(X_test)
    k5Score = accuracy_score(y_test, k5_pred)
    return round(k5Score,3)

    
first_score =  knn_4features(merge_df.copy(deep=True))
eng_score  = featureeng(merge_df.copy(deep=True))
pca_score  = pcaknn(merge_df.copy(deep=True))
print("Accuracy of feature engineering:",eng_score)
print("Accuracy of PCA:",pca_score)
print("Accuracy of first four features:",first_score)
