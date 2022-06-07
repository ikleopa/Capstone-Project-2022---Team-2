# -*- coding: utf-8 -*-
"""
Created on Sun May 29 12:42:32 2022

@author: Group 4
"""

#Solution without Feature Values###############################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import lightgbm as lgb   #conda install -c conda-forge lightgbm
from sklearn.neural_network import MLPClassifier


df_item_features = pd.read_csv(r'Dataset_dressipi_recsys2022\item_features.csv')
#df_candidate_items = pd.read_csv(r'Dataset_dressipi_recsys2022\candidate_items.csv')
df_train_purchases = pd.read_csv(r'Dataset_dressipi_recsys2022\sample_train_purchases.csv')
df_train_sessions = pd.read_csv(r'Dataset_dressipi_recsys2022\sample_train_sessions.csv')

#df_item_features=df_item_features[:100000]

df_item_features_dropped=df_item_features.copy()
#df_item_features_dropped=df_item_features_dropped.drop(['feature_value_id'],axis=1)

pivot_features = df_item_features_dropped.pivot_table(index=['item_id'], columns='feature_category_id', 
                                                    values='feature_value_id').reset_index()    
pivot_features=pivot_features.fillna(0)

#df_item_features[df_item_features_small['feature_category_id']==1]



cols=pivot_features.columns # check for the columns of the pivot table
cols.delete(0) # remove item_id / keep only the 73 category features

for col in cols:
    pivot_features[col]=pivot_features[col].astype('category')
    
pivot_features['item_id']=pivot_features.astype('int')  


pivot_features.iloc[:, 1:] = pivot_features.iloc[:, 1:].clip(upper=1)

session = df_train_sessions.merge(pivot_features, on=["item_id"], how='inner')

final_session = session.drop(['item_id'],axis=1)
final_session = final_session.drop(['date'],axis=1)

final_session_sum=final_session.groupby(['session_id']).sum().reset_index()  


purchase = df_train_purchases.merge(pivot_features, on=["item_id"], how='inner')

final_purchase = purchase.drop(['item_id'],axis=1)
final_purchase = final_purchase.drop(['date'],axis=1)

final_session_sum = final_session_sum.add_prefix('featuresfromsessions_')
final_session_sum = final_session_sum.rename(columns={'featuresfromsessions_session_id': 'session_id'})
final_purchase = final_purchase.add_prefix('featuresfrompurchases_')
final_purchase = final_purchase.rename(columns={'featuresfrompurchases_session_id': 'session_id'})


algorithm_test=final_session_sum.merge(final_purchase, on=["session_id"], how='left')
algorithm_test_withoutonehotpurchases=final_session_sum.merge(df_train_purchases, on=["session_id"], how='left')

algorithm_test_withoutonehotpurchases= algorithm_test_withoutonehotpurchases.drop(['date'],axis=1)
algorithm_test_withoutonehotpurchases.set_index('session_id',inplace=True)
#
#
######################################## WIth one hot the purchases ##########

X = algorithm_test.filter(regex=('featuresfromsessions_'))
Y = algorithm_test.filter(regex=('featuresfrompurchases_')) 
Y = Y.iloc[:,:72]  #Loo why it looses a columns

Y=Y.astype('int')  

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


Y=Y.astype('int')
dtree = DecisionTreeClassifier()
drandomtree = RandomForestClassifier()
dtree = dtree.fit(X_train, Y_train)
drandomtree = drandomtree.fit(X_train, Y_train)
#
#
Y_pred_dtree = dtree.predict(X_test)   
ac_dtree=accuracy_score(Y_test,Y_pred_dtree)
Y_pred_drandomtree = drandomtree.predict(X_test)   
ac_drandomtree=accuracy_score(Y_test,Y_pred_drandomtree)

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,Y_train)
Y_pred_mlp = mlp.predict(X_test)   
ac_mlp=accuracy_score(Y_test,Y_pred_mlp)


######################################## Without one hot the purchases ##########

X2 = algorithm_test_withoutonehotpurchases.filter(regex=('featuresfromsessions_'))
Y2 = algorithm_test_withoutonehotpurchases.item_id
Y2=Y2/1000
Y2=Y2.astype('int')  
Y2=Y2.astype('str') 
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.2) 


#Y2=Y2.astype('int')
dtree2 = DecisionTreeClassifier()
dtree2 = dtree2.fit(X2_train, Y2_train)
drandomtree2 = RandomForestClassifier()
drandomtree2 = drandomtree2.fit(X2_train, Y2_train)
#
#
Y2_pred_dtree = dtree2.predict(X2_test)   
ac_dtree2=accuracy_score(Y2_test,Y2_pred_dtree)
Y2_pred_drandomtree = drandomtree2.predict(X2_test)   
ac_drandomtree2=accuracy_score(Y2_test,Y2_pred_drandomtree)


#########Linear Support Vector Classification############
#
#
#clf = OneVsRestClassifier(LinearSVC())
#clf.fit(X_train, Y_train)
# 
## model accuracy for X_test 
#ac_svm = clf.score(X_test, Y_test)
# 
#lsvc = LinearSVC(verbose=0)
#print(lsvc)



#from numpy import unique
#from numpy import where
#from sklearn.datasets import make_classification
#from sklearn.cluster import KMeans
#from matplotlib import pyplot
## define dataset
#X, _ = make_classification(n_samples=2, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
## define the model
#model = KMeans(n_clusters=2)
## fit the model
#model.fit(X)
## assign a cluster to each example
#yhat = model.predict(X)
## retrieve unique clusters
#clusters = unique(yhat)
## create scatter plot for samples from each cluster
#for cluster in clusters:
## get row indexes for samples with this cluster
#    row_ix = where(yhat == cluster)
## create scatter of these samples
#pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
## show the plot
#pyplot.show()














