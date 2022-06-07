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
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import lightgbm as lgb   #conda install -c conda-forge lightgbm


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


#
#
########################################
#
#X=algorithm_test.iloc[:,3:]
#Y=algorithm_test.iloc[:,1]
X = algorithm_test.filter(regex=('featuresfromsessions_'))
Y = algorithm_test.filter(regex=('featuresfrompurchases_'))
Y = Y.iloc[:,:72]  #Loo why it looses a columns
#cols=Y.columns
#for col in cols:
#    Y[col]=Y[col].astype('category')
#X = X.clip(upper=1)
#
Y=Y.astype('int')  

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#

Y=Y.astype('int')
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, Y_train)
#
#
Y_pred_dtree = dtree.predict(X_test)   
ac_dtree=accuracy_score(Y_test,Y_pred_dtree)
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

















