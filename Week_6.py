# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 21:11:36 2022

@author: John Sugar
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from pandasql import sqldf

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df_item_features = pd.read_csv(r'Dataset_dressipi_recsys2022\item_features.csv')
#df_candidate_items = pd.read_csv(r'Dataset_dressipi_recsys2022\candidate_items.csv')
df_train_purchases = pd.read_csv(r'Dataset_dressipi_recsys2022\train_purchases.csv')
df_train_sessions = pd.read_csv(r'Dataset_dressipi_recsys2022\train_sessions.csv')
junk = pd.read_csv(r'Dataset_dressipi_recsys2022\junk.csv', index_col=0)


df_item_features=df_item_features[:900000]
df_train_purchases = df_train_purchases[:50000]
df_train_sessions = df_train_sessions[:50000]

df_train_purchases['Purchase'] = 1
df_train_sessions['Purchase'] = 0

whole = [df_train_sessions,df_train_purchases]

whole = pd.concat(whole)


pivot_features = df_item_features.pivot_table(index=['item_id'], columns='feature_category_id', 
                                                    values='feature_value_id').reset_index()    
pivot_features=pivot_features.fillna(0)

cols=pivot_features.columns # check for the columns of the pivot table
cols.delete(0) # remove item_id / keep only the 73 category features

for col in cols:
    pivot_features[col]=pivot_features[col].astype('category')
    
pivot_features['item_id']=pivot_features.astype('int') 

pivot_features = pd.get_dummies(pivot_features)
pivot_features = pivot_features.drop(pivot_features.filter(regex='_0.0').columns, axis=1)

#
#
algorithm = whole.merge(pivot_features, on=["item_id"], how='left')
algorithm=algorithm.fillna(0)
algorithm = algorithm.drop(['date'],axis=1)

#mysql = lambda q: sqldf(q, globals())

#algorithm1=algorithm[:100000]
#algorithm2=algorithm[20000:40000]
#algorithm3=algorithm[40000:60000]
#q="SELECT session_id, Purchase, GROUP_CONCAT ( item_id ) as 'strr' FROM algorithm1 where Purchase='0' group by session_id "
#q2="SELECT session_id, Purchase, GROUP_CONCAT ( item_id ) as 'strr' FROM algorithm2 where Purchase='0' group by session_id "
#q3="SELECT session_id, Purchase, GROUP_CONCAT ( item_id ) as 'strr' FROM algorithm3 where Purchase='0' group by session_id "

#x=mysql(q)
#x2=mysql(q2)
#x3=mysql(q3)


#x = [x,x2]
#x = [x,x3]

#x = pd.concat(x)
#x.reset_index(drop=True, inplace=True)
#y = x['strr'].str.rsplit(',',  expand=True)
#y=y.fillna(0)
#y=y.astype('int') 
#y1=y.merge(pivot_features, how='left',left_on=y[0], right_on='item_id')
#
#y1=y1.fillna(0)
#y1=y1.astype('int') 
#y1=y1.merge(pivot_features, how='left',left_on=y1[1], right_on='item_id', suffixes=('_item1', '_item2'))
#
#
#junk =pd.concat([x, y1], axis=1)
#
#junk=junk.fillna(0)
#
#
#junk = junk.drop(junk.iloc[:, 3:94],axis = 1)


algorithm1=algorithm.drop(algorithm[algorithm['Purchase']==0].index)
X=junk[:1000] #Sessions/Data

Y=algorithm1[:1000] #Purchases/Target


Y = Y.drop(['item_id'],axis=1)
Y = Y.drop(['Purchase'],axis=1)
Y = Y.drop(['session_id'],axis=1)
X = X.drop(['item_id_item1'],axis=1)
X = X.drop(['item_id_item2'],axis=1)
X = X.drop(['session_id'],axis=1)
X = X.drop(['Purchase'],axis=1)
X = X.drop(['strr'],axis=1)

#X = X.filter(regex=('item1'))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


#Create KNN Classifier
knn = KNeighborsClassifier()

#Train the model using the training sets
knn.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = knn.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

