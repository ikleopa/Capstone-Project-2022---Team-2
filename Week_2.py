# -*- coding: utf-8 -*-
"""
Created on Sun May  8 18:00:20 2022

@author: Group 2
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer


df_item_features = pd.read_csv(r'C:\Users\Iris\Desktop\Deree\Master\CAPSTONE\Recommender System\dressipi_recsys2022\item_features.csv')
#df_candidate_items = pd.read_csv(r'Dataset_dressipi_recsys2022\candidate_items.csv')
df_train_purchases = pd.read_csv(r"C:/Users/Iris/Desktop/Deree/Master/CAPSTONE/Recommender System/dressipi_recsys2022/train_purchases.csv")
df_train_sessions = pd.read_csv(r"C:/Users/Iris/Desktop/Deree/Master/CAPSTONE/Recommender System/dressipi_recsys2022/train_sessions.csv")

df_item_features_small=df_item_features[:10000]
df_sample_train_purchases = df_train_purchases[:10000]
df_sample_train_sessions = df_train_sessions[:10000]

pivot_features = df_item_features_small.pivot_table(index=['item_id'], columns='feature_category_id', 
values='feature_value_id').reset_index()    
pivot_features=pivot_features.fillna(0)

#df_item_features[df_item_features_small['feature_category_id']==1]



cols=pivot_features.columns
cols.delete(0)

for col in cols:
    pivot_features[col]=pivot_features[col].astype('category')
    
pivot_features['item_id']=pivot_features.astype('int')  


df = pd.get_dummies(pivot_features)
result_features = pd.concat([df_item_features_small['item_id'], df], axis=1)
result_features = result_features.iloc[:,1:]

result_sessions = result_features.merge(df_sample_train_sessions, on=["item_id"])
#result_sessions = result_sessions.iloc[:,1:]

result_sessions = result_sessions.drop(df.filter(regex='_0.0').columns, axis=1)


df1 = pd.get_dummies(df_sample_train_purchases)

# TEST TEST TEST

