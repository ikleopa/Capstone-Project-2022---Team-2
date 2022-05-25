# -*- coding: utf-8 -*-
"""
Created on Sun May  8 18:00:20 2022

@author: Group 2
"""

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score

# models
from sklearn.naive_bayes import GaussianNB
from sklearn import svm


df_item_features = pd.read_csv(r'C:\Users\Iris\Desktop\Deree\Master\CAPSTONE\Recommender System\dressipi_recsys2022\item_features.csv')
#df_candidate_items = pd.read_csv(r'Dataset_dressipi_recsys2022\candidate_items.csv')
df_train_purchases = pd.read_csv(r"C:/Users/Iris/Desktop/Deree/Master/CAPSTONE/Recommender System/dressipi_recsys2022/train_purchases.csv")
df_train_sessions = pd.read_csv(r"C:/Users/Iris/Desktop/Deree/Master/CAPSTONE/Recommender System/dressipi_recsys2022/train_sessions.csv")

df_item_features_small=df_item_features[:10000]
df_sample_train_purchases = pd.read_csv(r'C:/Users/Iris/Downloads/sample_train_sessions.csv')
df_sample_train_sessions =pd.read_csv(r'C:/Users/Iris/Downloads/sample_train_purchases.csv')

pivot_features = df_item_features_small.pivot_table(index=['item_id'], columns='feature_category_id', 
values='feature_value_id').reset_index()    
pivot_features=pivot_features.fillna(0)

#df_item_features[df_item_features_small['feature_category_id']==1]



cols=pivot_features.columns # check for the columns of the pivot table
cols.delete(0) # remove item_id / keep only the 73 category features

for col in cols:
    pivot_features[col]=pivot_features[col].astype('category')
    
pivot_features['item_id']=pivot_features.astype('int')  


df = pd.get_dummies(pivot_features)
result_features = pd.concat([df_item_features_small['item_id'], df], axis=1)
result_features = result_features.iloc[:,1:]

result_sessions = df_sample_train_sessions.merge(result_features, on=["item_id"], how='outer')
# result_sessions = result_sessions.iloc[:,1:]

result_sessions = result_sessions.drop(df.filter(regex='_0.0').columns, axis=1)

#####################################################################################################
df1 = pd.get_dummies(df_sample_train_purchases, columns = ['item_id']) # one hot encoding for the train purchases

final = df1.merge(result_sessions, on=['session_id'])
final_df = final.drop(['date_x'],axis=1)
final_df = final_df.drop(['date_y'],axis=1)


# X: the attributes of the dataset
X = final_df.iloc[:, 824:]
# Y: the class of the dataset (click_out)
Y = final_df.iloc[:, 3:823]

# X = df_sample_train_purchases.drop(['session_id', 'date'], axis=1)


print("\nSpliting dataset on train and test sets...")
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.2, shuffle = True)



#  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
X_train=X_train.fillna(0)
Y_train=Y_train.fillna(0)

X_train = (X_train.values).reshape(-1, 1)
Y_train = (Y_train.values).reshape(-1, 1)
X_test  = (X_test.values).reshape(-1, 1)

X_train = X_train.to_numpy()
print(type(X_train))

Y_train = Y_train.to_numpy()
X_test = X_test.to_numpy()

# X_train = preprocessing.scale(np.array(X_train))
# Y_train = np.array(Y_train)
# X_test = preprocessing.scale(np.array(X_test))
# Y_test = np.array(Y_test)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)

