# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:01:49 2022

@author: Group 2
"""


import pandas as pd
from sklearn.cluster import KMeans
import prince
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.style as style
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score
from sklearn.neighbors import KNeighborsClassifier
from gensim.models import Word2Vec
from sklearn.metrics import mean_squared_error,mean_absolute_error
import os

cwd = os.getcwd()

#Import csv files
global_path = ""  
if global_path == "":
    path = cwd+'/Dataset_dressipi_recsys2022/'    
else :
    path = global_path    
# Encoding
enc = 'ISO-8859-15'

# Read the csv files and import them to a DataFrame 
path_item_features = path + 'item_features.csv'
read_item_features = pd.read_csv(path_item_features, encoding = enc)
features= pd.DataFrame(read_item_features)

sessions = path + 'train_sessions.csv'
sessions = pd.read_csv(sessions, encoding = enc)
sessions= pd.DataFrame(sessions)

purchases = path + 'train_purchases.csv'
purchases = pd.read_csv(purchases, encoding = enc)
purchases= pd.DataFrame(purchases)


#Pivot table for features-values
pivot_features = features.pivot_table(index=['item_id'], columns='feature_category_id', 
                                      values='feature_value_id').reset_index()    
#Replace nan with 0s
pivot_features=pivot_features.fillna(0)


###############################################################################################################
##########################################Data Engineering#####################################################
###############################################################################################################


######################################################################################
#######Normalize the vales of features between zero and one###########################
######################################################################################
df_min_max_scaled = pivot_features.copy()
# apply min_max_scaled normalization
for column in df_min_max_scaled.columns[1:]:
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    
df_min_max_scaled.drop('item_id', axis=1, inplace=True)

######################################################################################
#######################Performing Categorical PCA#####################################
######################################################################################
mca=prince.MCA(n_components=19, n_iter=3, copy=True, check_input=True, engine='auto', random_state=42)
mca_fit = mca.fit(df_min_max_scaled)
mca_transformed = mca_fit.transform(df_min_max_scaled) 
mca_fit.eigenvalues_ #Empirical Kaiser Criterion (EKC): We keep eigenvalues with value more than 1   57.0304293
# mca_fit.total_inertia_
# mca_fit.explained_inertia_

######################################################################################
##In case MCA cannot capture much information we will use the features onehotencoded##
######################################################################################
cols=pivot_features.columns # check for the columns of the pivot table
cols.delete(0) # remove item_id / keep only the 73 category features
for col in cols:
    pivot_features[col]=pivot_features[col].astype('category')
pivot_features.drop('item_id', axis=1, inplace=True)
df = pd.get_dummies(pivot_features)
# result_features = pd.concat([features['item_id'], df], axis=1)
# result_features = df.merge(features, on='item_id', how='left')
result_features = df.iloc[:,1:]
result_features = result_features.drop(result_features.filter(regex='_0.0').columns, axis=1)
result_features=result_features.fillna(0)

######################################################################################
#######Using Word2Vector to tranform items into vectors to be used for CLustering ####
######################################################################################
#Import a pre prepared excel file that contains the sequence of items in session
whole=pd.read_excel('Whole.xlsx', index_col=0)  

##Create a corpus from the items in sessions
corpus = []
for col in whole.whole:
   word_list = col.split(",")
   corpus.append(word_list)

#######Build and train Word2Vec model
model = Word2Vec(window = 5, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

model.build_vocab(corpus, progress_per=200)

model.train(corpus, total_examples = model.corpus_count, 
            epochs=10, report_delay=1)

########Take the item_ids and their vectors
word_vectors = model.wv
values=word_vectors.vectors
index=word_vectors.key_to_index
wordvectors = pd.DataFrame(
    {
     'item_id': index
    })

wordvectors.reset_index(drop=True, inplace=True)
col_names = ['col' + str(i) for i in np.arange(values.shape[1]) + 1]
dftemp = pd.DataFrame(data=values, columns=col_names)

#create dataframe with the vector of the items to be used for clustering
df2 = pd.merge(wordvectors, dftemp, left_index=True, right_index=True)
df2.drop('item_id', axis=1, inplace=True)

###############################################################################################################
##############Experiments to identify the right number of clusters#############################################
###############################################################################################################

######################################################################################
#################Elbow Method using Euclidean Distance as metric######################
######################################################################################
distortions = []
#total number of clusters
K = range(1,30)
#for every cluster value we calculate distortion  
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df2)
    distortions.append(sum(np.min(cdist(df2, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df2.shape[0])

# Plotting the graph
style.use("fivethirtyeight")
plt.plot(K, distortions, 'bx-')
plt.xlabel('k-clusters')
plt.ylabel('Distortion')
plt.xlim([0, 30])
# plt.ylim([0, 3]) can set the y limit accordingly if and when needed otherwise it sets default value depending on graph
plt.title('Elbow method with Euclidean Distance')
plt.show()

######################################################################################
####################Elbow Method using Inertia as metric##############################
######################################################################################
wcss = []
for i in range(1,30):
    kmeans = KMeans(n_clusters=i,max_iter=300,n_init=10,random_state=0)
    km=kmeans.fit(df_min_max_scaled)
    wcss.append(kmeans.inertia_)

# Now that we have all the distortions we will plot the graph
style.use("fivethirtyeight")
plt.plot(K, wcss, 'bx-')
plt.xlabel('k-clusters')
plt.ylabel('Inertia')
plt.xlim([0, 30])
# plt.ylim([0, 3]) can set the y limit accordingly if and when needed otherwise it sets default value depending on graph
plt.title('Elbow method with Inertia')
plt.show()

######################################################################################
###################Elbow Method using Silhouette as metric############################
smodel = KMeans()
visualizer = KElbowVisualizer(smodel, k=(15,33), metric='silhouette', timings=False)
visualizer.fit(df_min_max_scaled)        # Fit the data to the visualizer
visualizer.show() 

######################################################################################
##CLustering using n clusters. Number of CLusters demends from the approach we use####
######################################################################################
number_of_clusters=19
kmeans_=KMeans(n_clusters=number_of_clusters)
reduced_cluster_labels_comp=kmeans_.fit_predict(mca_transformed)
centers_comp=kmeans_.cluster_centers_

######################################################################################
########################Assign a Cluster to each item#################################
######################################################################################
cluster_assign=pd.DataFrame(reduced_cluster_labels_comp,columns=['Cluster_pred'])
cluster_assign["item_id2"]=np.unique(features[['item_id']].values).tolist()

cluster_assign = cluster_assign[['item_id2', 'Cluster_pred']]


######################################################################################################
############################################Sessions##################################################
######################################################################################################
# First product in the session
# First product date
# Last Product in the session
# Last product display date
# Session time measured in seconds
# Average time
# The part of the day that the session took place
# Sequence of the products in the session


df_sessions = pd.read_csv(path+'train_sessions.csv')
# df_sessions = df_sessions[:50000]
df_sessions.head()

session_sorted = df_sessions.sort_values(by=['date'])

# First date of session
begin_df = session_sorted.groupby('session_id')['date'].first().rename('start_date') 

# First product of session
first_prod=session_sorted.groupby('session_id')['item_id'].first().rename('first_prod')

# Number of products seen in the session
prod_count=session_sorted.groupby('session_id')['item_id'].count().rename('prod_count')

# Last date of the session
end_df = session_sorted.groupby('session_id')['date'].last().rename('end_date')

# Last product of the session
last_prod=session_sorted.groupby('session_id')['item_id'].last().rename('last_prod')

# Concat the datasets
times_df=pd.concat([begin_df,end_df,first_prod,last_prod,prod_count],axis=1)

# Proper time format
times_df['start_date']=pd.to_datetime(times_df['start_date'])
times_df['end_date']=pd.to_datetime(times_df['end_date'])

# Calculatind the date difference
times_df['time_diff']=(times_df.end_date-times_df.start_date).astype('timedelta64[s]')

# Calculating the time per product
times_df['time_per_prod']=times_df['time_diff']/times_df['prod_count']

# Adding the time of the day
mask=(times_df.start_date.dt.hour>=0) & (times_df.start_date.dt.hour<7)
times_df.loc[mask,'time_first_prod']='Morning'
mask=(times_df.start_date.dt.hour>=7) & (times_df.start_date.dt.hour<12)
times_df.loc[mask,'time_first_prod']='Day'
mask=(times_df.start_date.dt.hour>=12) & (times_df.start_date.dt.hour<18)
times_df.loc[mask,'time_first_prod']='Noon'
mask=(times_df.start_date.dt.hour>=18) & (times_df.start_date.dt.hour<24)
times_df.loc[mask,'time_first_prod']='Night'
# Get the product chain
df_sessions['idx'] = df_sessions.groupby('session_id').cumcount()
df_sessions['item_idx'] = 'product_' + df_sessions.idx.astype(str)

item = df_sessions.pivot(index='session_id',columns='item_idx',values='item_id')

item['item_serie']=item.astype(str).agg('-'.join,axis=1)

item['item_serie'].replace(r'.0','', regex=True, inplace=True)
item['item_serie'].replace(r'-nan','', regex=True, inplace=True)

item.head()

sessions=pd.concat([times_df,item['item_serie']], axis=1)
sessions

######################################################################################################
############################################Purchases#################################################
######################################################################################################

purchases['date']=pd.to_datetime(purchases['date'])
purchases.rename(columns={'item_id':'item_purch','date':'purch_date'}, inplace=True)



##############Combining item clusters with the session information###############
item_clust = cluster_assign[['item_id2', 'Cluster_pred']].set_index('item_id2')
df_sess_clust=df_sessions.merge(item_clust, left_on='item_id', right_index=True)

df_clust_count = df_sess_clust.groupby(['session_id', 'Cluster_pred'])['session_id'].aggregate('count').unstack().fillna(0)
df_clust_count["most_seen_cluster"] = df_clust_count.idxmax(axis=1)

df=pd.concat([sessions, purchases, df_clust_count], axis=1)

df = df.reset_index().merge(item_clust, how='left', left_on='first_prod', right_on=item_clust.index).set_index('session_id')
df = df.rename(columns = {'Cluster_pred':'first_item_cluster'})
df = df.reset_index().merge(item_clust, how='left', left_on='last_prod', right_on=item_clust.index).set_index('session_id')
df = df.rename(columns = {'Cluster_pred':'last_item_cluster'})


#############Taking a subset of the data###################
df_limited = df.head(20000)
# df_limited = df[df.index<100000]
df_limited.drop('index', axis=1, inplace=True)
df_limited = df_limited.reset_index(drop=True)

######################################################################################################
#########################OnehotEncoding of Categorical Variables######################################
######################################################################################################
########time_first_prod - Day/Night#######
df_temp=pd.get_dummies(df_limited.time_first_prod)
df_limited= pd.concat([df_limited, df_temp], axis=1)
df_limited.drop('time_first_prod', axis=1, inplace=True)
df_limited.rename(columns = {'Day':'time_first_prod'}, inplace = True)


##########First Product Seen###############
df_temp=pd.get_dummies(df_limited.first_prod,prefix = 'first_prod')
df_limited= pd.concat([df_limited, df_temp], axis=1)
df_limited.drop('first_prod', axis=1, inplace=True)

##########Last Product Seen###############
df_temp=pd.get_dummies(df_limited.last_prod,prefix = 'last_prod')
df_limited= pd.concat([df_limited, df_temp], axis=1)
df_limited.drop('last_prod', axis=1, inplace=True)

##########most_common_cat###############
# df_temp=pd.get_dummies(df_limited.most_common_cat,prefix = 'most_common_cat')
# df_limited= pd.concat([df_limited, df_temp], axis=1)
# df_limited.drop('most_common_cat', axis=1, inplace=True)

##########most_seen_cluster###############
df_temp=pd.get_dummies(df_limited.most_seen_cluster,prefix = 'most_seen_cluster')
df_limited= pd.concat([df_limited, df_temp], axis=1)
df_limited.drop('most_seen_cluster', axis=1, inplace=True)

##########first_item_cluster###############
df_temp=pd.get_dummies(df_limited.first_item_cluster,prefix = 'first_item_cluster')
df_limited= pd.concat([df_limited, df_temp], axis=1)
df_limited.drop('first_item_cluster', axis=1, inplace=True)

##########last_item_cluster###############
df_temp=pd.get_dummies(df_limited.last_item_cluster,prefix = 'last_item_cluster')
df_limited= pd.concat([df_limited, df_temp], axis=1)
df_limited.drop('last_item_cluster', axis=1, inplace=True)


##########################Scaling using min-max feature scaling####################################
scaler = MinMaxScaler()
df_limited[['prod_count','time_diff','time_per_prod']] = scaler.fit_transform(df_limited[['prod_count','time_diff','time_per_prod']])

df_limited.drop('start_date', axis=1, inplace=True)
df_limited.drop('end_date', axis=1, inplace=True)
df_limited.drop('purch_date', axis=1, inplace=True)

######################################################################################################
#########################train_test_split the data to perform the experiments#########################
######################################################################################################
Y=df_limited['item_purch']
Yonehot = pd.get_dummies(Y) #Onehotencoding the target item in case it should be used instead of the item id
df_limited.drop('item_purch', axis=1, inplace=True)

X=df_limited.iloc[:,5:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Yonehot,test_size=0.2)

################Initiaize and fit the classifiers##################
drandomtree = RandomForestClassifier(criterion='entropy', max_depth=6, max_features= 'auto', n_estimators= 200,random_state=0)
decisiontree = DecisionTreeClassifier(criterion='entropy', max_depth=7, max_features= None, splitter= 'best',random_state=0)
knn = KNeighborsClassifier(leaf_size='30', metric= 'minkowski', n_neighbors = 5, p= 2)

drandomtree = drandomtree.fit(X_train, Y_train)
decisiontree = decisiontree.fit(X_train, Y_train)
knn = knn.fit(X_train, Y_train)

Y_pred_drandomtree = drandomtree.predict(X_test)   
Y_pred_decisiontree = decisiontree.predict(X_test)   
Y_pred_knn = knn.predict(X_test)  

######################################################################################################
######################################Results#########################################################
######################################################################################################

######################F1 Score##########################
print('F1 Score of Random Forest:')
print(f1_score(Y_test, Y_pred_drandomtree, average='weighted'))

print('F1 Score of Decision Tree:')
print(f1_score(Y_test, Y_pred_decisiontree, average='weighted'))

print('F1 Score of KNeighbors:')
print(f1_score(Y_test, Y_pred_knn, average='weighted'))


################Precision Score##########################
print('Precision Score of Random Forest:')
print(precision_score(Y_test, Y_pred_drandomtree, average='macro'))

print('Precision Score of Decision Tree:')
print(precision_score(Y_test, Y_pred_decisiontree, average='macro'))

print('Precision Score of KNeighbors:')
print(precision_score(Y_test, Y_pred_knn, average='macro'))


################Accuracy Score##########################
print('Accuracy Score of Random Forest:')
print(accuracy_score(Y_test,Y_pred_drandomtree))

print('Accuracy Score of Decision Tree:')
print(accuracy_score(Y_test,Y_pred_decisiontree))

print('Accuracy Score of KNeighbors:')
print(accuracy_score(Y_test,Y_pred_knn))


################Mean Absolute Error (MAE)##########
print('MAE of Random Forest:')
print(mean_absolute_error(Y_test,Y_pred_drandomtree))

print('MAE of Decision Tree:')
print(mean_absolute_error(Y_test,Y_pred_decisiontree))

print('MAE of KNeighbors:')
print(mean_absolute_error(Y_test,Y_pred_knn))

#################mean_reciprocal_rank##################
def mean_reciprocal_rank(model, set_to_test, rr = 0, cont = 0):

    predictions = model.predict_proba(set_to_test)

    pred_df_test = pd.DataFrame(predictions)
    pred_df_test.columns = model.classes_

    pred_df_test["session_id"] = X_test.index
    pred_df_test = pred_df_test.merge(df[["item_purch"]], how='inner', on='session_id')
    pred_df_test = pred_df_test[["session_id", "item_purch"] + list(pred_df_test.columns[:-2])]

    for index, row in pred_df_test.iterrows():
        item_purch_act = int(row.iloc[1])
        row = row.iloc[2:]
        row_sorted = row.sort_values(ascending=False)
        items = row_sorted.index.to_list()
        if item_purch_act in items:
            rank = int(items.index(item_purch_act)) + 1
            if rank <= 100:
                rr += 1/rank
        cont += 1
    mrr = rr/cont
    return mrr

print('The mean reciprocal rank of Random Forest:')
mean_reciprocal_rank(drandomtree, X_test) #######RANK 270 in the Leaderboard

print('The mean reciprocal rank of Decision Tree:')
mean_reciprocal_rank(decisiontree, X_test) #######RANK 271 in the Leaderboard

print('The mean reciprocal rank of KNeighbors:')
mean_reciprocal_rank(knn, X_test)


######################################################################################################
#############Experiments with GridSearch to find optimal parameters for the classifiers###############
######################################################################################################

# from sklearn.model_selection import GridSearchCV

##########Best Parameters for Decision Tree Classifier##########################
# params = {
#     'criterion':  ['gini', 'entropy'],
#     'max_depth':  [None, 8, 10, 12],
#     'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, ],
#     'splitter': ['best', 'random']
# }

# clf = GridSearchCV(
#     estimator=DecisionTreeClassifier(),
#     param_grid=params,
#     cv=5,
#     n_jobs=5,
#     verbose=1,
# )

# clf.fit(X_train, Y_train)
# print(clf.best_params_)

# Best Parameters for Decision Tree Classifier: {'criterion': 'entropy', 'max_depth': 10, 'max_features': None, 'splitter': 'best'}


######################Best Parameters for Random Forest Classifier##########################
# param_grid = { 
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [7,8,10],
#     'criterion' :['gini', 'entropy']
# }

# CV_rfc = GridSearchCV(estimator=drandomtree, param_grid=param_grid, cv= 5)
# CV_rfc.fit(X_train, Y_train)
# print(CV_rfc.best_params_)

# Best Parameters for Random Forest Classifier:{'criterion': 'entropy', 'max_depth': 7, 'max_features': 'auto', 'n_estimators': 200}



# # ###################Best Parameters for KNN Classifier##########################
# from sklearn.model_selection import GridSearchCV
# leaf_size = list(range(1,10))
# n_neighbors = list(range(1,10))
# p=[1,2]
# hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
# clf = GridSearchCV(knn, hyperparameters, cv=5)
# best_model = clf.fit(X_train,Y_train)

# Best Parameters for Random KNeighborsClassifier:{leaf_size': 30, 'metric': 'minkowski', n_neighbors': 5, p': 2}










