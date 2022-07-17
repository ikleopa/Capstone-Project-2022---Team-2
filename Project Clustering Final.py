# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 11:40:20 2022

@author: Group 2
"""
import pandas as pd
from sklearn.cluster import KMeans
import prince
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.style as style
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Import csv files
global_path = ""  #Change value to the path of your desired directory it is  important to end the path with a '/'
if global_path == "":
    path = 'C:/Users/IoannisZacharis/Desktop/Capstone/Dataset_dressipi_recsys2022/'    
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

#######Normalize the vales of features between zero and one##############
df_min_max_scaled = pivot_features.copy()


# apply normalization techniques
for column in df_min_max_scaled.columns[1:]:
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    
  

######Performing PCA##################

pca_comp=PCA(n_components=5)
pca=pca_comp.fit(df_min_max_scaled)

######Performing Categorical PCA##################
mca=prince.PCA(n_components=8, n_iter=3, copy=True, check_input=True, engine='auto')
mca_fit = mca.fit(df_min_max_scaled)

mca_transformed = mca_fit.transform(df_min_max_scaled)

mca_fit.eigenvalues_
mca_fit.total_inertia_
mca_fit.explained_inertia_
# ax = mca_fit.plot_row_coordinates(X = df_min_max_scaled, figsize = (6,6))

####Since MCA cannot capture much information we will use the features-values onehotencoded
####OneHot Features##################

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

####################Performing Clustering in the onehotencoding features-values######################
# result_features=result_features[:100]

distortions = []
#total number of clusters
K = range(1,25)
#for every cluster value we calculate distortion  
#Use siloutte coefficience instead of euclidean#######################
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(result_features)
    distortions.append(sum(np.min(cdist(result_features, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / result_features.shape[0])

# Now that we have all the distortions we will plot the graph
style.use("fivethirtyeight")
plt.plot(K, distortions, 'bx-')
plt.xlabel('k-clusters')
plt.ylabel('Distortion')
plt.xlim([0, 25])
# plt.ylim([0, 3]) can set the y limit accordingly if and when needed otherwise it sets default value depending on graph
plt.title('Elbow method with Euclidean Distance')
plt.show()


#####################################Silhouette############
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,33), metric='silhouette', timings=False)
visualizer.fit(result_features)        # Fit the data to the visualizer
visualizer.show() 



####Performing CLustering using 20 clusters
number_of_clusters=7
kmeans_=KMeans(n_clusters=number_of_clusters)
reduced_cluster_labels_comp=kmeans_.fit_predict(result_features)
centers_comp=kmeans_.cluster_centers_

###Assign a Cluster to each item
cluster_assign=pd.DataFrame(reduced_cluster_labels_comp,columns=['Cluster_pred'])
cluster_assign["item_id2"]=np.unique(features[['item_id']].values).tolist()
#print(np.unique(features[['item_id']].values).tolist())
cluster_assign = cluster_assign[['item_id2', 'Cluster_pred']]


#######################################################################
########## Session ##############
######################################################################
# first product seen
# First product display date
# Last Product Viewed
# Last product display date
# Total session time (s)
# Average time on articles
# Login time of day
# A chain of products seen in order


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


###############Purchases#####################
purchases['date']=pd.to_datetime(purchases['date'])
purchases.rename(columns={'item_id':'item_purch','date':'purch_date'}, inplace=True)
purchases.head()



###########Features######################
item_clust = cluster_assign[['item_id2', 'Cluster_pred']].set_index('item_id2')
df_sess_clust=df_sessions.merge(item_clust, left_on='item_id', right_index=True)

df_clust_count = df_sess_clust.groupby(['session_id', 'Cluster_pred'])['session_id'].aggregate('count').unstack().fillna(0)
df_clust_count["most_seen_cluster"] = df_clust_count.idxmax(axis=1)

df=pd.concat([sessions, purchases, df_clust_count], axis=1)

df = df.reset_index().merge(item_clust, how='left', left_on='first_prod', right_on=item_clust.index).set_index('session_id')
df = df.rename(columns = {'Cluster_pred':'first_item_cluster'})
df = df.reset_index().merge(item_clust, how='left', left_on='last_prod', right_on=item_clust.index).set_index('session_id')
df = df.rename(columns = {'Cluster_pred':'last_item_cluster'})


df_limited = df.head(20000)
# df_limited = df[df.index<100000]
df_limited.drop('index', axis=1, inplace=True)
df_limited = df_limited.reset_index(drop=True)


########OnehotEncoding of Categories##################

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


############Scaling using min-max feature scaling##############
scaler = MinMaxScaler()
df_limited[['prod_count','time_diff','time_per_prod']] = scaler.fit_transform(df_limited[['prod_count','time_diff','time_per_prod']])

df_limited.drop('start_date', axis=1, inplace=True)
df_limited.drop('end_date', axis=1, inplace=True)
df_limited.drop('purch_date', axis=1, inplace=True)

# df_limited=df_limited.fillna(0)
Y=df_limited['item_purch']

df_limited.drop('item_purch', axis=1, inplace=True)

X=df_limited.iloc[:,5:]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2)


# drandomtree = RandomForestClassifier()
decisiontree = DecisionTreeClassifier(random_state=0)

# drandomtree = drandomtree.fit(X_train, Y_train)
decisiontree = decisiontree.fit(X_train, Y_train)

# Y_pred_drandomtree = drandomtree.predict(X_test)   
Y_pred_decisiontree = decisiontree.predict(X_test)   
# ac_drandomtree=accuracy_score(Y_test,Y_pred_drandomtree)
ac_decisiontree=accuracy_score(Y_test,Y_pred_decisiontree)

# print(ac_drandomtree)
print(ac_decisiontree)



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
    print("The mean reciprocal rank for the " + str(model) + " is " + str(mrr))
    return mrr


# mean_reciprocal_rank(drandomtree, X_test) #######RANK 270 0.01640813445718889
mean_reciprocal_rank(decisiontree, X_test) #######RANK 271 0.015921861637646417















