#Capstone 02-06-2022

##########################################################
################## IMPORTING LIBRARIES ###################
##########################################################
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
%matplotlib inline
import datetime
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import datetime as dt
##########################################################
################## READING FILES  ########################
##########################################################

global_path = 'C:/Users/marko/OneDrive/Υπολογιστής/Capostone_2022/'
if global_path == "":
    path ={}.format(global_path) 
else :
    path = global_path    
store_Results = path

# Encoding
enc = 'ISO-8859-15'

################### ITEM FEATURES ##########################
path_item_features = path + 'item_features.csv'
read_item_features = pd.read_csv(path_item_features, encoding = enc)
df_item_features= pd.DataFrame(read_item_features)
df_item_features.columns = df_item_features.columns.str.strip().str.lower().str.replace('-', '_')

################### TRAIN SESSION ##########################
'''
# A session is equal to a day, so a session is one user's activity on one day. '''
df_train_sessions = path + 'train_sessions.csv'
df_train_sessions = pd.read_csv(df_train_sessions, encoding = enc)
df_train_sessions= pd.DataFrame(df_train_sessions)
df_train_sessions.columns = df_train_sessions.columns.str.strip().str.lower().str.replace('-', '_')

################### TRAIN PURCHASES #########################
df_train_purchases = path + 'train_purchases.csv'
df_train_purchases = pd.read_csv(df_train_purchases, encoding = enc)
df_train_purchases= pd.DataFrame(df_train_purchases)
df_train_purchases.columns = df_train_purchases.columns.str.strip().str.lower().str.replace('-', '_')

################### TAKING SUBSET ##########################
df_item_features=df_item_features[:30000]
df_train_purchases = df_train_purchases[:30000]
df_train_sessions = df_train_sessions[:30000]

###################### INSIGHTS OF DF's #####################
# Find out how many items there are for each purchase.
'''
# There are many items that have been purchased only once, and it seems difficult 
to predict such items due to the amount of data.
# It seems that it has been purchased less than once as a volume zone 10^2
# On the other hand, it can be confirmed that a few items occupy more than one purchase.10^3
'''

df_purchases_counts = df_train_purchases["item_id"].value_counts().values #convert to array 
ax = plt.figure(figsize = (16, 8)).add_subplot(111)
ax.hist(x = df_purchases_counts, bins = np.logspace(0, 4, 50), color = 'green', alpha = 0.75)
ax.set_xscale('log')
ax.set_xlabel('Number of Purchases')
ax.set_ylabel("Number of Items")

#################### EXTRACT DAYWEEK/MONTH/HOUR ###############

# Check if there is a tendency for the day of the week or time of purchase

#new column date as datetime -- from here we can extract time and hour and month of the purchases 

df_train_purchases['date_as_datetime']=pd.to_datetime(df_train_purchases['date'])
#extract day
df_train_purchases['day'] = df_train_purchases['date_as_datetime'].dt.date
#extract weekday 
df_train_purchases['weekday'] = df_train_purchases['date_as_datetime'].dt.weekday #Monday=0 Sunday=6
#extract weekday name 
df_train_purchases['weekday_name'] = df_train_purchases['date_as_datetime'].dt.day_name().str[:3]
#extract month 
df_train_purchases['month'] = df_train_purchases['date_as_datetime'].dt.month 
#extract month name 
df_train_purchases['month_name'] = df_train_purchases['date_as_datetime'].dt.month_name().str[:3]
#extract_time
df_train_purchases['time'] = df_train_purchases['date_as_datetime'].dt.time
#extract year 
df_train_purchases['year'] = df_train_purchases['date_as_datetime'].dt.year
#extract hour
df_train_purchases['hour'] = df_train_purchases['date_as_datetime'].dt.hour


## Extract Time Zone of each purchase 
def timezone(hour):
    if hour> 4.0 and hour <=8.0:
        return "Early Morning"
    elif hour > 8.0 and hour<=12.0:
        return "Morning"
    elif hour > 12.0 and hour <=16.0:
        return "Afternoon"
    elif hour > 16.0 and hour <=20.0:
        return "Evening"
    elif hour > 20.0 and hour <=24.0:
        return "Night"
    else:
        return "midnight"
#xreate a new column with the results of def timezone to the data 
df_train_purchases["Timezone"] = df_train_purchases["hour"].apply(timezone) 


#Merge Train_session and train_purchases on session_id to see the sequence of items viewed and purchsed at the end of the session

# 1. Check how many products viewed in in first  sessions
df_train_sessions_counts = df_train_sessions.groupby("session_id").count()["item_id"].values

ax = plt.figure(figsize = (16, 8)).add_subplot(111)
ax.hist(x = df_train_sessions_counts, bins = np.logspace(0, 2, 50), color = 'green', alpha = 0.75)
ax.set_xscale('log')
ax.set_xlabel('Duration of each session')
ax.set_ylabel("Number of sessions")

print("Average duration each session", df_train_sessions_counts.mean())

#################### PREPROCESSING  ######################

# There seems to be some features that are commonly used. -two items may have same feature category 56 and 47  share the same feature :  23691
df_item_features["feature_category_id"].value_counts()

#Pivot Table shows : which items have specific values in each of the catefories 
df_item_features_pivot = df_item_features.pivot_table(index = "item_id",values = "feature_value_id",columns = "feature_category_id", aggfunc = "count")
df_item_features_pivot.fillna(0.0, inplace = True)

#normalization of the data  
df_item_features_pca = df_item_features_pivot.apply(lambda x:  (x - x.mean()) / x.std(), axis = 0) 

print("============== Fill all NA Values with Zero  ==============")
# Column 56 has NA  Values 0 - fill them with 0 
display(df_item_features_pca.loc[:,df_item_features_pca.isnull().any()]) 
df_item_features_pca.fillna(0.0, inplace = True) 

#################### PCA ######################
#Apply PCA in the features data 
pca = PCA()
pca.fit(df_item_features_pca)
feature = pca.transform(df_item_features_pca)


############# EXPLAINED RATIO of PCA #################
var = pca.explained_variance_ratio_[0:10] #percentage of variance explained
labels = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']

plt.figure(figsize=(15,7))
plt.bar(labels,var,)
plt.xlabel('Pricipal Component')
plt.ylabel('Proportion of Variance Explained')

############################

pca_test = PCA().fit(df_item_features_pca)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

########################
pca_noisy = PCA(0.5).fit(pca_test)
pca.n_components_



print("=============THE RESULTS OF PCA FOR ALL THE ITEMS ARE: ==============")
df_item_features_pca = pd.DataFrame(feature, columns = ["PC{}".format(x + 1) for x in range(len(df_item_features_pca.columns))], index = df_item_features_pivot.index)
display(df_item_features_pca)


# Let's plot items for some principal components. It seems to be divided into good feelings as it is.
# It may be better to drop items into 4 to 5 categories.
pca_pair_list = [["PC1", "PC2"], ["PC2", "PC3"], ["PC1", "PC3"]]

fig = plt.figure(figsize = (24, 8))

for idx, pair in enumerate(pca_pair_list):
    # Clustering the results dropped in two dimensions.
    pred = KMeans(n_clusters = 4).fit_predict(df_item_features_pca[pair].values) 
    
    ax = fig.add_subplot(1, len(pca_pair_list), idx + 1)
    pca_x, pca_y = pair # x-axis and y-axis
    ax.scatter(df_item_features_pca[pca_x], df_item_features_pca[pca_y], alpha = 0.5, c = pred)
    ax.set_xlabel(pca_x)
    ax.set_ylabel(pca_y)


ax = plt.figure(figsize = (12, 6)).add_subplot(1, 1, 1)
ax.plot(np.arange(len(pca.explained_variance_)), pca.explained_variance_)
ax.set_yscale("log")
ax.set_xlabel('Order of eigenvectors')
ax.set_ylabel("Eigenvalues of each eigenvector")


# pca.components_ is a representation of the principal component vector in the feature space. From here, we can also see how each feature works on the main components.
df_pca_components = pd.DataFrame(pca.components_, index = ["PC{}".format(x + 1) for x in range(len(df_item_features_pca.columns))], columns = df_item_features_pivot.columns)
display(df_pca_components)











