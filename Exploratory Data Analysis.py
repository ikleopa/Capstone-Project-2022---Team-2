import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for prettier plots
from prettytable import PrettyTable #pip install pretty table
import re
import datetime

##############################################################################
#####                           IMPORT DATASETS                          #####
##############################################################################

global_path = ""  #Change value to the 'path' variable of your desired directory it is  important to end the path with '/'
if global_path == "":
    path = 'C:/Users/Iris/Desktop/Deree/Master/CAPSTONE/Recommender System/dressipi_recsys2022/'    
else :
    path = global_path    
store_Results = path

# Read the csv files and import them to a DataFrame 
    # Item Features
item_features_df = path + 'item_features.csv'
item_features_df = pd.read_csv(item_features_df)
item_features_df= pd.DataFrame(item_features_df)
    # Sessions
train_sessions_df = path + 'train_sessions.csv'
train_sessions_df = pd.read_csv(train_sessions_df)
train_sessions_df= pd.DataFrame(train_sessions_df)
    # Purchases
train_purchases_df  = path + 'train_purchases.csv'
train_purchases_df  = pd.read_csv(train_purchases_df)
train_purchases_df = pd.DataFrame(train_purchases_df )
    # Candidate items - - Used only for visualization purposes
candidate_items_df = path + 'candidate_items.csv'
candidate_items_df = pd.read_csv(candidate_items_df)
candidate_items_df = pd.DataFrame(candidate_items_df)
    # Leaderboard - - Used only for visualization purposes
test_leaderboard_sessions_df  = path + 'test_leaderboard_sessions.csv'
test_leaderboard_sessions_df  = pd.read_csv(test_leaderboard_sessions_df)
test_leaderboard_sessions_df  = pd.DataFrame(test_leaderboard_sessions_df )
    # Final Sessions - - Used only for visualization purposes
test_final_sessions_df  = path + 'test_final_sessions.csv'
test_final_sessions_df  = pd.read_csv(test_final_sessions_df)
test_final_sessions_df  = pd.DataFrame(test_final_sessions_df )

###################################### ITEM FEATURES ######################################
print('-'*50,'\n UNDERSTANDING OF THE ITEM FEATURES DATASET', '-'*50)

    #Check the number of rows and columns
rows,columns=item_features_df.shape
print(' ---> Dataset Dimensions: \n','The Number of Rows are: ',rows,'\n The Number of Columns are: ',columns)
  
    # Preview of the item_features dataset 
print('\n ---> A preview of the dataset for: \n', 'The five first items are: \n', item_features_df.head(5), '\n The five last items: \n', item_features_df.tail(5))
    
    # Columns of item_features
itemfeatures_columns = PrettyTable(['item_id','feature_category_id','feature_value_id'])
print('The columns of the item_features dataset are :\n',itemfeatures_columns)
    
    # Number of features categories & values
feature_category_groupby = item_features_df.groupby(['feature_category_id'])
feature_value_groupby = item_features_df.groupby(['feature_value_id'])
print('Number of Feature Categories: {}'.format(len(feature_category_groupby)))
print('Number of Feature Values: {}'.format(len(feature_value_groupby)))

    # Number of unique items
item_ids_features = item_features_df['item_id'].unique()
print("The number of unique items is {}: ".format(len(item_ids_features)))

    # Min & max category id
f'Min feature id: {item_features_df["feature_category_id"].min()}, Max feature id: {item_features_df["feature_category_id"].max()}'

    # Check item_features  datatypes and timestampt information 
print('\n ---> The type of each feature item action\n', item_features_df.dtypes)

    # Memory usage
print("\n ---> Memory Usage: \n", item_features_df.info(), "\n")
    
# Descriptive Statistics 
# print('\n ---> Descriptive Statistics\n', item_features_df.describe())
    # as we have categorical variables this is not meaninglfull so we convert the type to object and then describe 
item_features_df.astype('object').describe()

    # Feature Categories distribution over Items
value_categorycount = item_features_df[['item_id', 'feature_category_id']].drop_duplicates().groupby(['item_id']).count()
value_categorycount.rename(columns={"feature_category_id": "feature_category_count"}, inplace=True)
sorted_value_categorycount = value_categorycount['feature_category_count'].sort_values(ascending=False)
print('Max number of feature categories per item: {}'.format(max(value_categorycount['feature_category_count'])))
print('Min number of feature categories per item: {}'.format(min(value_categorycount['feature_category_count'])))
print('Average number of feature categories per item: {}'.format(round(np.average(value_categorycount['feature_category_count']))))
print('Median number of feature categories per item: {}'.format(sorted_value_categorycount[int(len(sorted_value_categorycount) / 2)]))

    # Plot - Frequency items per of feature categories
item_features_df.feature_category_id.value_counts().plot(kind="bar",
                           title="Frequency of each Feature Category",
                           rot=90,
                           xlabel="Feature Category IDs",
                           ylabel="Number of Items per Feature Category",
                           fontsize = 8 ,
                           figsize = (15,5)
                           )
    # Plot - Items distribution over Feature Values
value_itemcount = item_features_df[['item_id', 'feature_value_id']].drop_duplicates().groupby(['feature_value_id']).count()
value_itemcount.rename(columns={"item_id": "item_count"}, inplace=True)
sorted_value_itemcount = value_itemcount['item_count'].sort_values(ascending=False)
sorted_value_itemcount[:100].plot(kind='bar', figsize=(24,7))
plt.xlabel('Feature Value IDs')
plt.ylabel('Number of Items per Feature Value')
plt.show()
    # Plot - Unique Feature Values distribution over Feature Categories
category_valuecount = item_features_df[['feature_category_id', 'feature_value_id']].drop_duplicates().groupby(['feature_category_id']).count()
category_valuecount.rename(columns={"feature_value_id": "feature_value_count"}, inplace=True)
sorted_category_valuecount = category_valuecount['feature_value_count'].sort_values(ascending=False)
sorted_category_valuecount.plot(kind='bar', figsize=(24,7))
plt.xlabel('Feature Category IDs')
plt.ylabel('Number of Distinct Possible Values per Feature Category')
plt.show()

    # Plot - Features category ids
item_features_df.groupby('item_id').feature_category_id.count().plot(kind='pie')
plt.show()

#Check for missing values
print('Item_Features dataset: Number of missing values across columns: \n',item_features_df.isnull().sum())
    # There are no missing records in the dataset.
    
# Checking for duplicated values
print('Checking for duplicated values... \n')
unique_items = item_features_df.item_id.unique()
count_items = len(unique_items)
print('The total number of the unique items in the item features is', count_items) 

###################################### SESSIONS ######################################
print('-'*50,'\n UNDERSTANDING OF THE TRAIN SESSIONS DATASET', '-'*50)

    #Check the number of rows and columns
rows,columns=train_sessions_df.shape
print(' ---> Dataset Dimensions: \n','The Number of Rows are: ',rows,'\n The Number of Columns are: ',columns)
 
    # Preview of the train_sesions dataset 
print('\n ---> A preview of the dataset for: \n', 'The ten first items are: \n', train_sessions_df.head(10), '\n The ten last items: \n', train_sessions_df.tail(10))

    # Columns of train_sessions
print('\n ---> The session actions\n', train_sessions_df.columns)
    # For display purposes: 
trainsessions_columns = PrettyTable(['session_id','item_id','date'])
print('The columns of the item_features dataset are :\n',itemfeatures_columns)

    # Check item_features  datatypes and timestampt information 
print('\n ---> The type of each session action\n', train_sessions_df.dtypes)

    # Memory usage
print("\n ---> Memory Usage: ", train_sessions_df.info(memory_usage="deep"), "\n")

# Descriptive Statistics for the train sessions dataset 
# print('\n ---> Descriptive Statistics\n', train_sessions_df.describe())
    # as we have categorical variables this is not meaninglfull so we convert the type to object and then describe 
train_sessions_df.astype('object').describe()

    # Number of total views, unique sessions and unique items
session_ids_train = train_sessions_df['session_id'].unique()
item_ids_train = train_sessions_df['item_id'].unique()
n_sessions_train = len(session_ids_train)
n_items_train = len(item_ids_train)
print("Number of total views: {}".format(len(train_sessions_df)))
print("Number of unique sessions: {}".format(n_sessions_train))
print("Number of unique items: {}".format(n_items_train))

train_sessions_with_duplicates = train_sessions_df[train_sessions_df.duplicated(['session_id', 'item_id'])]['session_id'].unique()
print(len(train_sessions_with_duplicates)/train_sessions_df['session_id'].nunique() * 100)
# About 28% of the train sessions have multiple views with the same items 

    # Session Lenght
train_sessions_length = train_sessions_df.groupby('session_id').count().reset_index()
train_sessions_length = train_sessions_length.drop('date', 1)
train_sessions_length.columns = ['session_id', 'length']
train_sessions_length = train_sessions_length.sort_values(by='length', ascending=False)
train_sessions_length
train_sessions_length['length'].value_counts().sort_index()
train_sessions_length['length'].value_counts().sort_index().plot(kind='bar', width=0.9, figsize=(20, 10))
plt.xlabel("Session length")
plt.ylabel("Number of sessions")
plt.show()

    # Sessions with more and less views
max_session_itemcount = train_sessions_length['length'].max()
print('Session with the longest number of view is of: ' + str(max_session_itemcount) + ' item')
min_session_itemcount = train_sessions_length['length'].min()
print('Min session length: ' + str(min_session_itemcount) + ' item')

#Check for missing values
print('Train_sessions dataset: Number of missing values across columns: \n',train_sessions_df.isnull().sum())
    # There are no missing records in the dataset.
    
# Checking for duplicated values
print('Checking for duplicated values... \n')
unique_item_sessions = train_sessions_df.item_id.unique()
count_item_sessions = len(unique_item_sessions)
print('The total number of the unique items in the sessions is', count_item_sessions) 

###################################### PURCHASE DATASET ######################################
print('-'*50,'\n UNDERSTANDING OF THE TRAIN PURCHASES DATASET', '-'*50)
#Check the number of rows and columns
rows,columns=train_purchases_df.shape
print(' ---> Dataset Dimensions: \n','The Number of Rows are: ',rows,'\n The Number of Columns are: ',columns)
 
# Preview of the train_purchases dataset 
print('\n ---> A preview of the dataset for: \n', 'The ten first items are: \n', train_purchases_df.head(10), '\n The ten last items: \n', train_purchases_df.tail(10))

# Columns of train_purchases
print('\n ---> The session actions\n', train_purchases_df.columns)
trainpurchases_columns = PrettyTable(['session_id','item_id','date'])
print('The columns of the item_features dataset are :\n',itemfeatures_columns)

# Check item_features  datatypes and timestampt information 
print('\n ---> The type of each session action\n', train_purchases_df.dtypes)

# Memory Usage
print("Memory Usage: ", train_purchases_df.info(memory_usage="deep"), "\n")

# Descriptive Statistics for the item_features dataset 
# print('\n ---> Descriptive Statistics\n', train_purchases_df.describe())
    # as we have categorical variables this is not meaninglfull so we convert the type to object and then describe 
train_purchases_df.astype('object').describe()
# unique value of each column
train_purchases_df.nunique()

    # Number of total purchases, unique sessions and unique items
session_ids_purchases = train_purchases_df['session_id'].unique()
item_ids_purchases = train_purchases_df['item_id'].unique()
n_sessions_purchases = len(session_ids_purchases)
n_items_purchases = len(item_ids_purchases)
print("Number of total purchases: {}".format(len(train_purchases_df)))
print("Number of unique sessions: {}".format(n_sessions_purchases))
print("Number of unique items: {}".format(n_items_purchases))
#Check for missing values
print('Train_purchases dataset: Number of missing values across columns: \n',train_purchases_df.isnull().sum())
    # There are no missing records in the dataset.
    
# Checking for duplicated values
count_item_sessions = train_purchases_df.item_id.unique()
count_item_purchases = len(count_item_sessions)
print('The total number of the unique items in the purchases is', count_item_purchases) 



###################################### SESSIONALITIES ######################################
# Sort sessions by date
train_sessions_df_sorted = train_sessions_df.sort_values('date')
train_sessions_df_sorted

session_ids_sorted = np.sort(train_sessions_df['session_id'].unique())
session_purchase_ids_sorted = np.sort(train_purchases_df['session_id'].unique())
print("Session ids correspond: {}".format((session_ids_sorted == session_purchase_ids_sorted).all()))
session_ids_sorted, session_purchase_ids_sorted
#Every session of train_sessions_df corresponds to a purchased item in train_purchases_d
train_purchases_df_2020 = train_purchases_df[train_purchases_df['date'] <= '2020-12-31 23:59:59']
train_purchases_df_2021 = train_purchases_df[train_purchases_df['date'] > '2020-12-31 23:59:59']
#Number of purchases by time slot
    #2020
purchases_per_period_2020 = train_purchases_df_2020.copy()

purchases_per_period_2020['period'] = [(pd.Timestamp(elem).hour // 4 + 1) for elem in purchases_per_period_2020['date']]
purchases_per_period_2020['period'].replace({1: '00:00 - 03:59',
                                             2: '04:00 - 07:59',
                                             3: '08:00 - 11:59',
                                             4: '12:00 - 15:59',
                                             5: '16:00 - 19:59',
                                             6: '20:00 - 23:59'},
                                            inplace=True)
purchases_per_period_2020

purchases_per_period_2020_count = purchases_per_period_2020.groupby(['period'])['session_id'].nunique()
purchases_per_period_2020_count

plt.figure(figsize=(23, 7))
plt.bar(purchases_per_period_2020_count.index, purchases_per_period_2020_count)
plt.show()

    # 2021
purchases_per_period_2021 = train_purchases_df_2021.copy()

purchases_per_period_2021['period'] = [(pd.Timestamp(elem).hour // 4 + 1) for elem in purchases_per_period_2021['date']]
purchases_per_period_2021['period'].replace({1: '00:00 - 03:59',
                                             2: '04:00 - 07:59',
                                             3: '08:00 - 11:59',
                                             4: '12:00 - 15:59',
                                             5: '16:00 - 19:59',
                                             6: '20:00 - 23:59'},
                                            inplace=True)
purchases_per_period_2021
purchases_per_period_2021_count = purchases_per_period_2021.groupby(['period'])['session_id'].nunique()
purchases_per_period_2021_count

plt.figure(figsize=(23, 7))
plt.bar(purchases_per_period_2021_count.index, purchases_per_period_2021_count)
plt.show()

# Datetime by sesion purchases 
train_purchases_df.rename(columns={'date':'datetime'},inplace=True)
train_purchases_df['datetime']=pd.to_datetime(train_purchases_df['datetime']).dt.date
pd.to_datetime(train_purchases_df['datetime']).describe()
train_purchases_df['datetime'].value_counts().sort_index().plot(kind='line', figsize=(20, 5))


# Correlation Matrixes
# # relationship analysis
#     #correlation matrix
# # item = item_features_df.drop(['item_id'],axis=1)
# item = item_features_df.drop([],axis=1)
# correlation = item.corr()
# sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot = True) 

# sns.pairplot(item)

# random_item_features_df = item_features_df.sample(n=5000)
# random_train_sessions_df = train_sessions_df.sample(n=5000)
# random_train_purchases_df = train_purchases_df.sample(n=5000)

# random_train_purchases_df.select_dtypes(['int64']).corr()
# sns.heatmap(random_train_purchases_df.select_dtypes(['int64']).corr(),annot=True)

# random_train_sessions_df.select_dtypes(['int64']).corr()
# sns.heatmap(random_train_sessions_df.select_dtypes(['int64']).corr(),annot=True)

# pd.crosstab(random_train_sessions_df.date, random_train_purchases_df.date)
# sns.heatmap(random_train_sessions_df.select_dtypes(['int64']).corr(),annot=True)

# as the variables are labels there is no correlation between the values ids






















