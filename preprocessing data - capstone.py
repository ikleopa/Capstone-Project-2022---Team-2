
import pandas as pd
# import re

# Vizualitaion
import matplotlib.pyplot as plt # basic plotting
import seaborn as sns # for prettier plots

# importing one hot encoder 
# from sklearn.preprocessing import OneHotEncoder

# Read the datasets
df_item_features = pd.read_csv(r'C:\Users\Iris\Desktop\Deree\Master\CAPSTONE\Recommender System\dressipi_recsys2022\item_features.csv')
df_candidate_items = pd.read_csv(r'C:\Users\Iris\Desktop\Deree\Master\CAPSTONE\Recommender System\dressipi_recsys2022\candidate_items.csv')

df_item_features.info()
#Check the number of rows and columns
rows,columns=df_item_features.shape
print('Number of rows: ',rows)
print('Number of columns: ',columns)

# Preview of the item_features dataset 
print('A preview of the dataset\n', df_item_features.head())
    # to get a preview of the data
# Preview of the candidate_items dataset 
print('A preview of the dataset\n', df_candidate_items.head())
    # to get a preview of the data

# Columns of item_features
print('The session actions\n', df_item_features.columns)
    # the info for each session
# Columns of candidate_features
print('The session actions\n', df_candidate_items.columns)
    # the info for each session
    
# Dimensions of the item_features dataset
print('The dimensions of the dataset are', df_item_features.shape)
# Dimensions of the candidate_items dataset
print('The dimensions of the dataset are', df_candidate_items.shape)
  
# Check datatypes and timestampt information for both datasets
print('The type of each session action\n', df_item_features.dtypes)
print('The type of each session action\n', df_candidate_items.dtypes)

# Descriptive Statistics for the item_features dataset 
print("Descriptive Statistics\n", df_item_features.describe())
# Descriptive Statistics for the candidate_items dataset 
print("Descriptive Statistics\n", df_candidate_items.describe())

#Number of features per item
item_features=df_item_features.groupby('item_id')['feature_category_id'].count()

#Number of features values per feature distinct
item_features2=df_item_features.groupby('feature_category_id')['feature_value_id'].nunique()

# number of items per cat 
x=df_item_features.groupby(['feature_category_id']).count()
x=x.sort_values(by='item_id',ascending=False)
x=x.iloc[0:10].reset_index()
x
# #plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.feature_category_id, x.item_id, alpha=0.8)
plt.title("Items per Category")
plt.ylabel('# of items', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()

#hist
df_item_features.isnull().values.any()
df_item_features.hist(figsize=(20, 15))

# Frequency of item features 
df_item_features.item_id.value_counts()

# Frequency of feature category id 
df_item_features.feature_category_id.value_counts()

# Frequency oga feature value id 
df_item_features.feature_value_id.value_counts()

#Check for missing values
print('Number of missing values across columns: \n',df_item_features.isnull().sum())
    # There are no missing records in the dataset.
# Number of duplicate items
df_item_features.shape[0] - df_item_features['item_id'].nunique()

# Checking for duplicated values
print('Checking for duplicated values...')

unique_items = df_item_features.item_id.unique()
count_items = len(unique_items)
print('The total number of the unique item IDs is', count_items) 
 
# ____________________________________________________________________________________ #

df_train_purchases = pd.read_csv(r"C:/Users/Iris/Desktop/Deree/Master/CAPSTONE/Recommender System/dressipi_recsys2022/train_purchases.csv") 
df_train_purchases.shape[0] - df_train_purchases['session_id'].unique()
# Number of unique values
len((df_item_features['feature_category_id'].unique()))
# 
len((df_item_features['feature_value_id'].unique()))

#Check for missing values
print('Number of missing values across columns: \n',df_train_purchases.isnull().sum())
    # There are no missing records in the dataset.
    
sample_df_train_purchases= df_train_purchases[:1000]

df_train_sessions = pd.read_csv(r"C:/Users/Iris/Desktop/Deree/Master/CAPSTONE/Recommender System/dressipi_recsys2022/train_sessions.csv")
sample_df_train_sessions= df_train_sessions[:1000]

#Check for missing values
print('Number of missing values across columns: \n',df_train_sessions.isnull().sum())
    #There are no missing records in the dataset.

# for the train purchases dataset: 
# Number of unique users
print('Number of unique users in Raw data = ', df_train_purchases['item_id'].nunique())
# Number of unique session id  in the data
print('Number of unique product in Raw data = ', df_train_purchases['session_id'].nunique())

# for the train sessions dataset:
# Number of unique users
print('Number of unique users in Raw data = ', df_train_sessions['item_id'].nunique())
# Number of unique session id  in the data
print('Number of unique product in Raw data = ', df_train_sessions['session_id'].nunique())



# # Pivot table
result = df_item_features.pivot_table(index=['item_id'], columns='feature_category_id',
values='feature_value_id').reset_index() 


 # Merge 
# df = pd.merge(df_item_features, sample_df_train_sessions, how = 'left', on = 'item_id')
result1=sample_df_train_sessions.merge(result, how = 'left', on='item_id')

df_item_features['feature_value_id']

#Summary statistics of rating variable
df_train_purchases['date'].describe().transpose()


# Sessions 
viz_purpose_sample_df_train_sessions= df_train_sessions[:500]
plt.rc("font", size=15)
viz_purpose_sample_df_train_sessions['session_id'].value_counts(sort=False).plot(kind='bar')
plt.title('Sessions Distribution\n')
plt.xlabel('Sessions')
plt.ylabel('Count')
plt.savefig('test1.png', bbox_inches='tight')
plt.show()





