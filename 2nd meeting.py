##############################################################################
#####                        IMPORT PACKAGES                             #####
##############################################################################
import csv
import json
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # basic plotting
import seaborn as sns # for prettier plots
from sklearn.model_selection import train_test_split, cross_val_score

# models
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

##############################################################################
#####                        READ DATA                             #####
##############################################################################
global_path = ""  #Change value to the path of your desired directory it is  important to end the path with a '/'
if global_path == "":
    path = 'C:/Users/Iris/Desktop/Deree/Master/CAPSTONE/Recommender System/dressipi_recsys2022/'    
else :
    path = global_path    
store_Results = path
# Encoding
enc = 'ISO-8859-15'

# Read the csv files and import them to a DataFrame 
path_item_features = path + 'item_features.csv'
read_item_features = pd.read_csv(path_item_features, encoding = enc)
df_item_features= pd.DataFrame(read_item_features)
df_item_features.columns = df_item_features.columns.str.strip().str.lower().str.replace('-', '_')

df_train_sessions = path + 'train_sessions.csv'
df_train_sessions = pd.read_csv(df_train_sessions, encoding = enc)
df_train_sessions= pd.DataFrame(df_train_sessions)
df_train_sessions.columns = df_train_sessions.columns.str.strip().str.lower().str.replace('-', '_')

df_train_purchases = path + 'train_purchases.csv'
df_train_purchases = pd.read_csv(df_train_purchases, encoding = enc)
df_train_purchases= pd.DataFrame(df_train_purchases)
df_train_purchases.columns = df_train_purchases.columns.str.strip().str.lower().str.replace('-', '_')

# Sample of the datasets 
df_item_features_small=df_item_features[:10000]
# df_item_features_small=df_item_features[:10000]
# df_item_features_small=df_item_features[:10000]

#########################################################################################
##################### Data Preparation / Description #####################
#########################################################################################

# DATA DESCRIPTION 
# Preview of the item_features dataset 
print('A preview of the item_features dataset\n', df_item_features.head(), '\n'*2)
# Preview of the item_features dataset 
print('A preview of the train_sessions dataset\n', df_train_sessions.head(), '\n'*2)
# Preview of the item_features dataset 
print('A preview of the train_purchased dataset\n', df_train_purchases.head())

    # Describe the dataset -- Descriptive Statistics
print("Descriptive Statistics for Items ")
print(df_item_features.describe(), '\n'*2)
print("Useful ingormation for Items ")
print(df_item_features.info())
print("--------------------------------------------------------------")

print("Descriptive Statistics for Sessions ")
print(df_train_sessions.describe(),'\n'*2)
print("Useful ingormation for Items ")
print(df_train_sessions.info())

print("Descriptive Statistics for Purchases ")
print(df_train_purchases.describe(),'\n'*2)
print("Useful ingormation for Items ")
print(df_train_purchases.info())

# Memory Requirements - Information about size
print("Memory Requirements for Dataframe of Items ")
df_item_features.info(memory_usage='deep')
print('\n'*2)
print("Memory Requirements for Dataframe of Sessions")
df_train_sessions.info(memory_usage='deep')
print('\n'*2)
print("Memory Requirements for Dataframe of Purchases")
df_train_purchases.info(memory_usage='deep')
print("--------------------------------------------------------------")

# Check for the number of rows and columns
# Item Features dataset 
rows_item,columns_item=df_item_features.shape
print('Number of rows in the item_features dataset: ',rows_item)
print('Number of columns in the item_features dataset: ',columns_item)
print('\n')
# Train Sessions dataset 
rows_sessions,columns_sessions = df_train_sessions.shape
print('Number of rows in the train_session dataset: ',rows_sessions)
print('Number of columns in the train_session dataset: ',columns_sessions)
print('\n')
# Train Purchased dataset 
rows_purchased,columns_purchased = df_train_purchases.shape
print('Number of rows in the train_purchased dataset: ',rows_purchased)
print('Number of columns in the train_purchased dataset: ',columns_purchased)


# Checking for unique entries
# Investigating item features table
    # Checking for unique items
uniq_items = df_item_features.item_id.nunique()
all_items = df_item_features.item_id.count()
print(f'No. of unique user_id entries: {uniq_items} | Total user_id entries: {all_items}')
print("--------------------------------------------------------------")

# Preview of the item_features dataset 
print('A preview of the dataset\n', df_item_features.head())
# Checking for unique session entries
uniq_sessions = df_train_sessions.session_id.nunique()
all_sessions = df_train_sessions.session_id.count()
print(f'No. of unique books: {uniq_sessions} | All book entries: {all_sessions}')
print("--------------------------------------------------------------")

# Checking for unique purchased entries
# checking for duplicates
uniq_purchased = df_train_purchases.session_id.nunique()
all_purchased  = df_train_purchases.session_id.count()
print(f'No. of unique books: {uniq_purchased} | All book entries: {all_purchased}')
print("--------------------------------------------------------------")

# Check datatypes and timestampt information
print('The type of each columns is: \n', df_item_features.dtypes,'\n'*2 )
print('The type of each columns is: \n', df_train_sessions.dtypes, '\n'*2 )
print('The type of each columns is: \n', df_train_purchases.dtypes, '\n'*2)

################################################################################################

#Number of features per item
item_features=df_item_features.groupby('item_id')['feature_category_id'].count()
print("The number of features per item is: \n", item_features)

#Number of features values per feature distinct
features=df_item_features.groupby('feature_category_id')['feature_value_id'].nunique()
print("The number of features values per feature distinct is: \n", features)

# number of items per cat 
x=df_item_features.groupby(['feature_category_id']).count()
x=x.sort_values(by='item_id',ascending=False)
x=x.iloc[0:10].reset_index()
print("The number of items per category is: ",x)

# #plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.feature_category_id, x.item_id, alpha=0.8)
plt.title("Items per Category")
plt.ylabel('# of items', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()

#histograms
    # For item_features: 
df_item_features.isnull().values.any()
df_item_features.hist(figsize=(20, 15))  
    # For train_sessions: 
df_train_sessions.isnull().values.any()
df_train_sessions.hist(figsize=(20, 15))
    # For train_purchased: 
df_train_purchases.isnull().values.any()
df_train_purchases.hist(figsize=(20, 15))

# Frequency of item features 
df_item_features.item_id.value_counts()
# Frequency of feature category id 
df_item_features.feature_category_id.value_counts()
# Frequency oga feature value id 
df_item_features.feature_value_id.value_counts()

# Frequency of items plot
viz_purpose_sample_df_item_features= df_item_features[:500]
plt.rc("font", size=15)
viz_purpose_sample_df_item_features['item_id'].value_counts(sort=False).plot(kind='bar')
plt.title('Items Distribution\n')
plt.xlabel('Items')
plt.ylabel('Count')
plt.savefig('test13.png', bbox_inches='tight')
plt.show()

# Frequency of sessions plot
# Sessions 
viz_purpose_sample_df_train_sessions= df_train_sessions[:500]
plt.rc("font", size=15)
viz_purpose_sample_df_train_sessions['session_id'].value_counts(sort=False).plot(kind='bar')
plt.title('Sessions Distribution\n')
plt.xlabel('Sessions')
plt.ylabel('Count')
plt.savefig('test1.png', bbox_inches='tight')
plt.show()


#Check for missing values
print('Item_Features dataset: Number of missing values across columns: \n',df_item_features.isnull().sum())
    # There are no missing records in the dataset.
print('Train_sessions dataset: Number of missing values across columns: \n',df_train_sessions.isnull().sum())
    # There are no missing records in the dataset.
print('Train_purchases dataset: Number of missing values across columns: \n',df_train_purchases.isnull().sum())

# Checking for duplicated values
print('Checking for duplicated values... \n')

    # item_features:
unique_items = df_item_features.item_id.unique()
count_items = len(unique_items)
print('The total number of the unique item IDs is', count_items) 
    # train_sessions
unique_sessions = df_train_sessions.item_id.unique()
count_sessions = len(unique_sessions)
print('The total number of the unique session IDs is', count_sessions) 
    # train_purchases
unique_purchases = df_train_purchases.item_id.unique()
count_purchases = len(unique_purchases)
print('The total number of the unique purchase IDs is', count_purchases) 

####################################################################################

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

