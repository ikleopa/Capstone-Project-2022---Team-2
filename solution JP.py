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

df_item_features=df_item_features[:30000]
df_train_purchases = df_train_purchases[:30000]
df_train_sessions = df_train_sessions[:30000]


# Find out how many items there are for each purchase.
# There are many items that have been purchased only once, and it seems difficult to predict such items due to the amount of data.
# It seems that it has been purchased less than once as a volume zone 10^2
# On the other hand, it can be confirmed that a few items occupy more than one purchase.10^3

df_purchases_counts = df_train_purchases["item_id"].value_counts().values
ax = plt.figure(figsize = (16, 8)).add_subplot(111)
ax.hist(x = df_purchases_counts, bins = np.logspace(0, 4, 50), color = 'dodgerblue', alpha = 0.75)
ax.set_xscale('log')
ax.set_xlabel('Number of purchases')
ax.set_ylabel("Number of items")

# Check if there is a tendency for the day of the week or time of purchase
w_list = ["Mon", "Tue", "Wed", "Thr", "Fri", "Sat", "Sun"]
for idx in df_train_purchases.index:
    if idx % 10000 == 0:
        print(idx, " / ", len(df_train_purchases), end = "\r")
    datetime_list = re.findall(r"\d+", df_train_purchases.at[idx, "date"])
    datetime_list = [int(s) for s in datetime_list]
    df_train_purchases.at[idx, "date_as_datetime"] = datetime.datetime(datetime_list[0], datetime_list[1], datetime_list[2],\
                                                                   datetime_list[3], datetime_list[4])
    #df_train_purchases.at[idx, "date_as_datetime"] = datetime.datetime.strptime(df_train_purchases.at[idx, "date"], '%Y-%m-%d %H:%M:%S.')
    df_train_purchases.at[idx, "day_of_week"] = w_list[int(df_train_purchases.at[idx, "date_as_datetime"].weekday())]
    df_train_purchases.at[idx, "hour"] = datetime_list[3]
display(df_train_purchases)

# The result is that there are many Saturdays and Fridays. Wednesdays tend to be less. It's a little surprising that there aren't many Sundays.
df_train_purchases.pivot_table(values = "session_id", index = "hour", columns = "day_of_week", aggfunc = "count").plot(figsize = (18, 6))

# Aggregated by time zone. There is no change in the overall trend on any day of the week. 
# \ There may be differences in products that are easy to order depending on the day of the week and the time of day. Confirmation required.
df_train_purchases.pivot_table(values = "session_id", index = "hour", columns = "day_of_week", aggfunc = "count").plot(figsize = (18, 6))

# Add a time_zoon column that represents the time zone. Classified by morning, afternoon, eveninng, night, midnight.
def hour_to_timezoon(hour):
    if hour < 6.0:
        return "midnight"
    elif hour < 10.0:
        return "morning"
    elif hour < 16.0:
        return "afternoon"
    elif hour < 21.0:
        return "night"
    else:
        return "midnight"
df_train_purchases["time_zoon"] = df_train_purchases["hour"].apply(hour_to_timezoon) # Add column
df_train_purchases

# Input training data (view of item in each session)
    # train_sessions.csv\ columns: session_id, item_id, date
    # \ The items that were viewed in a session. 
    # The "date" column is a timestamp to miliseconds. 
    # A session is equal to a day, so a session is one user's activity on one day. 
    # The session goes up to and not including the first time the user viewed the item that they bought in the end. 
    # The last item in the session will be the last item viewed before viewing the item that they bought. 
    # To find they item they bought link to train_purchases.csv on session_id.

# In each session, a list of products that were viewed until just before the purchase is given as input. 
    # Based on this, guess which product you purchased.

# Check how many products viewed in each session 
df_train_sessions_counts = df_train_sessions.groupby("session_id").count()["item_id"].values
ax = plt.figure(figsize = (16, 8)).add_subplot(111)
ax.hist(x = df_train_sessions_counts, bins = np.logspace(0, 2, 50), color = 'dodgerblue', alpha = 0.75)
ax.set_xscale('log')
ax.set_xlabel('Duration of each session')
ax.set_ylabel("Number of sessions")

print("Average duration period of each session", df_train_sessions_counts.mean())
print("Median duration period of each session ", np.median(df_train_sessions_counts))

# Feature Data
    # item_features.csv columns: item_id, feature_category_id, feature_value_id 
    # The label data of items. A feature_category_id represents an aspect of the item such as "colour", the feature_value_id is the value for that aspect, e.g. "blue". 
    # Some items may not share many feature_cateogry_ids if they different types of items, for example trousers will share almost nothing with shirts. 
    # Even things like colour will not be shared, the colour aspect for trousers and shirts are two different feature_category_ids.

# The features of each item are summarized. 
    # Even for features such as colour, different feature ids are assigned to each product, so it seems unlikely that different products will have a common feature id.

# There seems to be some features that are commonly used. Check the upper features.
df_item_features["feature_category_id"].value_counts()

#Create a co-occurrence matrix df_item_features_pivot of item and features.
# pivot
df_item_features_pivot = df_item_features.pivot_table(values = "feature_value_id", index = "item_id", columns = "feature_category_id", aggfunc = "count")
df_item_features_pivot.fillna(0.0, inplace = True) # fill na with zero 
display(df_item_features_pivot)

# PCA is easier to understand than singular value decomposition, so amended here. If you check carefully, there are features (47, 56) that contain 0 for all items, so be careful.
df_item_features_pca = df_item_features_pivot.apply(lambda x:  (x - x.mean()) / x.std(), axis = 0) 
print("============== Αυτά τα χαρακτηριστικά περιέχουν 0 για όλα τα στοιχεία ==============")
display(df_item_features_pca.loc[:,df_item_features_pca.isnull().any()]) # Ελέγξτε τη θέση των στηλών που είναι 0 για όλα τα στοιχεία
df_item_features_pca.fillna(0.0, inplace = True) # Οι στήλες που είναι 0 για όλα τα στοιχεία δεν θα τυποποιηθούν και θα εμφανιστεί NaN, επομένως είναι απαραίτητο να συμπληρώσετε ξανά αυτήν τη χρονική στιγμή.

pca = PCA()
pca.fit(df_item_features_pca)
feature = pca.transform(df_item_features_pca)
print("=============THE RESULTS OF PCA FOR ALL THE ITEMS ARE: ==============")
df_item_features_pca = pd.DataFrame(feature, columns = ["PC{}".format(x + 1) for x in range(len(df_item_features_pca.columns))], index = df_item_features_pivot.index)
display(df_item_features_pca)
# ============== It is these features that contain 0 for all items ==============

# Let's plot items for some principal components. It seems to be divided into good feelings as it is. It may be better to drop items into 4 to 5 categories.
pca_pair_list = [["PC1", "PC2"], ["PC2", "PC3"], ["PC1", "PC3"]]

fig = plt.figure(figsize = (24, 8))

for idx, pair in enumerate(pca_pair_list):
    pred = KMeans(n_clusters = 4).fit_predict(df_item_features_pca[pair].values) # Clustering the results dropped in two dimensions.
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

# Check how the features of PC1 and PC2 are effective. Can this be used to classify features? (I don't know how to use it)
pred = KMeans(n_clusters = 4).fit_predict(pca.components_[[0,1]].T) # Clustering the results dropped in two dimensions.

ax = plt.figure(figsize = (12, 12)).add_subplot(1, 1, 1)

#Plot the contribution of the observed variables in the first and second principal components
for x, y, name in zip(pca.components_[0], pca.components_[1], df_item_features_pivot.columns.values):
    ax.text(x, y, name, fontsize = 14)
ax.scatter(pca.components_[0], pca.components_[1], alpha=0.5, c = pred, s = 100.0)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

# Correlation analysis with correct label
# Relationship with time
# Investigate whether there is a tendency for purchased products by time zone or day of the week. 
# However, note that the actual prediction does not give the "at the time of purchase" time, so it is necessary to approximate it with the view immediately before the purchase (probably).


# Aggregate by day of the week:
day_of_week_list = ["Mon", "Tue", "Wed", "Thr", "Fri", "Sat", "Sun"]
df_train_purchases_pivot = df_train_purchases.pivot_table(values = "session_id", index = "item_id", columns = "day_of_week", aggfunc = "count")
df_train_purchases_pivot = df_train_purchases_pivot.reindex(columns = day_of_week_list).fillna(0).copy()

# The ranking of the products that are often bought and the number of times are summarized for each day of the week.
# Table to summarize in ranking format
df_rank = pd.DataFrame(index = range(1, len(df_train_purchases_pivot)+1)) # Need to add +1 to rank
for dow in day_of_week_list:
    df_rank[dow + "_item"] = df_train_purchases_pivot[dow].sort_values(ascending = False).index
    # The number of times can also be summarized as a count, but it is difficult to see, so comment out
    df_rank[dow + "_count"] = df_train_purchases_pivot[dow].sort_values(ascending = False).values.astype("int")
df_rank["all_item"] = df_train_purchases_pivot.sum(axis = 1).sort_values(ascending = False).index
df_rank["all_count"] = df_train_purchases_pivot.sum(axis = 1).sort_values(ascending = False).values.astype("int")
df_rank.head(10).to_excel("rank_every.xlsx")
df_rank.head(10)

# Aggregate by time zone
time_zoon_list = ["morning", "afternoon", "night", "midnight"]
df_train_purchases_pivot = df_train_purchases.pivot_table(values = "session_id", index = "item_id", columns = "time_zoon", aggfunc = "count")
df_train_purchases_pivot = df_train_purchases_pivot.reindex(columns = time_zoon_list).fillna(0).copy()
# Table to summarize in ranking format
df_rank = pd.DataFrame(index = range(1, len(df_train_purchases_pivot)+1)) # Need to add +1 to rank
for time_zoon in time_zoon_list:
    df_rank[time_zoon + "_item"] = df_train_purchases_pivot[time_zoon].sort_values(ascending = False).index
    # The number of times can also be summarized as a count, but it is difficult to see, so comment out
    df_rank[time_zoon + "_count"] = df_train_purchases_pivot[time_zoon].sort_values(ascending = False).values.astype("int")
df_rank["all_item"] = df_train_purchases_pivot.sum(axis = 1).sort_values(ascending = False).index
df_rank["all_count"] = df_train_purchases_pivot.sum(axis = 1).sort_values(ascending = False).values.astype("int")
df_rank.head(10).to_excel("rank_ by_timezone.xlsx")
df_rank.head(10)


# Naive Model
df_test_leaderboard = pd.read_csv(path + r"\test_leaderboard_sessions.csv")
display(df_test_leaderboard)

    # Output the standings as they are
# Create a standings
num_prediction_list = 100 # Output up to 100th place
df_rank_table = pd.DataFrame(columns = ["item_id", "rank"])
df_rank_table["item_id"] = df_rank["all_item"].values[0:num_prediction_list]
df_rank_table["rank"] = range(1, num_prediction_list+1, 1)
df_rank_table

# It seems that pandas does not have cross_join (seriously?)
def cross_join(df_a, df_b, common_key=None):
    # Note that it will not work if you have the same column name
    df_a['tmp'] = 1
    df_b['tmp'] = 1
    return pd.merge(df_a, df_b, how='outer').drop("tmp", axis = 1)

# Create a function to predict based on the standings
def predict_by_rank_table(df_test):
    session_id_list = df_test["session_id"].unique()
    df_pred = pd.DataFrame(columns = ["session_id"])
    df_pred["session_id"] = session_id_list
    return cross_join(df_pred, df_rank_table)

df_submit = predict_by_rank_table(df_test_leaderboard)
path_submit = r'C:/Users/Iris/Desktop/Deree/Master/CAPSTONE/Recommender System/dressipi_recsys2022/'
df_submit.to_csv(path_submit + r"\rank_table.csv", index = False)
df_submit # The score is 0.01626522346478938 and the ranking is 1,101.

# Define distance based on PCA
    # Since item_id should not be the answer, find the closest other one.
df_item_features_pca

# A function that defines the distance in the principal component space and lists the closest ones.
def search_nearest_neighborhood_by_pca(item_id_list, num_prediction_list = 100):
    mean_vec = df_item_features_pca.loc[item_id_list].mean(axis = 0).values # Calculate the centroid of the coordinates in the principal component space for each item in the view
    # This is overwhelmingly faster than the lambda expression.
    df_item_features_pca_minus_mean = df_item_features_pca - mean_vec # Calculate how far apart in the principal component space
    df_item_features_pca_distance = df_item_features_pca_minus_mean * df_item_features_pca_minus_mean # Square each component
    # Calculate the squared distance by summing the principal components. By the way, sort.
    df_item_features_pca_distance = df_item_features_pca_distance.sum(axis = 1).to_frame().sort_values(by = 0, ascending = True)
    df_item_features_pca_distance.drop(index = item_id_list, inplace = True) # What is in the view should not be the answer, so drop it
    return df_item_features_pca_distance[0: num_prediction_list].index.values


# Prediction based on distance in principal space
def predict_by_pca(df_test):
    df_pred = pd.DataFrame(columns = ["session_id", "item_id", "rank"])
    session_id_list = df_test["session_id"].unique()
    #df_pred["session_id"] = pd.Series(session_id_list).repeat(num_prediction_list) # It seems that it will take time to connect vertically, so first make an empty container
    
    df_pred = pd.DataFrame(columns = ["session_id", "item_id", "rank"], index = range(len(session_id_list)*num_prediction_list))
    df_pred["session_id"] = np.repeat(session_id_list, num_prediction_list, axis = 0)
    df_pred["rank"] = np.tile(range(1, num_prediction_list), len(session_id_list))
    item_id_values = np.zeros(len(df_pred))
    
    # print(session_id_list)
    for idx, session_id in enumerate(session_id_list):
        if idx % 100 == 0:
            print(idx, "/", len(session_id_list), end = "\r")
        item_id_list = df_test[df_test["session_id"] == session_id]["item_id"].values
        pred_list = search_nearest_neighborhood_by_pca(item_id_list)
        
        item_id_values[idx*num_prediction_list: (idx+1)*num_prediction_list] = np.array(pred_list)
        # df_pred.loc[idx*num_prediction_list:(idx+1)*num_prediction_list, "item_id"] = pred_list
        # df_pred.loc[idx*num_prediction_list:(idx+1)*num_prediction_list-1, "rank"] = range(num_prediction_list)
        
        # df_pred.loc[df_pred["session_id"] == session_id, "item_id"] = pred_list
        # df_pred.loc[df_pred["session_id"] == session_id, "rank"] = range(1, num_prediction_list+1, 1)
        
    df_pred["item_id"] = item_id_values
    df_pred["item_id"] = df_pred["item_id"].astype("int")
    return df_pred


df_submit = predict_by_pca(df_test_leaderboard)
# df_submit["rank"] = df_submit["rank"] + 1

path_submit = r'C:/Users/Iris/Desktop/Deree/Master/CAPSTONE/Recommender System/dressipi_recsys2022/'
df_submit.to_csv(path_submit + r"\nearest_neighborhood_by_pca.csv", index = False)

path_submit = r'C:/Users/Iris/Desktop/Deree/Master/CAPSTONE/Recommender System/dressipi_recsys2022/'
df_submit.to_csv(path_submit + r"\nearest_neighborhood_by_pca.csv", index = False)
display(df_submit)

