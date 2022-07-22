# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 19:03:05 2022

@author: Group 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prince
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.style as style
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import KElbowVisualizer

# read datasets
global_path = ""  #Change value to the path of your desired directory it is  important to end the path with a '/'
if global_path == "":
    path = 'C:/Users/30694/Desktop/Capstone/Dataset_dressipi_recsys2022/'    
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


pivot_features = features.pivot_table(index=['item_id'], columns='feature_category_id', 
                                                    values='feature_value_id').reset_index()    


pivot_features=pivot_features.fillna(0)
pivot_features=pivot_features.set_index('item_id')

#Scaling using min-max feature scaling

df_min_max_scaled = pivot_features.copy()
  
# apply normalization techniques
for column in df_min_max_scaled.columns:
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    
  
# view normalized data
print(df_min_max_scaled)



#pivot_features=pivot_features.astype('category')
# Categorical PCA
mca=prince.PCA(n_components=8, n_iter=3, copy=True, check_input=True, engine='auto')
mca = mca.fit(pivot_features)

mca_transformed = mca.transform(df_min_max_scaled)

mca.eigenvalues_
mca.total_inertia_
mca.explained_inertia_

##########ELBOW##################################################
# K-means using Euclidean distance and distortion concept
distortions = []
#total number of clusters
K = range(1,25)
#for every cluster value we calculate distortion  
#Use siloutte coefficience instead of euclidean#######################
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(mca_transformed)
    distortions.append(sum(np.min(cdist(mca_transformed, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / mca_transformed.shape[0])

# Now that we have all the distortions we will plot the graph
style.use("fivethirtyeight")
plt.plot(K, distortions, 'bx-')
plt.xlabel('k-clusters')
plt.ylabel('Distortion')
plt.xlim([0, 25])
# plt.ylim([0, 3]) can set the y limit accordingly if and when needed otherwise it sets default value depending on graph
plt.title('Elbow method with Euclidean Distance')
plt.show()


#############Silouette######################
model = KMeans()

visualizer = KElbowVisualizer(model, k=(4,12), metric='silhouette', timings=False)
visualizer.fit(mca_transformed)        # Fit the data to the visualizer
visualizer.show() 






