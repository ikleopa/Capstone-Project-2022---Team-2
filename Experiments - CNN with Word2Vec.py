# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:02:29 2022

@author: Group 2
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score
import tensorflow_ranking as tfr
import tensorflow as tf
from keras.utils.vis_utils import plot_model
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


######################################################################################
######Using Word2Vector to tranform items into vectors to be used for Experiments#####
######################################################################################
#Import a pre prepared excel file that contains the sequence of items in session
whole=pd.read_excel('Whole.xlsx', index_col=0)  

##Create a corpus from the items in sessions
corpus = []
for col in whole.whole:
   word_list = col.split(",")
   corpus.append(word_list)

#######Build and train Word2Vec model###############
#######The vector size is very import in case dimensionality is important#####
model = Word2Vec(window = 5, sg = 1, hs = 0, vector_size=300,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

model.build_vocab(corpus, progress_per=200)

model.train(corpus, total_examples = model.corpus_count, 
            epochs=10, report_delay=1)

########Take the item_ids and their vectors####################
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

#Merge sessions with the item vectors#######################
test=pd.merge(sessions,df2,on='item_id',how='inner')
test.sort_values(by=['session_id'], inplace=True)
test.drop('date', axis=1, inplace=True)

#Take a substet
test2=test[:2000]

test2=pd.merge(purchases,test2,on='session_id',how='inner') 
test2.drop('date', axis=1, inplace=True)
test2.drop('item_id_x', axis=1, inplace=True)

######################################################################################################
###################Iterate the sessions and combine the items of a sessions in one row################
###################Perform padding adding 0s until the max length is reached##########################
######################################################################################################
session=-1
session2=-1
dic={}
for i,r in test2.iterrows():
    session=test2.iat[i,0]
    if session==-1 or session!=session2:
        session2=test2.iat[i,0] 
        tempList=[]
        for j in range(1,test2.columns.size):
            tempList.append(test2.iat[i,j])
        tempList.pop(0)
        dic[session]=tempList
    else:   
        tempList=[]
        for j in range(1,test2.columns.size):
            tempList.append(test2.iat[i,j])
        tempList.pop(0)
        dic[session].extend(tempList)

##############Covert Dictionary to dataframe################
temp= pd.DataFrame.from_dict(dic,orient='Index')
temp=temp.fillna(0)
temp.reset_index(inplace=True)
temp.rename(columns = {'index':'session_id'}, inplace = True)

######Create the final dataframe to be used for experiments##########
######Contains session_id, purchased item and the vectors of each item in the session###############
final=pd.merge(purchases,temp,on='session_id',how='inner') 
final.drop('date', axis=1, inplace=True)
final.rename(columns = {'item_id':'item_purched'}, inplace = True)

######################################################################################################
#########################train_test_split the data to perform the experiments#########################
######################################################################################################
Y=final['item_purched']
#Onehotencoding the target item in case it should be used instead of the item id
Yonehot = pd.get_dummies(Y) 
final.drop('item_purched', axis=1, inplace=True)

X=final.iloc[:,5:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Yonehot,test_size=0.2)

########Convert X,Y variables to numpy
X_train_np = X_train.to_numpy(dtype='float', copy = True)
X_test_np =  X_test.to_numpy(dtype='float', copy = True)
y_train_np = Y_train.to_numpy(copy=True)
y_test_np = Y_test.to_numpy(copy=True)

#######Save X,Y variables to be used for our experiments
# np.save('X_train_np.npy',X_train_np)
# np.save('X_test_np.npy',X_test_np)
# np.save('y_train_np.npy',y_train_np)
# np.save('y_test_np.npy',y_test_np)

#######Import X,Y variables to be used for our experiments
import numpy as np
X_train_np = np.load('X_train_np.npy')
X_test_np =  np.load('X_test_np.npy')
y_train_np = np.load('y_train_np.npy')
y_test_np = np.load('y_test_np.npy')

######################################################################################################
######################################Experiments#####################################################
######################################################################################################
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import Dropout

########Early Stopping and saving model and its log################################################
callbacks = [
    tf.keras.callbacks.ModelCheckpoint( 'model.h5', save_best_only=True, verbose=1),
    tf.keras.callbacks.CSVLogger('training.log'),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=20, verbose=1,
                                     mode="min")]

################Initiate and fit the model#########################################################
model = Sequential()
#model.add(LSTM(24,input_shape=(35,100),return_sequences=True))
#model.add(LSTM(24,return_sequences=False))
model.add(Conv1D(32, kernel_size=(10), input_shape=(1160,1), activation='relu')) # Input based on the X_train_np dimension
model.add(Conv1D(64, kernel_size=(5), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=4,input_shape=(696,)))
model.add(Flatten())
model.add(Dense(200,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.2))
#model.add(tf.keras.layers.Dense(100,kernel_initializer='normal',activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(15368,kernel_initializer='normal',activation='softmax')) # Output based on the Y_train_np dimension
model.compile(loss="categorical_crossentropy",optimizer ='adam',metrics=['accuracy'])
history=model.fit(X_train_np,y_train_np,epochs=40,batch_size=20,validation_split=0.25,verbose=1, callbacks=callbacks)
scores = model.evaluate(X_train_np,y_train_np,verbose=1,batch_size=20)
print('Accurracy: {}'.format(scores[1])) 


predict=model.predict(X_test_np)
predict = np.argmax(predict, axis=1)
y_test_np = np.argmax(y_test_np, axis=1)


######################################################################################################
######################################Results#########################################################
######################################################################################################
f1 = f1_score(y_test_np, predict, average='macro')
p = precision_score(y_test_np, predict, average='macro')
a = accuracy_score(y_test_np, predict)

print("f1: " + str(f1))
print("p: " + str(p))
print("a: " + str(a))

cnn_mrr = tfr.keras.metrics.MRRMetric()
cnn_mrr(y_test_np.reshape(-1,1), predict.reshape(-1,1)).numpy()
mrr=cnn_mrr(y_test_np.reshape(-1,1), predict.reshape(-1,1))

print("MRR: " +str(mrr))

#######Plot the loss value
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Convolutional Neural Network loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#######Plot the accuracy value
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Convolutional Neural Network Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#########Show model structure
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)




print(model.wv.most_similar('11529', topn=100))

check=df2.query('item_id == 4028')

