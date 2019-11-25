#!/usr/bin/env python
# coding: utf-8

# In[1]:


# set of imports
import numpy as np
import seaborn as sns
import os

import zipfile
import json
import re
import operator
import requests
import io
import datetime
import pandas as pd
from pandas.io.json import json_normalize  

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# In[2]:


sessions_folder = ['manual_sessions']
#sessions_folder = ['sessions']

folder_items = sorted(os.listdir("manual_sessions"))
sessions = [sessions_folder[0] +'/'+s for s in folder_items if s.endswith('.zip') ]


# In[3]:



dfALL = pd.DataFrame() # Dataframe with all summarised data
dfAnn = pd.DataFrame() # Dataframe containing the annotations

# for each session in the list of sessions
for s in sessions:
    
    #1. Reading data from zip file
    with zipfile.ZipFile(s) as z:
        
        # get current absolute time in seconds. This is necessary to add the delta correctly
        for info in z.infolist():
            file_datetime = datetime.datetime(*info.date_time)
        current_time_offset = pd.to_datetime(pd.to_datetime(file_datetime, format='%H:%M:%S.%f'),unit='s')
        
         # First look for annotation.json
        for filename in z.namelist():
            
            if not os.path.isdir(filename):
                
                if '.json' in filename:
                    
                    with z.open(filename) as f:
                         data = json.load(f) 
                    # if it has the 'intervals' then then it is an annotatation file 
                    
                    if 'intervals' in data:
                        
                        # concatenate the data with the intervals normalized and drop attribute 'intervals'
                        df = pd.concat([pd.DataFrame(data), 
                            json_normalize(data['intervals'])], 
                            axis=1).drop('intervals', 1)
                        
                        # convert to numeric (when reading from JSON it converts into object in the pandas DF)
                        # with the parameter 'ignore' it will skip all the non-numerical fields 
                        df = df.apply(pd.to_numeric, errors='ignore')
                        
                        # remove the prefix 'annotations.' from the column names
                        df.columns = df.columns.str.replace("annotations.", "")
                        
                        # from string to timedelta + offset
                        df.start = pd.to_timedelta(df.start) + current_time_offset
                        
                        # from string to timedelta + offset
                        df.end = pd.to_timedelta(df.end) + current_time_offset
                        
                        # duration as subtractions of delta in seconds
                        df['duration'] = (df.end-df.start) / np.timedelta64(1, 's')   
                        
                        # append this dataframe to the dataframe annotations
                        dfAnn = dfAnn.append(df) 
                    # if it has 'frames' then it is a sensor file 
                    elif 'frames' in data:
                        
                        # concatenate the data with the intervals normalized and drop attribute 'frames'
                        df = pd.concat([pd.DataFrame(data), 
                            json_normalize(data['frames'])], 
                            axis=1).drop('frames', 1)
                        
                        # remove underscore from columnfile e.g. 3_Ankle_Left_X becomes 3AnkleLeftX
                        df.columns = df.columns.str.replace("_", "")
                        
                        # from string to timedelta + offset
                        df['frameStamp']= pd.to_timedelta(df['frameStamp']) + current_time_offset
                        
                        # retrieve the applicaiton name
                        appName = df.applicationName.all()
                        
                        # remove the prefix 'frameAttributes.' from the column names
                        df.columns = df.columns.str.replace("frameAttributes", df.applicationName.all())
                        
                        # set the timestamp as index 
                        df = df.set_index('frameStamp').iloc[:,2:]
                        
                        # exclude duplicates (taking the first occurence in case of duplicates)
                        df = df[~df.index.duplicated(keep='first')]
                        
                        # convert to numeric (when reading from JSON it converts into object in the pandas DF)
                        # with the parameter 'ignore' it will skip all the non-numerical fields 
                        df = df.apply(pd.to_numeric, errors='ignore')
                        
                        # Keep the numeric types only (categorical data are not supported now)
                        df = df.select_dtypes(include=['float64','int64'])
                        
                        # Remove columns in which the sum of attributes is 0 (meaning there the information is 0)
                        df = df.loc[:, (df.sum(axis=0) != 0)]
                        
                        # The application KienctReader can track up to 6 people, whose attributes are 
                        # 1ShoulderLeftX or 3AnkleRightY. We get rid of this numbers assuming there is only 1 user
                        # This part has to be rethinked in case of 2 users
                        df.rename(columns=lambda x: re.sub('KinectReader.\d','KinectReader.',x),inplace=True)
                        df.rename(columns=lambda x: re.sub('Kinect.\d','Kinect.',x),inplace=True)

                        # Concate this dataframe in the dfALL and then sort dfALL by index
                        dfALL = pd.concat([dfALL, df], ignore_index=False,sort=False).sort_index()

    
df1 =  dfALL.apply(pd.to_numeric).fillna(method='bfill')


# Exclude irrelevant attributes 
to_exclude = ['Ankle','Hip','Hand']
for el in to_exclude:
    df1 = df1[[col for col in df1.columns if el not in col]]
masked_df = [
    df1[(df2_start <= df1.index) & (df1.index <= df2_end)]
    for df2_start, df2_end in zip(dfAnn['start'], dfAnn['end'])
]


# In[5]:


# Calculate the longest interval (chest-compression)
interval_max = 0
for dt in masked_df:
    delta = np.timedelta64(dt.index[-1]-dt.index[0],'ms')/np.timedelta64(1, 'ms')
    if delta > interval_max:
        interval_max = delta
        #KinectElbowLeftX = dt.iloc[:,0] #example 
resample_rate = 40#interval_max/80


df_resampled = [dt.resample(str(resample_rate)+'ms').first() if not dt.empty else None for dt in masked_df]
bin_size = 8
#(bin_size = len(max(df_resampled, key=len))+len(min(df_resampled, key=len)))/2
# create a dummy ndarray with same size
batch = np.empty([bin_size, np.shape(df_resampled[0])[1]], dtype=float)
for dfs in df_resampled:
    if np.shape(dfs)[0]< bin_size:
        interval = np.pad(dfs.fillna(method='ffill').fillna(method='bfill'),((0,bin_size-np.shape(dfs)[0]),(0,0)),'edge')
    elif np.shape(dfs)[0]>= bin_size:
        interval = dfs.iloc[:bin_size].fillna(method='ffill').fillna(method='bfill')
    batch = np.dstack((batch,np.array(interval)))
batch = batch[:,:,1:].swapaxes(2,0).swapaxes(1,2)#(197, 11, 59)
print(("The shape of the batch is " + str(batch.shape)))
print(('Batch is containing nulls? ' + str(np.isnan(batch).any())))

# Data preprocessing - scaling the attributes
scalers = {}
for i in range(batch.shape[1]):
    scalers[i] = preprocessing.MinMaxScaler(feature_range=(0, 1))
    batch[:, i, :] = scalers[i].fit_transform(batch[:, i, :]) 
def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


# In[9]:


#targets = ['classRate','classDepth','classRelease']
targets = ['armsLocked','bodyWeight']

#fig = plt.figure()
for target in targets: 
    labels = dfAnn[target].fillna(method='bfill').values #fix bodyweight forgotten
    print("batch size: " + str(np.shape(batch)) + " labels: " + str(np.shape(labels)))
    train_set, test_set, train_labels, test_labels = train_test_split(batch, labels, test_size=0.33, random_state=88)
    INPUT_TUPLE = (batch.shape[1], batch.shape[2]) # time-steps, data-dim
    HIDDEN_DIM = 128
    verbose, epochs, batch_size = 0, 30, 25
    OUTPUT_DIM = dfAnn[target].nunique()
    print("Neural Network - "+ target + " " +str(train_set.shape)+ "->(" + str(HIDDEN_DIM) + ")->(" + str(OUTPUT_DIM) + ")")
    print("-------------------------------------")
    model = keras.Sequential([
        keras.layers.LSTM(HIDDEN_DIM, input_shape=(INPUT_TUPLE)), 
        keras.layers.Dense(OUTPUT_DIM, activation='softmax') 
    ])
    
    model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model_history = model.fit(train_set, train_labels, validation_data=(test_set, test_labels), epochs=epochs, verbose=0)
    

    y_prob = model.predict(test_set) 
    a = model_history.history['loss']
    b = model_history.history['val_loss']
    d = [a_i - b_i for a_i, b_i in zip(a, b)]
    linex = next(d.index(i) for i in reversed(d) if i >= 0)
    # plt.figure(figsize = (5,4))
    # plt.plot(model_history.history['loss'], color='darkblue')
    # plt.plot(model_history.history['val_loss'], color='cornflowerblue')
    # #plt.title('Loss '+target)
    # plt.ylabel('loss '+target)
    # plt.xlabel('epoch')
    # plt.axvline(x=linex, color='firebrick', linestyle='--', label='overfitting point')
    # plt.legend(['train_loss', 'valid_loss','overfitting point'],loc='best')
    # plt.savefig('loss_'+target+'.pdf', bbox_inches = "tight")
    # plt.show()
    #
    # plt.figure(figsize = (5,4))
    # plt.plot(model_history.history['acc'], color='green')
    # plt.plot(model_history.history['val_acc'], color='darkseagreen')
    # #plt.title('Accuracy '+target)
    # plt.ylabel('accuracy '+target)
    # plt.xlabel('epoch')
    # plt.axvline(x=linex, color='firebrick', linestyle='--', label='overfitting point')
    # plt.legend(['train_acc', 'valid_acc','overfitting point'],loc='best')
    # plt.savefig('acc_'+target+'.pdf', bbox_inches = "tight")
    # plt.show()
    #
    
    modelrefit = model.fit(train_set, train_labels, validation_data=(test_set, test_labels), epochs=linex, verbose=0)
    test_loss, test_acc = model.evaluate(test_set, test_labels)
    
    print('Test accuracy:', test_acc) 
    print('Test loss:', test_loss)
    # if target == 'classRelease':
    #     print('Roc Auc', roc_auc_score(test_labels, y_prob.argmax(axis=1)))
    #
    # matrix = metrics.confusion_matrix(test_labels, y_prob.argmax(axis=1))
    # df_cm = pd.DataFrame(matrix, list(range(OUTPUT_DIM)),list(range(OUTPUT_DIM)))
    # sns.set(font_scale=1.4)#for label size
    # plt.figure(figsize = (5,4))
    # sns.heatmap(df_cm, annot=True,annot_kws={"size": 16},fmt='g')
    # plt.title(target)
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.savefig('confusion-'+target+'.pdf', bbox_inches = "tight")
    #plt.plot(model_history.history['loss'])
    #t = model_history.history['loss']
    #q = model_history.history['acc']
    #plt.plot(t,label="Loss "+target)
    #plt.plot(q,label="Accuracy "+target)
#plt.legend(loc='best')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#fig.savefig('Loss.pdf')


# In[50]:


dfB = dfAnn.copy()
a = dfB[['classRate','compMeanRate']].groupby(['classRate']).count()['compMeanRate'].rename('classRate')
b = dfB[['classRelease','compReleaseDepth']].groupby(['classRelease']).count()['compReleaseDepth'].rename('classRelease')
c = dfB[['classDepth','compDepth']].groupby(['classDepth']).count()['compDepth'].rename('classDepth')

d = dfB[['armsLocked','bodyWeight']].groupby(['armsLocked']).count()['bodyWeight']
e = dfB[['bodyWeight','armsLocked']].groupby(['bodyWeight']).count()['armsLocked']

#pd.concat([a,b,c],1).T.plot(kind='bar',rot=0)
pd.concat([d,e],1).T.plot(kind='bar',rot=0)


# plt.ylabel('Class distribution')
# plt.savefig('manual-annotations-class-distribution.pdf')


# In[378]:


# dfA = dfAnn.copy()
# dfA.recordingID = dfA.recordingID.str.slice(0, 3, 1)
# dfA['recordingID'] = dfA['recordingID'].map({'P0-': 'P00', 'P2-': 'P01', 'P4-': 'P02', 'P5-': 'P03', 'P6-': 'P04', 'P7-': 'P05', 'P8-': 'P06', 'P9-': 'P07', 'P10': 'P08','P11': 'P09','P13': 'P10'})
# dfA = dfA.sort_index(axis=1)
# fig = plt.figure()
# dfA[['recordingID','classRate','compMeanRate']].groupby(['recordingID','classRate']).count().unstack().fillna(0)['compMeanRate'].plot(kind="bar")
# plt.xlabel('Participants')
# plt.ylabel('Class distribution')
# plt.savefig('participants-classRate-distribution.pdf')
# dfA[['recordingID','classDepth','compDepth']].groupby(['recordingID','classDepth']).count().unstack().fillna(0)['compDepth'].plot(kind="bar")
# plt.xlabel('Participants')
# plt.ylabel('Class distribution')
# plt.savefig('participants-classDepth-distribution.pdf')
# dfA[['recordingID','classRelease','compReleaseDepth']].groupby(['recordingID','classRelease']).count().unstack().fillna(0)['compReleaseDepth'].plot(kind="bar")
# plt.xlabel('Participants')
# plt.ylabel('Class distribution')
# plt.savefig('participants-classRelease-distribution.pdf')


# In[386]:


y_prob = model.predict(test_set) 
matrix = metrics.confusion_matrix(test_labels, y_prob.argmax(axis=1))
y_true = pd.Series(test_labels)
y_pred = pd.Series(y_prob.argmax(axis=1))

pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[91]:


# import matplotlib.dates as mdates
#
# index = 10
# g = dt.iloc[:,index]
# h = dt.iloc[:,index].resample(str(49)+'ms').first()
# dfh = h.to_frame()
# dfh["c"] = np.arange(0,np.size(h)).tolist()
# plt.figure(figsize=(6,5))
# ax1 = g.plot(label='actual time-series')
# ax2 = dfh[dt.iloc[:,index].name].plot(color='orange',label='resampled time-series')
# plt.xlabel('Time bins')
# plt.ylabel(dt.iloc[:,index].name)
# ax1.legend()
# ax2.legend()
# ax2.xaxis.set_major_locator(mdates.MicrosecondLocator(interval=45000))
# ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m:%S.%f'))
# plt.savefig('cpr-resampling.pdf')


# In[ ]:





# In[245]:


# print(dfAnn.iloc[0:5,:].dropna().to_latex().encode('ascii','ignore'))


# In[92]:


# # add fake intervals adding random time offset
# import random
# xm = dfAnn.duration.median()
# dfAnnFake = dfAnn.copy()
# format = lambda x: x + pd.DateOffset(seconds=float(random.randint(-int(xm*100), int(xm*100)))/100)
# dfAnnFake.start = dfAnn.start.map(format)
# dfAnnFake.end = dfAnnFake.start + pd.to_timedelta(dfAnnFake.duration, unit='s')


# In[21]:



# dfALL.to_csv('sensor_data_good.csv')


# In[40]:


# #pd.DataFrame(test_set).to_csv('test_set_good.csv')
# np.savetxt("test_set_good.csv", test_set, delimiter=",")


# In[27]:

# print(('train_set '+str(train_set.shape)))
# print(('test_set ' + str(test_set.shape)))
# print(('train_labels ' + str(train_labels.shape)))
# print(('test_labels ' + str(test_labels.shape)))



# # In[57]:
#
#
# m,n,r = test_set.shape
# out_arr = np.column_stack((np.repeat(np.arange(m),n),test_set.reshape(m*n,-1)))
# out_df = pd.DataFrame(out_arr)
# np.shape(out_df.values)
# np.shape(test_set)
# #.to_csv('test_set_good.csv')

#
# #np.save('test_set_good.npy', test_set) # save
# xtest_set = np.load('test_set_broken.npy') # load
# ytest_set = np.load('test_set_good.npy') # load
# xtest_labels = np.load('test_labels_broken.npy') # load
#
# xtest_loss, xtest_acc = model.evaluate(xtest_set, xtest_labels)
# #print('Test accuracy:', xtest_set)
# #print('Test loss:', xtest_loss)
# np.isnan(xtest_set).any()




