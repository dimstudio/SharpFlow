# system imports
import zipfile, os, json, re, datetime, time
import pickle
# ML imports
import numpy as np
# import seaborn as sns
import pandas as pd
# import matplotlib.pyplot as plt
from pandas.io.json import json_normalize

import tensorflow as tf
from tensorflow import keras

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
# from sklearn import metrics

import logging
logging.getLogger('tensorflow').disabled = True

#####################################################

# FUNCTIONS
# function that reads the input session in zip format
# # return: list of files
def read_zips_from_folder(folder_name):
    sessions_folder = [folder_name]
    folder_items = sorted(os.listdir(folder_name))
    zip_files = [sessions_folder[0] + '/' + s for s in folder_items if s.endswith('.zip')]
    return zip_files


# function that combines the data across multiple sessions
# return: list of files
def read_data_files(sessions):
    df_all = pd.DataFrame()  # Dataframe with all summarised data
    df_ann = pd.DataFrame()  # Dataframe containing the annotations
    # for each session in the list of sessions
    for s in sessions:
        # 1. Reading data from zip file
        print("Processing session: "+s)
        with zipfile.ZipFile(s) as z:
            # get current absolute time in seconds. This is necessary to add the delta correctly
            for info in z.infolist():
                file_datetime = datetime.datetime(*info.date_time)
            current_time_offset = pd.to_datetime(pd.to_datetime(file_datetime, format='%H:%M:%S.%f'), unit='s')
            # First look for annotation.json
            for filename in z.namelist():
                if not os.path.isdir(filename):
                    if '.json' in filename:
                        with z.open(filename) as f:
                            data = json.load(f)
                        if 'intervals' in data:
                            df = annotation_file_to_array(data, current_time_offset)
                            df_ann = df_ann.append(df)
                        elif 'frames' in data:
                            sensor_file_start_loading = time.time()
                            df = sensor_file_to_array(data, current_time_offset)
                            sensor_file_stop_loading = time.time()
                            #print(('Sensor file loading  ' + str(sensor_file_stop_loading - sensor_file_start_loading)))
                            # Concatenate this dataframe in the dfALL and then sort dfALL by index
                            df_all = pd.concat([df_all, df], ignore_index=False, sort=False).sort_index()
                            df_all = df_all.apply(pd.to_numeric).fillna(method='bfill')

    return df_all, df_ann


# transform a sensor file into a nd-pandas array
# use this only if using learning-hub format and containing frames
# IN: sensor-file in json format read into json.load(data)
# OUT: concatenated data frame df_all
def sensor_file_to_array(data, offset):
    # concatenate the data with the intervals normalized and drop attribute 'frames'
    df = pd.concat([pd.DataFrame(data),
                    json_normalize(data['frames'])],
                   axis=1).drop('frames', 1)

    # remove underscore from column-file e.g. 3_Ankle_Left_X becomes 3AnkleLeftX
    df.columns = df.columns.str.replace("_", "")

    # from string to timedelta + offset
    df['frameStamp'] = pd.to_timedelta(df['frameStamp']) + offset

    # retrieve the application name
    # app_name = df.applicationName.all()
    # remove the prefix 'frameAttributes.' from the column names
    df.columns = df.columns.str.replace("frameAttributes", df.applicationName.all())

    # set the timestamp as index
    df = df.set_index('frameStamp').iloc[:, 2:]
    # exclude duplicates (taking the first occurence in case of duplicates)
    df = df[~df.index.duplicated(keep='first')]

    # convert to numeric (when reading from JSON it converts into object in the pandas DF)
    # with the parameter 'ignore' it will skip all the non-numerical fields
    # df = df.apply(pd.to_numeric, errors='ignore')
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))

    # Keep the numeric types only (categorical data are not supported now)
    df = df.select_dtypes(include=['float64', 'int64'])
    # Remove columns in which the sum of attributes is 0 (meaning there the information is 0)
    df = df.loc[:, (df.sum(axis=0) != 0)]
    # KINECT FIX
    # The application KienctReader can track up to 6 people, whose attributes are
    # 1ShoulderLeftX or 3AnkleRightY. We get rid of this numbers assuming there is only 1 user
    # This part has to be rethought in case of 2 users
    df.rename(columns=lambda x: re.sub('KinectReader.\d', 'KinectReader.', x), inplace=True)
    df.rename(columns=lambda x: re.sub('Kinect.\d', 'Kinect.', x), inplace=True)

    return df


# transform an annotation file into a nd-pandas array
# use this only if using learning-hub format and containing frames
# IN: sensor-file in json format read into json.load(data)
# OUT: concatenated dataframe df_all
def annotation_file_to_array(data, offset):
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
    df.start = pd.to_timedelta(df.start) + offset
    # from string to timedelta + offset
    df.end = pd.to_timedelta(df.end) + offset
    # duration as subtractions of delta in seconds
    df['duration'] = (df.end - df.start) / np.timedelta64(1, 's')
    # append this dataframe to the dataframe annotations
    df = df.fillna(method='bfill')
    return df


# in case of training tensor_transformation
def tensor_transform(df_all, df_ann, res_rate, to_exclude=None):
    if df_ann.empty or df_all.empty:
        return None

    if to_exclude is not None:
        for el in to_exclude:
            df_all = df_all[[col for col in df_all.columns if el not in col]]
    # What is happening here?
    # Include the data from the annotation times
    masked_df = [  # mask the dataframe
        df_all[(df2_start <= df_all.index) & (df_all.index <= df2_end)]
        for df2_start, df2_end in zip(df_ann['start'], df_ann['end'])
    ]
    # What is interval_max for?
    interval_max = 0
    for dt in masked_df:
        delta = np.timedelta64(dt.index[-1] - dt.index[0], 'ms') / np.timedelta64(1, 'ms')
        if delta > interval_max:
            interval_max = delta
    # This results in different length of entries
    df_resampled = [dt.resample(str(res_rate) + 'ms').first() if not dt.empty else None for dt in masked_df]

    return df_resampled


# create a dummy ndarray with same size
def createBatches(df, bin_size):
    batch = np.empty([bin_size, np.shape(df[0])[1]], dtype=float)
    for dfs in df:
        if np.shape(dfs)[0] < bin_size:
            interval = np.pad(dfs.fillna(method='ffill').fillna(method='bfill'),
                              ((0, bin_size - np.shape(dfs)[0]), (0, 0)), 'edge')
        elif np.shape(dfs)[0] >= bin_size:
            interval = dfs.iloc[:bin_size].fillna(method='ffill').fillna(method='bfill')
        # if not np.isnan(np.array(interval)).any():
        batch = np.dstack((batch, np.array(interval)))
    batch = batch[:, :, 1:].swapaxes(2, 0).swapaxes(1, 2)  # (197, 11, 59)
    print(("The shape of the batch is " + str(batch.shape)))
    print(('Batch is containing nulls? ' + str(np.isnan(batch).any())))

    return batch  # tensor


def model_training(input_tensor, targets, epochs=30, modelname=""):
    # Hyperparameters
    test_size = 0.33
    random_state = 88
    print("batch size: " + str(np.shape(input_tensor)) + " labels: " + str(np.shape(targets)))
    train_set, test_set, train_labels, test_labels = train_test_split(input_tensor, targets, test_size=test_size, random_state=random_state)
    input_tuple = (input_tensor.shape[1], input_tensor.shape[2])  # time-steps, data-dim
    hidden_dim = 128
    verbose = 2
    # TODO make outputdim number of targets (and no softmax later on, because no classification)
    output_dim = targets.shape[1]
    # model definition
    print(('Keras model sequential target: ' + modelname))
    print(f"Output dimension: {output_dim}")
    # TODO Try stacking LSTMs
    # NOTE when stacking LSTMs you have to include return_sequences=True
    model = keras.Sequential([
        keras.layers.LSTM(units=hidden_dim, input_shape=input_tuple, return_sequences=True),
        keras.layers.LSTM(units=hidden_dim//2),
        keras.layers.Dense(16, activation='tanh'),
        keras.layers.Dense(output_dim, activation='tanh')
    ])

    # model compiling
    # TODO change loss to root_mean_squared_error, metrics to MSE
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='mse',
                  metrics=['mse'])
    # model fitting
    logdir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    model_history = model.fit(train_set, train_labels, validation_data=(test_set, test_labels), epochs=epochs,
                              verbose=verbose, callbacks=[tensorboard_callback])

    # Start Tensorboard in command line: tensorboard --logdir logs/
    # model storing
    store_model(model, 'models/model_' + modelname + '.h5')

    # y_prob = model.predict(test_set)
    test_loss, test_acc = model.evaluate(test_set, test_labels)
    print(('Test accuracy:', test_acc))
    print(('Test loss:', test_loss))
    # if target == 'classRelease':
    #     print(('Roc Auc', roc_auc_score(test_labels, y_prob.argmax(axis=1))))

    # Don't know what this is for...
    # y_true = pd.Series(test_labels)
    # y_pred = pd.Series(y_prob.argmax(axis=1))
    # pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# function
# IN: keras.model and path/to/model.h5
def store_model(the_model, path_model):
    print(('Saved model path: ' + path_model))
    the_model.save(path_model)


def load_model(target_name):
    filename = 'models/model_' + target_name + '.h5'
    if os.path.isfile(filename):
        new_model = keras.models.load_model(filename)
    else:
        print((filename + ' not found'))
    return new_model


if __name__ == "__main__":
    #####################################################
    ### VARIABLES
    # current_time_offset = 1
    to_exclude = ['Ankle', 'Hip', 'Hand']  # variables to exclude
    # # read data from session folder
    folder = 'manual_sessions/CPR'
    sessions = read_zips_from_folder(folder)
    # get the sensor data and annotation files (if exist)
    if os.path.exists(f"{folder}/annotations.pkl") and os.path.exists(f"{folder}/sensor_data.pkl"):
        with open(f"{folder}/annotations.pkl", "rb") as f:
            annotations = pickle.load(f)
        with open(f"{folder}/sensor_data.pkl", "rb") as f:
            sensor_data = pickle.load(f)
    else:
        sensor_data, annotations = read_data_files(sessions)
        with open(f"{folder}/annotations.pkl", "wb") as f:
            pickle.dump(annotations, f)
        with open(f"{folder}/sensor_data.pkl", "wb") as f:
            pickle.dump(sensor_data, f)

    # Check/Remove outliers?
    # Scale the features
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(sensor_data)
    sensor_data[sensor_data.columns] = scaler.transform(sensor_data[sensor_data.columns])
    # include only the relevant classes we are interested in
    targetClasses = ['classRate', 'classDepth', 'classRelease', 'armsLocked', 'bodyWeight']
    targets = annotations[targetClasses].values
    tensor = tensor_transform(sensor_data, annotations, res_rate=75, to_exclude=to_exclude)
    median_signal_length = np.median([len(l) for l in tensor])
    # Create batches for training by using the median signal length
    tensorBatch = createBatches(tensor, median_signal_length)

    # Data preprocessing - scaling the attributes
    # need the scaler for later rescaling
    # Scale Targets
    targetScaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    targetScaler.fit(targets)
    targets_scaled = targetScaler.transform(targets)

    model_training(tensorBatch, targets_scaled, epochs=120, modelname="_".join(targetClasses))

    # need the scaler to rescale for later predictions
    # for t in targetClasses:
    #     model_loaded = load_model(t)