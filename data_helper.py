import numpy as np
import zipfile, os, json, re, datetime, time
import pandas as pd
from pandas.io.json import json_normalize
import pickle

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
def read_data_files(sessions, ignoreKinect=False):
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
                if "Kinect" in filename and ignoreKinect:
                    continue
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
    median_signal_length = int(np.median([len(l) for l in df_resampled]))
    print(f"Median signal length: {median_signal_length}")
    df_tensor = create_batches(df_resampled, median_signal_length)
    return df_tensor


# create a dummy ndarray with same size
# df is a list of dataframes
def create_batches(df, bin_size):
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


def get_data_from_files(folder, ignoreKinect=False):
    sessions = read_zips_from_folder(folder)
    # get the sensor data and annotation files (if exist)
    if os.path.exists(f"{folder}/annotations.pkl") and os.path.exists(f"{folder}/sensor_data.pkl"):
        with open(f"{folder}/annotations.pkl", "rb") as f:
            annotations = pickle.load(f)
        with open(f"{folder}/sensor_data.pkl", "rb") as f:
            sensor_data = pickle.load(f)
    else:
        sensor_data, annotations = read_data_files(sessions, ignoreKinect=ignoreKinect)
        with open(f"{folder}/annotations.pkl", "wb") as f:
            pickle.dump(annotations, f)
        with open(f"{folder}/sensor_data.pkl", "wb") as f:
            pickle.dump(sensor_data, f)
    return sensor_data, annotations