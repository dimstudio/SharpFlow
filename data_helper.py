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
def read_data_files(sessions, ignore_files=None):
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
                # check whether the current file is in files to ignore
                if ignore_files is not None:
                    skip = sum([ign_f.lower() in filename.lower() for ign_f in ignore_files]) > 0
                    if skip:
                        continue
                if not os.path.isdir(filename):
                    if '.json' in filename:
                        with z.open(filename) as f:
                            data = json.load(f)
                            if 'intervals' in data or 'Intervals' in data:
                                df = annotation_file_to_array(data, current_time_offset)
                                df_ann = df_ann.append(df)
                            elif 'frames' in data or 'Frames' in data:
                                df = sensor_file_to_array(data, current_time_offset)
                                # Concatenate this dataframe in the dfALL and then sort dfALL by index
                                df_all = pd.concat([df_all, df], ignore_index=False, sort=False).sort_index()
                                df_all = df_all.apply(pd.to_numeric,errors='ignore').fillna(method='bfill')
    return df_all, df_ann


# transform a sensor file into a nd-pandas array
# use this only if using learning-hub format and containing frames
# IN: sensor-file in json format read into json.load(data)
# OUT: concatenated data frame df_all
def sensor_file_to_array(data, offset):
    # concatenate the data with the intervals normalized and drop attribute 'frames'
    framesKey= 'frames'
    if 'Frames' in data:
        framesKey = 'Frames'
    applicationNameKey = 'applicationName'
    if 'ApplicationName' in data:
        applicationNameKey = 'ApplicationName'
    # check in case of null values
    data[framesKey] = [x for x in data[framesKey] if x]
    df = pd.concat([pd.DataFrame(data),
                    json_normalize(data[framesKey])],
                   axis=1).drop(framesKey, 1)

    # remove underscore from column-file e.g. 3_Ankle_Left_X becomes 3AnkleLeftX
    df.columns = df.columns.str.replace("_", "")

    if not df.empty:
        # from string to timedelta + offset
        df['frameStamp'] = pd.to_timedelta(df['frameStamp']) + offset

        # retrieve the application name
        app_name = df[applicationNameKey].all()
        # remove the prefix 'frameAttributes.' from the column names
        df.columns = df.columns.str.replace("frameAttributes", df[applicationNameKey].all())

        # set the timestamp as index
        df = df.set_index('frameStamp').iloc[:, 2:]
        # exclude duplicates (taking the first occurence in case of duplicates)
        df = df[~df.index.duplicated(keep='first')]

        # convert to numeric (when reading from JSON it converts into object in the pandas DF)
        # with the parameter 'ignore' it will skip all the non-numerical fields
        df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))

        # Keep the numeric types only (categorical data are not supported now)
        if (app_name!="Feedback"):
            df = df.select_dtypes(include=['float64', 'int64'])
        # Remove columns in which the sum of attributes is 0 (meaning there the information is 0)
        df = df.loc[:, (df.sum(axis=0) != 0)]
        # KINECT FIX
        # The application KienctReader can track up to 6 people, whose attributes are
        # 1ShoulderLeftX or 3AnkleRightY. We get rid of this numbers assuming there is only 1 user
        # This part has to be rethought in case of 2 users
        df = df[df.nunique().sort_values(ascending=False).index]
        df.rename(columns=lambda x: re.sub('KinectReader.\d', 'KinectReader.', x), inplace=True)
        df.rename(columns=lambda x: re.sub('Kinect.\d', 'Kinect.', x), inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]

    return df


# transform an annotation file into a nd-pandas array
# use this only if using learning-hub format and containing frames
# IN: sensor-file in json format read into json.load(data)
# OUT: concatenated dataframe df_all
def annotation_file_to_array(data, offset):
    # concatenate the data with the intervals normalized and drop attribute 'intervals'
    intervalsKey = 'intervals'
    if 'Intervals' in data:
        intervalsKey = 'Intervals'
    df = pd.concat([pd.DataFrame(data),
                    json_normalize(data[intervalsKey])],
                   axis=1).drop(intervalsKey, 1)
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
        print("Df annotaiton or df all returned as none")
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
    #print(('Batch is containing nulls? ' + str(np.isnan(batch).any())))

    return batch  # tensor


def get_data_from_files(folder, ignore_files=None, res_rate=25, to_exclude=None):
    # get the sensor data and annotation files (if exist)
    if ignore_files is None:
        ann_name = f"{folder}/annotations.pkl"
        sensor_name = f"{folder}/sensor_data.pkl"
    else:
        ann_name = f"{folder}/annotations_ignorefiles{'_'.join(ignore_files)}.pkl"
        sensor_name = f"{folder}/sensor_data_ignorefiles{'_'.join(ignore_files)}.pkl"

    if os.path.exists(ann_name) and os.path.exists(sensor_name):
        with open(ann_name, "rb") as f:
            annotations = pickle.load(f)
        with open(sensor_name, "rb") as f:
            tensor_data = pickle.load(f)
    else:
        sessions = read_zips_from_folder(folder)
        if len(sessions) <= 0:
            raise FileNotFoundError(f"No recording sessions found in {folder}")
        sensor_data, annotations = read_data_files(sessions, ignore_files=ignore_files)

        # TODO this is a workaround and only works for the CPR dataset!!!
        #annotations = annotations.loc[
        #    ~((annotations.armsLocked == 1) & (annotations.bodyWeight == 1) & (annotations.classDepth == 0))]

        # Transform sensor_data to tensor_data and save it
        tensor_data = tensor_transform(sensor_data, annotations, res_rate=res_rate, to_exclude=to_exclude)
        with open(ann_name, "wb") as f:
            pickle.dump(annotations, f)
        with open(sensor_name, "wb") as f:
            pickle.dump(tensor_data, f)

    return tensor_data, annotations


def get_feedback_from_files(folder, ignore_files=None):
    # get the sensor data and annotation files (if exist)
    sessions = read_zips_from_folder(folder)
    if len(sessions) <= 0:
        raise FileNotFoundError(f"No recording sessions found in {folder}")
    feedback_data, annotations = read_data_files(sessions, ignore_files=ignore_files)
    return feedback_data, annotations


def create_train_test_folders(data, new_folder_location=None, train_test_ratio=0.85, ignore_files=None, to_exclude=None):
    if new_folder_location is None:
        new_folder_location = data
    # Train and test data is chosen randomly
    sessions = read_zips_from_folder(data)
    sensor_data, annotations = read_data_files(sessions, ignore_files=ignore_files)
    tensor_data = tensor_transform(sensor_data, annotations, res_rate=25, to_exclude=to_exclude)
    # mask with train_test_ratio*len(annotations) amount of ones
    train_mask = np.zeros(len(annotations), dtype=int)
    train_mask[:int(len(annotations)*train_test_ratio)] = 1
    np.random.shuffle(train_mask)
    train_mask = train_mask.astype(bool)

    train_annotations = annotations[train_mask]
    train_sensor_data = tensor_data[train_mask]
    test_annotations = annotations[~train_mask]
    test_sensor_data = tensor_data[~train_mask]

    if ignore_files is None:
        ann_name = "annotations.pkl"
        sensor_name = "sensor_data.pkl"
    else:
        ann_name = f"annotations_ignorefiles{'_'.join(ignore_files)}.pkl"
        sensor_name = f"sensor_data_ignorefiles{'_'.join(ignore_files)}.pkl"

    os.makedirs(f'{new_folder_location}/train', exist_ok=True)
    with open(f'{new_folder_location}/train/{ann_name}', "wb") as f:
        pickle.dump(train_annotations, f)
    with open(f'{new_folder_location}/train/{sensor_name}', "wb") as f:
        pickle.dump(train_sensor_data, f)

    os.makedirs(f'{new_folder_location}/test', exist_ok=True)
    with open(f'{new_folder_location}/test/{ann_name}', "wb") as f:
        pickle.dump(test_annotations, f)
    with open(f'{new_folder_location}/test/{sensor_name}', "wb") as f:
        pickle.dump(test_sensor_data, f)


