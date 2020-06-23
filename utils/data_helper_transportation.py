import numpy as np
import zipfile, os, json, re, time
import csv
from datetime import datetime
import pandas as pd
from pandas.io.json import json_normalize
import pickle
from io import StringIO
from tqdm import tqdm
from numba import jit
from numba.typed import Dict
from numba.core import types
import re


classes = {"Stationary": 0,
           "Walking": 1,
           "Running": 2,
           "Car": 3,
           "Train/Bus": 4,
           "Bike": 5
           }


@jit(nopython=True)
def get_samples_from_csv_acc_magnitude_numba(sensor_times, acc_data, selfreport_times, selfreport_mode, selfreport_status):
    window = 512

    grav = np.zeros(acc_data.shape)
    grav[0] = acc_data[0]
    magnitude = np.zeros(len(acc_data))
    for i, val in enumerate(acc_data[1:]):
        # remove gravity: Grx_k = 0.8 * Grx_k-1 + (1-0.8)*Accx_k
        #  Lx_k = Accx_k - Grx_k
        grav[i+1] = 0.8 * grav[i] + 0.2*val
        lin_acc = val - grav[i+1]
        # TODO apply smoothing
        # Acceleration-magnitude:
        # np.sqrt(np.sum([acc_x**2, acc_y**2, acc_z**2]))
        magnitude[i+1] = np.sqrt(np.sum(lin_acc**2))

    current_mode = 0
    selfreport_idx = 0
    data = np.zeros((len(range(0, len(sensor_times)-window, 64)), window))
    annotations = np.zeros(len(range(0, len(sensor_times)-window, 64)))
    # go through list
    for ind, i in enumerate(range(0, len(sensor_times)-window, 64)):
        ''' 
        1. get time-window
        2. check in selfreport_csv the class that overlaps most with time-window
        3. write data (that is not in to_exclude) to data
            and class to annotations
        '''
        selected_rows = sensor_times[i:i + window]
        sensor_start = selected_rows[0]
        sensor_end = selected_rows[-1]
        report_start = selfreport_times[min(selfreport_idx, len(selfreport_times)-1)]
        report_end = selfreport_times[min(selfreport_idx+1, len(selfreport_times)-1)]

        # No overlap
        if sensor_end < report_start:
            annotate = current_mode
        elif sensor_start > report_end:
            annotate = current_mode
            selfreport_idx += 1
        # There is overlap
        else:
            if selfreport_status[selfreport_idx]:
                current_mode = selfreport_mode[selfreport_idx]
            else:
                current_mode = 0
            annotate = current_mode
        annotations[ind] = annotate
        # get data and ignore time (Time is unimportant)
        data_values = magnitude[i:i+window]
        # data_values = selected_rows.loc[:, ~selected_rows.columns.isin(["Time"])]
        data[ind] = data_values

    return data, annotations


def get_samples_from_csv(sensor_data, selfreport_data):
    sensor_data["Time"] = pd.to_datetime(sensor_data["Time"], format="%Y-%m-%dT%H:%M:%S.%f")
    selfreport_data["Time"] = pd.to_datetime(selfreport_data["Time"], format="%Y-%m-%dT%H:%M:%S.%f")
    window = 512
    current_mode = "Stationary"
    selfreport_idx = 0
    data = []
    annotations = []
    # go through list
    for i in tqdm(range(0, len(sensor_data)-window, 64)):
        ''' 
        1. get time-window
        2. check in selfreport_csv the class that overlaps most with time-window
        3. write data (that is not in to_exclude) to data
            and class to annotations
        '''
        selected_rows = sensor_data[i:i + window]
        sensor_start = selected_rows.iloc[0]["Time"]
        sensor_end = selected_rows.iloc[-1]["Time"]
        report_start = selfreport_data.iloc[min(selfreport_idx, len(selfreport_data)-1)]["Time"]
        report_end = selfreport_data.iloc[min(selfreport_idx+1, len(selfreport_data)-1)]["Time"]

        # No overlap
        if sensor_end < report_start:
            annotate = classes[current_mode]
        elif sensor_start > report_end:
            annotate = classes[current_mode]
            selfreport_idx += 1
        # There is overlap
        else:
            if selfreport_data.iloc[selfreport_idx]["Status"] == "true" or selfreport_data.iloc[selfreport_idx]["Status"] == True:
                current_mode = selfreport_data.iloc[selfreport_idx]["Transportation_Mode"]
            else:
                current_mode = "Stationary"
            annotate = classes[current_mode]
        annotations.append(annotate)
        # get data and ignore time (Time is unimportant)
        data_values = selected_rows.loc[:, ~selected_rows.columns.isin(["Time"])]
        data.append(np.array(data_values))

    return np.stack(data), np.stack(annotations)


def create_train_test_folders(data, sub_folder=None, train_test_ratio=0.85, to_exclude=None, acc_magnitude=False):
    # To exclude are sensors to exclude
    if sub_folder is None:
        sub_folder = data
    else:
        sub_folder = os.path.join(data, sub_folder)

    # Read in csv-files (ignore those without ID)
    tensor_data, annotations = None, None
    for session_zipfile in os.listdir(data):
        if "ID_" not in session_zipfile:
            continue
        print("Processing zip-file: " + session_zipfile)
        archive = zipfile.ZipFile(os.path.join(data, session_zipfile), "r")
        # Get sensor file and corresponding selfreport file
        # print(archive)
        filelist = archive.infolist()
        sensor_csv, selfreport_csv = None, None
        for f in filelist:
            if "_sensors" in f.filename:
                sensor_csv = archive.read(f)
            elif "_selfreport" in f.filename:
                selfreport_csv = archive.read(f)
            # NOTE it could be that there are multiple sensors and selfreports
            if sensor_csv is not None and selfreport_csv is not None:
                # get data from files
                print("Session:", f.filename)
                sensor_data = pd.read_csv(StringIO(sensor_csv.decode('utf-8')))
                selfreport_data = pd.read_csv(StringIO(selfreport_csv.decode('utf-8')),
                                              names=["Time", "Transportation_Mode", "Status"],
                                              skiprows=1)
                if to_exclude is not None:
                    sensor_data = sensor_data.loc[:, ~sensor_data.columns.isin(to_exclude)]
                # This is for fixing possible faulty first entries
                if (selfreport_data.iloc[0] == ["Time", "Transportation_Mode", "Status"]).any():
                    selfreport_data = selfreport_data.drop(0)
                if acc_magnitude:
                    # convert pandas DF to numpy and apply numba
                    sensor_times = pd.to_datetime(sensor_data["Time"], format="%Y-%m-%dT%H:%M:%S.%f").values.astype(np.int64) // 10 ** 9
                    selfreport_times = pd.to_datetime(selfreport_data["Time"], format="%Y-%m-%dT%H:%M:%S.%f").values.astype(np.int64) // 10 ** 9
                    acc_data = sensor_data[["Acc_x", "Acc_y", "Acc_z"]].values
                    selfreport_mode = selfreport_data["Transportation_Mode"].values
                    selfreport_mode = np.array([classes[mode] for mode in selfreport_mode])
                    # "Status" is sometimes boolean and sometimes a string with "true" or "false"
                    selfreport_status = selfreport_data["Status"].values
                    tensor_data_file, annotations_file = get_samples_from_csv_acc_magnitude_numba(sensor_times, acc_data, selfreport_times, selfreport_mode, selfreport_status)
                    tensor_data_file = np.stack(tensor_data_file)
                    annotations_file = np.stack(annotations_file)
                else:
                    tensor_data_file, annotations_file = get_samples_from_csv(sensor_data, selfreport_data)
                # Instantiate dataset or add to dataset
                if tensor_data is None:
                    tensor_data = tensor_data_file
                    annotations = annotations_file
                else:
                    tensor_data = np.vstack([tensor_data, tensor_data_file])
                    annotations = np.append(annotations, annotations_file)
                # Reset then for next session (if there are more sessions in zipfile)
                sensor_csv, selfreport_csv = None, None

    # mask with train_test_ratio*len(annotations) amount of ones
    train_mask = np.zeros(len(annotations), dtype=int)
    train_mask[:int(len(annotations)*train_test_ratio)] = 1
    np.random.shuffle(train_mask)
    train_mask = train_mask.astype(bool)

    train_annotations = annotations[train_mask]
    train_sensor_data = tensor_data[train_mask]
    test_annotations = annotations[~train_mask]
    test_sensor_data = tensor_data[~train_mask]

    ann_name = "annotations.pkl"
    sensor_name = "sensor_data.pkl"

    # Save the annotation and sensor data
    os.makedirs(f'{sub_folder}/train', exist_ok=True)
    with open(f'{sub_folder}/train/{ann_name}', "wb") as f:
        pickle.dump(train_annotations, f)
    with open(f'{sub_folder}/train/{sensor_name}', "wb") as f:
        pickle.dump(train_sensor_data, f)

    os.makedirs(f'{sub_folder}/test', exist_ok=True)
    with open(f'{sub_folder}/test/{ann_name}', "wb") as f:
        pickle.dump(test_annotations, f)
    with open(f'{sub_folder}/test/{sensor_name}', "wb") as f:
        pickle.dump(test_sensor_data, f)


def create_csv_from_MLT(mlt_file):
    # if mlt_file ends with zip: open and get _selfreport.json
    # else: assume its _selfreport.json
    # flag if we extracted the json from a zip (gets deleted afterwards)
    print("Filename:", mlt_file)
    toDelete = False
    if "zip" in mlt_file[-4:]:
        archive = zipfile.ZipFile(mlt_file, "r")
        # Find the selfreport file
        for f in archive.infolist():
            if "selfreport" in f.filename:
                extraction_destination = os.path.join(*mlt_file.split("/")[:-1])
                archive.extract(f, path=extraction_destination)
                selfreport_file = os.path.join(extraction_destination, f.filename)
                toDelete = True
    elif "json" in mlt_file[-4:]:
        selfreport_file = mlt_file
    else:
        raise FileNotFoundError("Unkown file name")

    # create a csv with the headers
    with open(selfreport_file[:-4]+"csv", "w") as csv_file:
        csv_file.write("Time,Transportation_Mode,Status\n")
        # Time,Transportation_Mode,Status
        # 2020-06-01T10:39:30.807,Car,true
        # 2020-06-01T15:49:50.260,Car,false

        # read in json file
        with open(selfreport_file, "r") as f:
            selfreport_json = json.load(f)
        # extract the time via regex
        # time we are looking for: 2020-06-02T10-40-46-094
        p = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{3}")
        m = p.findall(selfreport_json["recordingID"])
        if len(m) > 0:
            print('Match found: ', m[0])
        else:
            print('No match')
        record_start_time = datetime.strptime(m[0], "%Y-%m-%dT%H-%M-%S-%f")
        for interval in selfreport_json["intervals"]:
            start_time = datetime.strptime(interval["start"], "%H:%M:%S.%f") - datetime.strptime("00:00", "%H:%M")
            end_time = datetime.strptime(interval["end"], "%H:%M:%S.%f") - datetime.strptime("00:00", "%H:%M")
            transport_mode = interval["annotations"]["Transportation_Mode"]
            # add to CSV:
            # record_start_time+start_time,transport_mode,true
            # record_start_time+end_time,transport_mode,false
            csv_file.write((record_start_time+start_time).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]+","+transport_mode+",true\n")
            csv_file.write((record_start_time+end_time).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]+","+transport_mode+",false\n")
    if toDelete:
        os.remove(selfreport_file)


if __name__ == "__main__":
    # folder = "../manual_sessions/blackforest-correct-annotations"
    # for fileToCorrect in os.listdir(folder):
    #     create_csv_from_MLT(os.path.join(folder, fileToCorrect))
    # create_csv_from_MLT("../manual_sessions/blackforestMLT/ID_ddm_bighike-mountain_2020-06-02T10-40-46-094_MLT.zip")

    folder = "../manual_sessions/blackforest_cleaner_annotations"
    # to_exclude requires exact sensor names. e.g. ["Acc_x", "Acc_y", "Acc_z"]
    # to_exclude = ["Gyro_x", "Gyro_y", "Gyro_z"]
    create_train_test_folders(data=folder, sub_folder="Acc_magnitude", to_exclude=None, acc_magnitude=True)
    with open(os.path.join(folder, "Acc_magnitude/train/sensor_data.pkl"), "rb") as f:
        data_train = pickle.load(f)
    print(data_train.shape)
    with open(os.path.join(folder, "Acc_magnitude/test/sensor_data.pkl"), "rb") as f:
        data_test = pickle.load(f)
    print(data_test.shape)