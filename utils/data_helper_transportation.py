import numpy as np
import zipfile, os, json, re, time
import csv
from datetime import datetime
import pandas as pd
from pandas.io.json import json_normalize
import pickle
from io import StringIO
from tqdm import tqdm


classes = {"Stationary": 0,
           "Walking": 1,
           "Running": 2,
           "Car": 3,
           "Train/Bus": 4,
           "Bike": 5
           }


def get_samples_from_csv(sensor_data, selfreport_data):
    # This is for fixing possible faulty first entries
    if (selfreport_data.iloc[0] == ["Time", "Transportation_Mode", "Status"]).any():
        selfreport_data = selfreport_data.drop(0)
    sensor_data["Time"] = pd.to_datetime(sensor_data["Time"], format="%Y-%m-%dT%H:%M:%S.%f")
    selfreport_data["Time"] = pd.to_datetime(selfreport_data["Time"], format="%Y-%m-%dT%H:%M:%S.%f")
    window = 512
    # Acceleration-magnitude:
    # np.sqrt(np.sum([acc_x**2, acc_y**2, acc_z**2]))
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


def create_train_test_folders(data, new_folder_location=None, train_test_ratio=0.85, to_exclude=None):
    # To exclude are sensors to exclude
    if new_folder_location is None:
        new_folder_location = data

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


def create_csv_from_MLT(mlt_file):
    # if mlt_file ends with zip: open and get _selfreport.json
    # else: assume its _selfreport.json
    # flag if we extracted the json from a zip (gets deleted afterwards)
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
    elif "json" in mlt_file[:-4]:
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
        record_start_time = datetime.strptime(selfreport_json["recordingID"], "%Y-%m-%dT%H-%M-%S-%f")
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
    # create_csv_from_MLT("../manual_sessions/blackforestMLT/ID_ddm_bighike-mountain_2020-06-02T10-40-46-094_MLT.zip")
    # to_exclude requires exact sensor names. e.g. ["Acc_x", "Acc_y", "Acc_z"]
    create_train_test_folders(data="../manual_sessions/blackforest", to_exclude=None)
    with open("../manual_sessions/blackforest/train/sensor_data.pkl", "rb") as f:
        data_train = pickle.load(f)
    print(data_train.shape)
    with open("../manual_sessions/blackforest/test/sensor_data.pkl", "rb") as f:
        data_test = pickle.load(f)
    print(data_test.shape)