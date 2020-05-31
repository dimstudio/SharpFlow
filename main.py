import json
import select
import socket
import re
import threading
import torch
import joblib
import pandas as pd
from pandas.io.json import json_normalize
from torch import nn
import time
import numpy as np
import ast

bind_ip = '127.0.0.1'
port_list = [20001]  # could work with 2 ports
servers = []
targets = ['classRelease', 'classDepth', 'classRate', 'armsLocked', 'bodyWeight']
to_exclude = ['Ankle', 'Hip']  # variables to exclude Kinect specific
request_count = 0
path_to_model = "models/checkpoints/lstm"
trained_model = 'models/checkpoints/lstm.pt'
complete_compression = 0
bin_size = 17


# load the json parsed data
def json_to_df(data):
    df = pd.DataFrame()
    try:

        df = pd.concat([pd.DataFrame(data),
                        json_normalize(data['Frames'])],
                       axis=1).drop('Frames', 1)

        df.columns = df.columns.str.replace("_", "")
        if not df.empty:
            df['frameStamp'] = pd.to_timedelta(df['frameStamp'])  # + start_script
            df.columns = df.columns.str.replace("frameAttributes", df["ApplicationName"].all())
            df = df.set_index('frameStamp').iloc[:, 2:]
            df = df[~df.index.duplicated(keep='first')]
            df = df.astype('float',errors='ignore')
            df = df.select_dtypes(include=['float64', 'int64'])
            df = df.loc[:, (df.sum(axis=0) != 0)]
            df = df.loc[:, ~df.columns.duplicated()]
            df = df[df.nunique().sort_values(ascending=False).index]
        else:
            print('Empty data frame. Did you wear Myo?')
    except AttributeError:
        print('Failed to parse the JSON file')
        pass

    return df


def start_tcp_server(ip, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip, port))
    server.listen(5)  # max backlog of connections
    print(('Listening on {}:{}'.format(ip, port)))
    initModel(path_to_model)
    servers.append(server)

def initModel(path_to_model):
    global scaler
    global model
    scaler = joblib.load(f"{path_to_model}_scaler.pkl")
    loaded = torch.load(f'{path_to_model}.pt')
    model = loaded['model']
    model.load_state_dict(loaded['state_dict'])
    model.to('cpu')
    model.eval()

def handle_client_connection(client_socket, port):
    timer2 = time.time()
    request = client_socket.recv(10000000)
    #print(request)
    return_dict = {}

    try:
        json_string = json.loads(request, encoding='ascii')
        for jst in json_string:
            if jst is not None:
                if jst["ApplicationName"] == "Kinect":
                    global df_kinect
                    timerKinect = time.time()
                    df_kinect = json_to_df(jst)
                    #print("Timer Kinect " + str(time.time() - timerKinect))

                elif jst["ApplicationName"] == "Myo":
                    global df_myo
                    try:
                        timerMyo = time.time()
                        df_myo = json_to_df(jst)
                        #print("Timer Myo " + str(time.time() - timerMyo))

                        return_dict = process_data()
                    except ValueError:  # includes simplejson.decoder.JSONDecodeError
                        print('Decoding Myo has failed')
                        pass

    except ValueError:  # includes simplejson.decoder.JSONDecodeError
        print('Decoding JSON has failed')
        return_dict = {'classRelease': '', 'classDepth': '', 'classRate': '', 'armsLocked': '', 'bodyWeight': ''}
        pass
    print("Entire process " + str(time.time() - timer2))
    client_socket.send(str(return_dict).encode())

    #text_file = open("example_request.txt", "w")
    #text_file.write(str(json_string))
    #text_file.close()

    client_socket.close()


def exampleData():
    with open("example_request.txt", 'r') as f:
        json_string = ast.literal_eval(f.read())

    for jst in json_string:
        if jst is not None:
            if jst["ApplicationName"] == "Kinect":
                global df_kinect
                df_kinect = json_to_df(jst)

            elif jst["ApplicationName"] == "Myo":
                global df_myo
                df_myo = json_to_df(jst)
    return_dict = process_data()


def process_data():
    global df_myo
    global df_kinect
    global df_all
    global complete_compression
    result = {}

    if (not df_myo.empty) and (not df_kinect.empty) and ("Kinect.ShoulderLeftY" in df_kinect):
        complete_compression = complete_compression + 1
        #df_kinect["Kinect.ShoulderLeftY"].plot()
        #batch = np.empty([17, 52], dtype=float)
        timer0 = time.time()
        df_all = pd.concat([df_kinect, df_myo], ignore_index=False, sort=False).sort_index()
        if to_exclude is not None:
            for el in to_exclude:
                df_all = df_all[[col for col in df_all.columns if el not in col]]
        print("Before resampling: "+str(np.shape(df_all)))
        resampled = df_all.resample(str(25) + 'ms').first()
        print(("Before filling " + str(resampled.shape)))
        if np.shape(resampled)[0] < bin_size:
            interval = np.pad(df_all.fillna(method='ffill').fillna(method='bfill'),
                              ((0, bin_size - np.shape(resampled)[0]), (0, 0)), 'edge')
        elif np.shape(resampled)[0] >= bin_size:
            interval = resampled.iloc[:bin_size].fillna(method='ffill').fillna(method='bfill')

        #print(("Shape of the interval is " + str(interval.shape)))
        #print("Processing time " + str(time.time() - timer0))
        # print(interval)
        timer1 = time.time()
        result = online_classification(interval)
        #print("Classification time "+str(time.time()-timer1))

    return result


def online_classification(input_sample):

    scaled_data = scaler.transform(input_sample)
    scaled_data = np.expand_dims(scaled_data, 0)
    #print(("Shape of the batch is " + str(scaled_data.shape))) # always Shape of the batch is (1, 17, 52)
    data_tensor = torch.tensor(scaled_data)
    prediction = model(data_tensor.float())
    result = {}
    for i, target_class in enumerate(targets):
        if not np.isnan(prediction.tolist()[0][0]):
            result[target_class] = round(prediction.tolist()[0][i])
        else:
            print(f"Error! Prediction is {prediction.tolist()[0][i]} with input: {data_tensor.float()}")
            result[target_class] = ''
    print(result)
    return result


if __name__ == '__main__':

    #exampleData()
    for port in port_list:
        start_tcp_server(bind_ip, port)

    while True:
        readable, _, _ = select.select(servers, [], [])
        ready_server = readable[0]
        request_count = request_count + 1
        connection, address = ready_server.accept()
        print(('Accepted connection from {}:{}'.format(address[0], address[1])))
        client_handler = threading.Thread(
            target=handle_client_connection,
            args=(connection, port)
        )
        client_handler.start()
