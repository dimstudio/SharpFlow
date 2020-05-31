import json
import os
import re
import select
import socket
import threading
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pylab import text

# tcp-server setting
bind_ip = '127.0.0.1'
port_list = [20001]  # could work with 2 ports
start_script = datetime.now()
servers = []
request_count = 0
complete_compression = 0

# general dataframes
df_kinect = pd.DataFrame()
df_myo = pd.DataFrame()
df_all = pd.DataFrame()

# processing settings
to_exclude = ['Ankle', 'Hip', 'Hand']  # variables to exclude Kinect specific
targets = ['classRate', 'classDepth', 'classRelease', 'armsLocked', 'bodyWeight']

rock = 0
fast = 0
slow = 0

performance = [1]


# load the json parsed data
def json_to_df(data):
    df = pd.concat([pd.DataFrame(data),
                    json_normalize(data['Frames'])],
                   axis=1).drop('Frames', 1)
    df.columns = df.columns.str.replace("_", "")
    if not df.empty:
        df['frameStamp'] = pd.to_timedelta(df['frameStamp'])  # + start_script
        df.columns = df.columns.str.replace("frameAttributes", df["ApplicationName"].all())
        df = df.set_index('frameStamp').iloc[:, 2:]
        df = df[~df.index.duplicated(keep='first')]
        df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        df = df.select_dtypes(include=['float64', 'int64'])
        df = df.loc[:, (df.sum(axis=0) != 0)]
        # KINECT fix
        df.rename(columns=lambda x: re.sub('KinectReader.\d', 'KinectReader.', x), inplace=True)
        df.rename(columns=lambda x: re.sub('Kinect.\d', 'Kinect.', x), inplace=True)
        # Exclude irrelevant attributes
        for el in to_exclude:
            df = df[[col for col in df.columns if el not in col]]
        df = df.apply(pd.to_numeric).fillna(method='bfill')
    else:
        print('Empty data frame. Did you wear Myo?')
    return df


def start_tcp_server(ip, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip, port))
    server.listen(5)  # max backlog of connections
    print(('Listening on {}:{}'.format(ip, port)))
    servers.append(server)


def handle_client_connection(client_socket, port):
    request = client_socket.recv(10000000)
    # print(request)
    json_string = json.loads(request, encoding='ascii')
    for jst in json_string:
        if jst is not None:
            if jst["ApplicationName"] == "Kinect":
                global df_kinect
                df_kinect = json_to_df(jst)

            elif jst["ApplicationName"] == "Myo":
                global df_myo
                df_myo = json_to_df(jst)
    process_data()
    client_socket.send("hello back".encode())
    client_socket.close()


def process_data():
    global df_myo
    global df_kinect
    global df_all
    global complete_compression
    global slow
    global fast
    global rock
    global performance
    if (not df_myo.empty) and (not df_kinect.empty) and ("Kinect.ShoulderLeftY" in df_kinect):

        complete_compression = complete_compression + 1

        if len(performance) > 10:
            performance = performance[1:]
        if df_kinect["Kinect.ShoulderLeftY"].index[-1] > timedelta(seconds=0.7):
            slow = slow + 1
            feedback = "Too slow"
            performance.append(0)
        elif df_kinect["Kinect.ShoulderLeftY"].index[-1] < timedelta(seconds=0.4):
            fast = fast + 1
            feedback = "Too fast "
            performance.append(0)
        else:
            rock = rock + 1
            feedback = "You rock "
            performance.append(1)

        # incorrect = str("incorrect {:0.0f}".format((1 - np.mean(performance)) * 100) + '%')
        # correct = str("correct {:0.0f}".format(np.mean(performance) * 100)+'%')

        cc_duration = df_kinect["Kinect.ShoulderLeftY"].index[-1].total_seconds()
        bpm = 60.0 / cc_duration
        plot_gauge(bpm, "gauge.png")
        df_all = pd.concat([df_kinect, df_myo], ignore_index=False, sort=False).sort_index()
        if not df_all.empty:
            tensor = tensor_transform(df_all, 150, 8)
            print("tensor transformation "+str(np.shape(tensor)))
            for t in targets:
                model_loaded = load_model(t, compile=False)
                test_predictions = model_loaded.predict(tensor,steps=1)
                print(('For target'+t+' predictions are'+test_predictions))
        df_kinect = pd.DataFrame()
        df_myo = pd.DataFrame()


def plot_gauge(bpm, filename):
    if 50 < bpm < 170:
        fig = go.Figure(go.Indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=bpm,
            mode="gauge+number+delta",
            title={'text': "CPR compressionRate"},
            gauge={'axis': {'range': [50, 170]},
                   'bar': {'color': "gray"},
                   'steps': [
                       {'range': [50, 100], 'color': "white"},
                       {'range': [100, 120], 'color': "green"},
                       {'range': [120, 170], 'color': "white"},
                   ],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': bpm}}))
        fig.write_image(filename)


def tensor_transform(df_all, res_rate, bin_size):
    if not df_all.empty:
        batch = df_all.resample(str(res_rate) + 'ms').first()[:bin_size].fillna(method='ffill').fillna(method='bfill')
        #batch = batch[:, :, 1:].swapaxes(2, 0).swapaxes(1, 2)  # (197, 11, 59)

        # Data preprocessing - scaling the attributes
        scalers = {}
        for i in range(batch.shape[1]):
            scalers[i] = preprocessing.MinMaxScaler(feature_range=(0, 1))
            batch[:, i, :] = scalers[i].fit_transform(batch[:, i, :])
        return batch  # tensor


def load_model(target_name):
    filename = 'models/checkpoints/model_' + target_name + '.h5'
    if os.path.isfile(filename):
        new_model = keras.models.load_model(filename)
    else:
        print((filename + ' not found'))
    return new_model


if __name__ == '__main__':

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
