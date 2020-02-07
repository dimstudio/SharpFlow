import json
import select
import socket
import re
import threading
import torch
import joblib
import pandas as pd
from pandas.io.json import json_normalize
import model_training_pytorch

bind_ip = '127.0.0.1'
port_list = [20001]  # could work with 2 ports
servers = []
targets = ['classRelease', 'classDepth', 'classRate', 'armsLocked', 'bodyWeight']
to_exclude = ['Ankle', 'Hip']  # variables to exclude Kinect specific
request_count = 0
trained_model = 'models/lstm.pt'
complete_compression = 0


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
    print(request)
    json_string = json.loads(request, encoding='ascii')
    for jst in json_string:
        if jst is not None:
            if jst["ApplicationName"] == "Kinect":
                global df_kinect
                df_kinect = json_to_df(jst)

            elif jst["ApplicationName"] == "Myo":
                global df_myo
                df_myo = json_to_df(jst)
    return_dict = process_data()

    # return_dict = {}
    # for t in targets:
    #     # uncomment this in case of 3 classes
    #     #if t == 'classDepth' or t == 'classRate':
    #     #    return_dict[t] = random.randrange(3)
    #     #else:
    #     return_dict[t] =  random.randrange(2)

    client_socket.send(str(return_dict).encode())
    client_socket.close()

def process_data():
    global df_myo
    global df_kinect
    global df_all
    global complete_compression
    result = {}
    if (not df_myo.empty) and (not df_kinect.empty) and ("Kinect.ShoulderLeftY" in df_kinect):
        complete_compression = complete_compression + 1
        df_all = pd.concat([df_kinect, df_myo], ignore_index=False, sort=False).sort_index()
        result = online_classification("models/lstm",df_all)
    return result

def online_classification(path_to_model, input_sample):
    scaler = joblib.load(f"{path_to_model}_scaler.pkl")

    model = model_training_pytorch.MyLSTM()
    loaded = torch.load(f'{path_to_model}.pt')
    model = model.load_state_dict(loaded['state_dict'])
    model.eval()

    scaled_data = scaler.transform(input_sample)
    prediction = model(scaled_data)

    result = dict()
    for i, target_class in enumerate(targets):
        result[target_class] = round(prediction[i])

    return result


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
