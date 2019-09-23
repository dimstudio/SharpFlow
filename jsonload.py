import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
import re

from datetime import datetime
start_script = datetime.now()

to_exclude = ['Ankle', 'Hip', 'Hand']  # variables to exclude Kinect specific

# load from json file
with open('data.json') as json_file:
    data = json.load(json_file)

# load the json parsed data
def jsonload(data):
    df = pd.concat([pd.DataFrame(data),
                    json_normalize(data['Frames'])],
                   axis=1).drop('Frames', 1)
    df.columns = df.columns.str.replace("_", "")
    df['frameStamp'] = pd.to_timedelta(df['frameStamp']) + start_script
    df.columns = df.columns.str.replace("frameAttributes", df["ApplicationName"].all())
    df = df.set_index('frameStamp').iloc[:, 2:]
    df = df[~df.index.duplicated(keep='first')]
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df = df.select_dtypes(include=['float64', 'int64'])
    df = df.loc[:, (df.sum(axis=0) != 0)]
    #KINECT fix
    df.rename(columns=lambda x: re.sub('KinectReader.\d', 'KinectReader.', x), inplace=True)
    df.rename(columns=lambda x: re.sub('Kinect.\d', 'Kinect.', x), inplace=True)
    # Exclude irrelevant attributes
    for el in to_exclude:
        df = df[[col for col in df.columns if el not in col]]
    df = df.apply(pd.to_numeric).fillna(method='bfill')
    return df

