import data_helper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import seaborn as sns

session_folder = "manual_sessions/CPR_feedback_results"
ignore_files = []
target_classes = ['classRelease' , 'classDepth', 'classRate', 'armsLocked', 'bodyWeight']
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
colors_dict = {'classRelease':'b','classDepth':'g', 'classRate':'r', 'armsLocked':'c', 'bodyWeight':'m'}
df, annotations = data_helper.get_feedback_from_files(session_folder, ignore_files=ignore_files)
feedback_columns = [col for col in df.columns if 'Feedback' in col]
feedback = df[feedback_columns].drop_duplicates(keep='last').dropna()
#print(feedback["Feedback.targetClass"])
print("Shape of the feedback_data is: " + str(np.shape(feedback)))
print("Shape of the annotation is: " + str(np.shape(annotations)))

rates_byid = annotations[target_classes+['RecordingID','start']].set_index('start')
rates = annotations.set_index('start')[target_classes]


errorRates = (1 - rates).rolling('10s').mean()
derivative = (1 - rates).rolling('10s').apply(lambda x: x[-1] - x[0]) / 2
fig = plt.figure()
derivative.mean().plot(kind='bar')

groups = rates_byid.groupby('RecordingID')

for name,group in groups:

    group = (1 - group[target_classes]).rolling('10s').mean()

    ax=group.plot(color=colors)
#ax = errorRates.plot(color=colors)
    handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=feedback.index, ymin=ymin, ymax=ymax, color=colors, linestyles='dashed', label=feedback["Feedback.targetClass"])
    plt.ylabel('Error rate %')
    plt.xlabel('Compressions')
    plt.title('Error rate session: '+name)
    handles.append(Line2D([0], [0], color='gray', linewidth=1, linestyle='--'))
    plt.legend(handles, labels+['Feedback'])

plt.show()
#plt.savefig(session_folder+'/class-distribution.pdf')


