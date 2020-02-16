import data_helper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import seaborn as sns

session_folder = "manual_sessions/CPR_feedback_results"
ignore_files = ['Kinect','Myo']
target_classes = ['classRelease' , 'classDepth', 'classRate', 'armsLocked', 'bodyWeight']
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
colors_dict = {'classRelease':u'tab:blue','classDepth':u'tab:orange', 'classRate':u'tab:green', 'armsLocked':u'tab:red', 'bodyWeight':u'tab:purple'}
df, annotations = data_helper.get_feedback_from_files(session_folder, ignore_files=ignore_files)
feedback_columns = [col for col in df.columns if 'Feedback' in col]
feedback = df[feedback_columns]
print("Shape of the feedback_data is: " + str(np.shape(feedback)))
print("Shape of the annotation is: " + str(np.shape(annotations)))
rates_byid = annotations[target_classes+['RecordingID','start']].set_index('start')
rates = annotations.set_index('start')[target_classes]
#errorRates = (1 - rates).rolling('10s').mean()
#derivative = (1 - rates).rolling('10s').apply(lambda x: x[-1] - x[0]) / 2

#derivative.mean().plot(kind='bar')
groups = rates_byid.groupby('RecordingID')

resultsError = pd.DataFrame(index=target_classes)
resultsDerivative = pd.DataFrame(index=target_classes)
for name,group in groups:

    group = (1 - group[target_classes]).rolling('10s').mean()

    errorRates = group[target_classes].mean()
    resultsError = pd.concat([resultsError, errorRates],axis=1)
    resultsDerivative = pd.concat([resultsDerivative,(group[target_classes].apply(lambda x: x[-1] - x[0]) / 2)],axis=1)

    sub_feedback = feedback[group.index[0]:group.index[-1]]
    color_map = [colors_dict[t] for t in sub_feedback["Feedback.targetClass"].values]
    ax=group.plot(color=colors)
    handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=sub_feedback.index, ymin=ymin, ymax=ymax, color=color_map, linestyles='dashed', label=feedback["Feedback.targetClass"])
    plt.ylabel('Error rate %')
    plt.xlabel('Compressions')
    plt.title('Error rate session: '+name)
    handles.append(Line2D([0], [0], color='gray', linewidth=1, linestyle='--'))
    plt.legend(handles, labels+['Feedback'])


#plt.savefig(session_folder+'/class-distribution.pdf')

by_row_index = resultsError.groupby(resultsError.index)
print(by_row_index.mean().mean(axis=1).round(3))

by_row_index = resultsDerivative.groupby(resultsDerivative.index)
print(by_row_index.mean().mean(axis=1).round(3))

plt.show()

