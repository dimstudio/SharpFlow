from utils import data_helper
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


import seaborn as sns
sns.set()
sns.set_style("darkgrid", {"axes.facecolor": ".97"})

session_folder = "manual_sessions/CPR_feedback_novices_F/classRate"
ignore_files = ['Kinect','Myo']
target_classes = ['classRelease' , 'classDepth', 'classRate', 'armsLocked', 'bodyWeight']
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
colors = ['#d73027','#fc8d59','#91bfdb','#fee090','#4575b4']
colors_dict = {'classRelease':u'#d73027','classDepth':u'#fc8d59', 'classRate':u'#91bfdb', 'armsLocked':u'#fee090', 'bodyWeight':u'#4575b4'}
df, annotations = data_helper.get_feedback_from_files(session_folder, ignore_files=ignore_files)
feedback_columns = [col for col in df.columns if 'Feedback' in col]
feedback = df[feedback_columns]
print("Shape of the feedback_data is: " + str(np.shape(feedback)))
print("Shape of the annotation is: " + str(np.shape(annotations)))
rates_byid = annotations[target_classes+['RecordingID','start']].set_index('start')
rates = annotations.set_index('start')[target_classes]

groups = rates_byid.groupby('RecordingID')

resultsError = pd.DataFrame(index=target_classes)
resultsDerivative = pd.DataFrame(index=target_classes)

#summary_bf = pd.DataFrame(columns=[target_classes])
#summary_af = pd.DataFrame(columns=[target_classes])
summary_bf = {}
summary_af = {}
for i in target_classes:
    summary_bf[i] = []
    summary_af[i] = []

for name,group in groups:

    group = (1 - group[target_classes]).rolling('10s').mean()
    errorRates = group[target_classes].mean()
    resultsError = pd.concat([resultsError, errorRates],axis=1)
    resultsDerivative = pd.concat([resultsDerivative,(group[target_classes].apply(lambda x: x[-1] - x[0]) / 2)],axis=1)
    sub_feedback = feedback[group.index[0]:group.index[-1]]

    bf = [  # mask the dataframe
        group[(df2_start <= group.index) & (group.index <= df2_end)]
        for df2_start, df2_end in zip(sub_feedback.index - pd.to_timedelta(10, unit='s'), sub_feedback.index)
    ]
    af = [  # mask the dataframe
        group[(df2_start <= group.index) & (group.index <= df2_end)]
        for df2_start, df2_end in zip(sub_feedback.index, sub_feedback.index + pd.to_timedelta(10, unit='s'))
    ]
    for i in range(len(bf)):
        tc = sub_feedback["Feedback.targetClass"][i]
        bf_s = bf[i][target_classes].mean()
        af_s = af[i][target_classes].mean()
        summary_bf[tc].append(bf_s.loc[tc])
        summary_af[tc].append(af_s.loc[tc])

    color_map = [colors_dict[t] for t in sub_feedback["Feedback.targetClass"].values]
    ax=group.plot(color=colors)
    handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=sub_feedback.index, ymin=ymin, ymax=ymax, color=color_map, linestyles='dashed', label=feedback["Feedback.targetClass"])
    plt.ylabel('Error rate %')
    plt.xlabel('Session time')
    frmt = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(frmt)
    handles.append(Line2D([0], [0], color='gray', linewidth=2, linestyle='--'))
    plt.legend(handles, labels+['Feedback'])
    plt.savefig(session_folder+'/class-distribution'+name+'.pdf')

for t in target_classes:
    print(t)
    result = [a-b for a, b in zip(summary_bf[t], summary_af[t])]
    #print(summary_bf[t])
    #print(summary_af[t])
    print(np.count_nonzero(summary_bf[t]))
    print(np.mean(summary_bf[t]).round(3))
    print(np.mean(summary_af[t]).round(3))
    print(np.mean(result).round(3))
#plt.savefig(session_folder+'/class-distribution.pdf')

by_row_index = resultsError.groupby(resultsError.index)
#print(by_row_index.mean().mean(axis=1).round(3))

by_row_index = resultsDerivative.groupby(resultsDerivative.index)
#print(by_row_index.mean().mean(axis=1).round(3))

plt.show()

