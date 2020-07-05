from utils import data_helper
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


import seaborn as sns
sns.set()
sns.set_style("darkgrid", {"axes.facecolor": ".97"})

session_folder = "../manual_sessions/CPR_feedback_learners/ordered/feedback"
ignore_files = ['Kinect','Myo']
target_classes = ['classRelease' , 'classDepth', 'classRate', 'armsLocked', 'bodyWeight']
#colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
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
#resultsDerivative = pd.DataFrame(index=target_classes)
summaryFeedback = pd.DataFrame(index=target_classes)
summaryFeedback['error_mean'] = np.nan
summaryFeedback['derivative_mean'] = np.nan
summaryFeedback['feedback_count'] = 0
summaryFeedback['d1-10s'] = np.nan
summaryFeedback['d1+10s'] = np.nan
summaryFeedback['d1+-5s'] = np.nan


for name,group in groups:
    group = (1 - group[target_classes]).rolling('10s').mean()
    # calculate mean of the group
    errorRates = group[target_classes].mean()
    if ~np.isnan(summaryFeedback['error_mean'][0]):
        summaryFeedback['error_mean'] = (summaryFeedback['error_mean'] + errorRates)/2
    else:
        summaryFeedback['error_mean'] = errorRates

    resultsError = pd.concat([resultsError, errorRates],axis=1)
    for target in target_classes:
        series = group[target]
        slope = pd.Series(np.gradient(series.values), series.index, name='slope')
        if ~np.isnan(summaryFeedback.loc[target, 'derivative_mean']):
            summaryFeedback.loc[target, 'derivative_mean'] = (summaryFeedback.loc[
                                                                  target, 'derivative_mean'] + slope.sum()) / 2
        else:
            summaryFeedback.loc[target, 'derivative_mean'] = slope.sum()

    if not feedback.empty:
        sub_feedback = feedback[group.index[0]:group.index[-1]]
        for index, row in sub_feedback.iterrows():
            targetClass = row['Feedback.targetClass']
            series = group[targetClass][index-pd.Timedelta(seconds=10):index]
            slope = pd.Series(np.gradient(series.values), series.index, name='slope')
            if ~np.isnan(summaryFeedback.loc[targetClass, 'd1-10s']):
                summaryFeedback.loc[targetClass, 'd1-10s'] = (summaryFeedback.loc[targetClass, 'd1-10s'] + slope.sum())/2
            else:
                summaryFeedback.loc[targetClass, 'd1-10s'] = slope.sum()

            series = group[targetClass][index - pd.Timedelta(seconds=5):index+pd.Timedelta(seconds=5)]
            slope = pd.Series(np.gradient(series.values), series.index, name='slope')
            if ~np.isnan(summaryFeedback.loc[targetClass, 'd1+-5s']):
                summaryFeedback.loc[targetClass, 'd1+-5s'] = (summaryFeedback.loc[
                                                                                targetClass, 'd1+-5s'] + slope.sum()) / 2
            else:
                summaryFeedback.loc[targetClass, 'd1+-5s'] = slope.sum()


            series = group[targetClass][index:index+pd.Timedelta(seconds=10)]
            slope = pd.Series(np.gradient(series.values), series.index, name='slope')
            if ~np.isnan(summaryFeedback.loc[targetClass, 'd1+10s']):
                summaryFeedback.loc[targetClass, 'd1+10s'] = (summaryFeedback.loc[targetClass, 'd1+10s'] + slope.sum())/2
            else:
                summaryFeedback.loc[targetClass, 'd1+10s'] = slope.sum()

            summaryFeedback.loc[targetClass, 'feedback_count'] = summaryFeedback.loc[targetClass, 'feedback_count'] + 1


#print(summaryFeedback.to_latex())


summaryFeedback[['d1-10s','d1+10s']].T.plot.line(marker='o',color=colors)
plt.margins(x=1)
plt.savefig(session_folder+'/kmtl-derivative.pdf')
#summaryFeedback[['error_mean','derivative_mean']].plot(kind='bar')



#for index,row in summaryFeedback.iterrows():
#    print(row)

    #print(np.count_nonzero(summary_bf[t]))
    #print(np.mean(summary_bf[t]).round(3))
    #print(np.mean(summary_af[t]).round(3))
    #print(np.mean(result).round(3))

    # Old way to get before and end the feedback?
    # bf = [  # mask the dataframe
    #     group[(df2_start <= group.index) & (group.index <= df2_end)]
    #     for df2_start, df2_end in zip(sub_feedback.index - pd.to_timedelta(10, unit='s'), sub_feedback.index)
    # ]
    # af = [  # mask the dataframe
    #     group[(df2_start <= group.index) & (group.index <= df2_end)]
    #     for df2_start, df2_end in zip(sub_feedback.index, sub_feedback.index + pd.to_timedelta(10, unit='s'))
    # ]
    #
    # for i in range(len(bf)):
    #     tc = sub_feedback["Feedback.targetClass"][i]
    #     bf_s = bf[i][target_classes].mean()
    #     af_s = af[i][target_classes].mean()
    #     summary_bf[tc].append(bf_s.loc[tc])
    #     summary_af[tc].append(af_s.loc[tc])


    # Generate the plot of feedback for each episode
    # color_map = [colors_dict[t] for t in sub_feedback["Feedback.targetClass"].values]
    # ax=group.plot(color=colors)
    # handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
    # ymin, ymax = ax.get_ylim()
    # ax.vlines(x=sub_feedback.index, ymin=ymin, ymax=ymax, color=color_map, linestyles='dashed', label=feedback["Feedback.targetClass"])
    # plt.ylabel('Error rate %')
    # plt.xlabel('Session time')
    # frmt = mdates.DateFormatter('%H:%M:%S')
    # ax.xaxis.set_major_formatter(frmt)
    # handles.append(Line2D([0], [0], color='gray', linewidth=2, linestyle='--'))
    # plt.legend(handles, labels+['Feedback'])
    # plt.savefig(session_folder+'/class-distribution'+name+'.pdf')

#plt.savefig(session_folder+'/class-distribution.pdf')

#by_row_index = resultsError.groupby(resultsError.index)
#print(by_row_index.mean().mean(axis=1).round(3))

#by_row_index = resultsDerivative.groupby(resultsDerivative.index)
#print(by_row_index.mean().mean(axis=1).round(3))


