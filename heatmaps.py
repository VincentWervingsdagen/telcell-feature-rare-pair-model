from telcell.data.parsers import parse_measurements_csv
from telcell.utils.transform import slice_track_pairs_to_intervals, create_track_pairs
from telcell.utils.transform import get_switches, is_colocated
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import pandas as pd

"""
Code for creating heatmap plots
"""

def features (pairs):
    colocated = []
    not_colocated = []
    for row in pairs:
            switches = get_switches(row[0],row[1])
            if (len(switches) != 0):    
                if is_colocated(row[0],row[1]):
                    colocated += list(map(lambda x: (x.distance/1000, x.time_difference.total_seconds()/3600), switches))
                else:
                    not_colocated += list(map(lambda x: (x.distance/1000, x.time_difference.total_seconds()/3600), switches))
    return colocated, not_colocated

path_train = os.path.join('experiments/data/Returners/Train/sampling2/output_cell_small.csv')  
    
data = list(slice_track_pairs_to_intervals(create_track_pairs(parse_measurements_csv(path_train), all_different_source=False),
                                           interval_length_h=24))
co,not_co= features(data)
co = pd.DataFrame(co, columns=['distance','time'])
not_co = pd.DataFrame(not_co, columns=['distance','time'])

num_bins = 30
time_binwidth_co = (co['time'].max() - co['time'].min()) / num_bins
distance_binwidth_co = (co['distance'].max() - co['distance'].min()) / num_bins

time_binwidth_not_co = (not_co['time'].max() - not_co['time'].min()) / num_bins
distance_binwidth_not_co = (not_co['distance'].max() - not_co['distance'].min()) / num_bins

axis_label_fontsize = 18
axis_ticks_fontsize = 14
number_of_ticks = 3  



g1 = sns.JointGrid(data=co, x="time", y="distance", marginal_ticks=False)
g1.plot_joint(
    sns.histplot, discrete=(False, False),
    cmap="light:#E60012", pmax=.8, 
    binwidth=(time_binwidth_co, distance_binwidth_co)
)
g1.plot_marginals(sns.histplot, element="step", color="#E60012",bins=30)
g1.set_axis_labels('time (h)', 'distance (km)', fontsize=axis_label_fontsize)
g1.ax_joint.xaxis.set_major_locator(ticker.MaxNLocator(number_of_ticks))
g1.ax_joint.yaxis.set_major_locator(ticker.MaxNLocator(number_of_ticks))
g1.ax_joint.tick_params(axis='both', labelsize=axis_ticks_fontsize)


g2 = sns.JointGrid(data=not_co, x="time", y="distance", marginal_ticks=False)
g2.plot_joint(
    sns.histplot, discrete=(False, False),
    cmap="light:#03012d", pmax=.8,
    binwidth=(time_binwidth_not_co, distance_binwidth_not_co)
)
g2.plot_marginals(sns.histplot, element="step", color="#03012d",bins=30)
g2.set_axis_labels('time (h)', 'distance (km)', fontsize=axis_label_fontsize)
g2.ax_joint.xaxis.set_major_locator(ticker.MaxNLocator(number_of_ticks))
g2.ax_joint.yaxis.set_major_locator(ticker.MaxNLocator(number_of_ticks))
g2.ax_joint.tick_params(axis='both', labelsize=axis_ticks_fontsize)


plt.show()

