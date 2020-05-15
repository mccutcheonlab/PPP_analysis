# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:21:24 2020

@author: admin
"""
import dill
import matplotlib.pyplot as plt

# from figs4roc import *

import trompy as tp

try:
    pickle_in = open('C:\\Github\\PPP_analysis\\data\\ppp_roc_results.pickle', 'rb')
except FileNotFoundError:
        print('Cannot access pickled file')
        
roc_results = dill.load(pickle_in)

def get_data_from_dict(roc_dict, prefsession, subset):
    data_to_plot = roc_dict[prefsession][subset]
    
    a = data_to_plot['a']
    p = data_to_plot['p']
    data = data_to_plot['data']
    data_flat = data_to_plot['data_flat']
    
    return [a, p, data, data_flat]

### PR rats, s10 (pref1), licks
colors_dis = ['grey', 'red']

[a, p, data, data_flat] = get_data_from_dict(roc_results, 's16', 'pr_licks')

f = plt.figure(figsize=(3.4, 2))

f, ax = tp.plot_ROC_and_line(f, a, p, data_flat[0], data_flat[1],
                  cdict=[colors_dis[0], 'white', colors_dis[1]],
                  colors = colors_dis,
                  labels=['Casein', 'Maltodextrin'],
                  ylabel='Z-score',
                  xlabel='Time from first lick')



try:
    pickle_folder = 'C:\\Github\\PPP_analysis\\data\\'   
    pickle_in = open(pickle_folder + 'ppp_dfs_pref.pickle', 'rb')
    df_behav, df_photo, df_reptraces, df_heatmap, df_reptraces_sip, df_heatmap_sip, longtrace = dill.load(pickle_in)

except FileNotFoundError:
    print('Cannot access pickled file(s)')


from ppp_pub_figs_fx import peakbargraph

diet = 'NR'

if diet == 'NR':
    colors = 'control'
else:
    colors = 'exptl'
    
scattersize=50

keys_traces = ['pref1_cas_licks_forced', 'pref1_malt_licks_forced']
peaktype='auc'
epoch=[100,110]
colorgroup=colors
scattersize=scattersize

f, ax = plt.subplots()

peakbargraph(ax, df_photo, diet, keys_traces, peaktype=peaktype, epoch=epoch,
             colorgroup=colors, ylim=[-0.04,0.12], grouplabeloffset=0.07,
             scattersize=scattersize)
ax.set_ylim([-3, 7.5])