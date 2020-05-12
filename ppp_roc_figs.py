# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:21:24 2020

@author: admin
"""
import dill
import matplotlib.pyplot as plt

from figs4roc import *

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
    
    return [a, p, data]

### PR rats, s10 (pref1), licks
colors_dis = ['grey', 'red']

[a, p, data] = get_data_from_dict(roc_results, 's10', 'pr_licks')

f = plt.figure(figsize=(3.4,2))
f, ax = plot_ROC_and_line(f, a, p, data[0], data[1],
                  cdict=[colors_dis[0], 'white', colors_dis[1]],
                  colors = colors_dis,
                  labels=['Casein', 'Maltodextrin'],
                  labeloffset=0,
                  ylabel='Z-score')