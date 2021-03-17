# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:36:18 2018

@author: James Rig
"""
import os
os.chdir(os.path.dirname(__file__))
cwd = os.getcwd()

import sys
sys.path.insert(0,cwd)

# import JM_general_functions as jmf
import matplotlib as mpl
import dill
import pandas as pd

#Colors
green = mpl.colors.to_rgb('xkcd:kelly green')
light_green = mpl.colors.to_rgb('xkcd:light green')


almost_black = mpl.colors.to_rgb('#262626')

## Colour scheme
col={}
col['nr_cas'] = 'xkcd:silver'
col['nr_malt'] = 'white'

col['pr_cas'] = 'xkcd:kelly green'
col['pr_malt'] = 'xkcd:light green'

col['pr_cas'] = 'xkcd:blue'
col['pr_malt'] = 'xkcd:sky blue'

heatmap_color_scheme = 'coolwarm'

## Size of scatter dots
scattersize=30

# Looks for existing data and if not there loads pickled file
try:
    pickle_folder = 'C:\\Github\\PPP_analysis\\data\\'
    
#    pickle_in = open(pickle_folder + 'ppp_dfs_sacc.pickle', 'rb')
#    df_sacc_behav = dill.load(pickle_in)
#    
#    pickle_in = open(pickle_folder + 'ppp_dfs_cond1.pickle', 'rb')
#    df_cond1_behav, df_cond1_photo = dill.load(pickle_in)
    
    pickle_in = open(pickle_folder + 'ppp_dfs_pref.pickle', 'rb')
    df_behav, df_photo, df_reptraces, df_heatmap, df_delta, df_pies = pd.read_pickle(pickle_in, compression=None)
    
    pickle_in = open(pickle_folder + 'ppp_dfs_cond.pickle', 'rb')
    df_cond = pd.read_pickle(pickle_in, compression=None)
    
    # pickle_in = open('C:\\Github\\PPP_analysis\\data\\ppp_latencies.pickle', 'wb')
    # x1data, y1data, x2data, y2data, x3data, y3data, x4data, y4data = pd.read_pickle(pickle_in, compression=None)

except FileNotFoundError:
    print('Cannot access pickled file(s)')

# usr = jmf.getuserhome()

savefigs=True
savefolder = 'C:\\Github\\PPP_analysis\\figs\\'
# savefolder = "C:\\Users\\jmc010\\Dropbox\\Publications in Progress\\PPP Paper\\04_JNS\\Figs\\"

#Set general rcparams

#mpl.style.use('classic')

mpl.rcParams['figure.figsize'] = (4.8, 3.2)
mpl.rcParams['figure.dpi'] = 100

mpl.rcParams['font.size'] = 8.0
mpl.rcParams['axes.labelsize'] = 'medium'
mpl.rcParams['ytick.labelsize'] = 'small'
mpl.rcParams['figure.subplot.bottom'] = 0.05

mpl.rcParams['errorbar.capsize'] = 5

mpl.rcParams['savefig.transparent'] = True

mpl.rcParams['axes.spines.top']=False
mpl.rcParams['axes.spines.right']=False

mpl.rc('lines', linewidth=0.5)
mpl.rc('axes', linewidth=0.5, edgecolor=almost_black, labelsize=6, labelpad=4)
mpl.rc('patch', linewidth=0.5, edgecolor=almost_black)
mpl.rc('font', family='Arial', size=6)
for tick,subtick in zip(['xtick', 'ytick'], ['xtick.major', 'ytick.major']):
    mpl.rc(tick, color=almost_black, labelsize=6)
    mpl.rc(subtick, width=0.5)
mpl.rc('legend', fontsize=8)
mpl.rcParams['figure.subplot.left'] = 0.05
mpl.rcParams['figure.subplot.top'] = 0.95

mpl.rc('lines', markeredgewidth=0.5, markerfacecolor='white', markeredgecolor=almost_black)
lw_barscatter=0.5


def inch(mm):
    result = mm*0.0393701
    return result