# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:16:41 2018

Loads in dataframes from pickled files created by ppp_averages

@author: jaimeHP
"""

import os
os.chdir(os.path.dirname(__file__))
cwd = os.getcwd()

import sys
sys.path.insert(0,cwd)


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

import JM_general_functions as jmf
import JM_custom_figs as jmfig
import ppp_pub_figs_fx as pppfig
from ppp_pub_figs_fx import almost_black

import dill

# Looks for existing data and if not there loads pickled file
try:
    type(df_photo)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    try:
        pickle_in = open('R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_dfs_pref.pickle', 'rb')
    except FileNotFoundError:
        print('Cannot access pickled file')
    df_behav, df_photo, df_reptraces = dill.load(pickle_in)

usr = jmf.getuserhome()

savefigs=True
savefolder = usr + '\Dropbox\Publications in Progress\PPP Paper\Figs\\'

#Set general rcparams

#mpl.style.use('classic')

mpl.rcParams['figure.figsize'] = (4.8, 3.2)
mpl.rcParams['figure.dpi'] = 100

mpl.rcParams['font.size'] = 12.0
mpl.rcParams['axes.labelsize'] = 'medium'
mpl.rcParams['ytick.labelsize'] = 'small'

mpl.rcParams['figure.subplot.left'] = 0.15
mpl.rcParams['figure.subplot.bottom'] = 0.20

mpl.rcParams['errorbar.capsize'] = 5

mpl.rcParams['savefig.transparent'] = True

mpl.rcParams['axes.spines.top']=False
mpl.rcParams['axes.spines.right']=False


mpl.rc('axes', linewidth=1, edgecolor=almost_black, labelsize=10, labelpad=4)
mpl.rc('patch', linewidth=1, edgecolor=almost_black)
mpl.rc('font', family='Arial', size=10)
for tick,subtick in zip(['xtick', 'ytick'], ['xtick.major', 'ytick.major']):
    mpl.rc(tick, color=almost_black, labelsize=10)
    mpl.rc(subtick, width=1)
mpl.rc('legend', fontsize=9)
mpl.rcParams['figure.subplot.left'] = 0.05
mpl.rcParams['figure.subplot.top'] = 0.95

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#fflicks_pref1_fig, ax = plt.subplots(figsize=(7.2, 2.5), ncols=3, sharey=False, sharex=False)
#fflicks_pref1_fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, wspace=0.65)
#pppfig.forcedandfreelicksandchoice(ax, df_behav, prefsession=1)
#fflicks_pref1_fig.savefig(savefolder + 'pref1_behav.eps')




clim_nr = [-0.15,0.20]
clim_pr = [-0.11,0.17]


dietswitch=False

photo_pref1_fig = pppfig.mainphotoFig(df_reptraces, df_photo)

#summaryFig = pppfig.makesummaryFig2(df_behav, df_photo)
#summaryFig.savefig('R:/DA_and_Reward/es334/PPP1/figures/MMiN/summary.pdf')

# Fig for Preference Test 1
keys_choicebars = ['ncas1', 'nmalt1']
keys_traces = ['cas1_licks_forced', 'malt1_licks_forced']
keys_photobars = ['cas1_licks_peak', 'malt1_licks_peak']

#pref1_photofig = pppfig.mainPhotoFig(df_behav, df_photo, keys_choicebars, keys_traces, keys_photobars)
##pref1_photofig.savefig(savepath + 'pref1_photofig.eps')
#
## Fig for Preference Test 2
#keys_choicebars = ['ncas2', 'nmalt2']
#keys_traces = ['cas2_licks_forced', 'malt2_licks_forced']
#keys_photobars = ['cas2_licks_peak', 'malt2_licks_peak']
#
#
#pref2_photofig = pppfig.mainPhotoFig(df1, df4, keys_choicebars, keys_traces, keys_photobars, dietswitch=True)
##pref2_photofig.savefig(savepath + 'pref2_photofig.eps')
#
## Fig for Preference Test 3
#keys_choicebars = ['ncas3', 'nmalt3']
#keys_traces = ['cas3_licks_forced', 'malt3_licks_forced']
#keys_photobars = ['cas3_licks_peak', 'malt3_licks_peak']
#
#pref3_photofig = pppfig.mainPhotoFig(df1, df4, keys_choicebars, keys_traces, keys_photobars, dietswitch=True)
##pref3_photofig.savefig(savepath + 'pref3_photofig.eps')
#
#
#testfig, ax = plt.subplots(figsize=(2,2))
#Ydata = df1['pref1']
#Xdata = df4['pref1_peak_delta']
#
#pppfig.behav_vs_photoFig(ax, Xdata, Ydata, df1['diet'])
#
#
#
#testfig, ax = plt.subplots(figsize=(2,2))
#Ydata = df1['pref2']
#Xdata = df4['pref2_peak_delta']
#pppfig.behav_vs_photoFig(ax, Xdata, Ydata, df1['diet'])
#
#testfig, ax = plt.subplots(figsize=(2,2))
#Ydata = df1['pref3']
#Xdata = df4['pref3_peak_delta']
#pppfig.behav_vs_photoFig(ax, Xdata, Ydata, df1['diet'])
#
#if savefigs == True:
#    forcedandfreelicksfig.savefig(savefolder + 'forcedandfree.eps')
#    
#    pref1_photofig.savefig(savefolder + 'pref1_photofig.eps')
#    pref2_photofig.savefig(savefolder + 'pref2_photofig.eps')
#    pref3_photofig.savefig(savefolder + 'pref3_photofig.eps')
#
##
##mpl.rcParams['figure.subplot.left'] = 0.30
##fig = plt.figure(figsize=(3.2, 4.0))
##ax = plt.subplot(1,1,1)
##onedaypreffig(df1, 'pref1', ax)
##    
##    ## Figure showing casein preference across all three test sessions
##mpl.rcParams['figure.subplot.left'] = 0.15
##fig = plt.figure(figsize=(4.4, 4.0))
##ax = plt.subplot(1,1,1)                
## 
##choicefig(df1, ['pref1', 'pref2', 'pref3'], ax)
##ax.set_ylabel('Casein preference')
##plt.yticks([0, 0.5, 1.0])