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
    pickle_folder = 'R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\'
    
#    pickle_in = open(pickle_folder + 'ppp_dfs_sacc.pickle', 'rb')
#    df_sacc_behav = dill.load(pickle_in)
#    
#    pickle_in = open(pickle_folder + 'ppp_dfs_cond1.pickle', 'rb')
#    df_cond1_behav, df_cond1_photo = dill.load(pickle_in)
    
    pickle_in = open(pickle_folder + 'ppp_dfs_pref.pickle', 'rb')
    df_behav, df_photo, df_reptraces, df_heatmap, df_reptraces_sip, df_heatmap_sip = dill.load(pickle_in)
except FileNotFoundError:
    print('Cannot access pickled file(s)')

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
mpl.rcParams['figure.subplot.bottom'] = 0.05

mpl.rcParams['errorbar.capsize'] = 5

mpl.rcParams['savefig.transparent'] = True

mpl.rcParams['axes.spines.top']=False
mpl.rcParams['axes.spines.right']=False


mpl.rc('axes', linewidth=1, edgecolor=almost_black, labelsize=8, labelpad=4)
mpl.rc('patch', linewidth=1, edgecolor=almost_black)
mpl.rc('font', family='Arial', size=8)
for tick,subtick in zip(['xtick', 'ytick'], ['xtick.major', 'ytick.major']):
    mpl.rc(tick, color=almost_black, labelsize=8)
    mpl.rc(subtick, width=1)
mpl.rc('legend', fontsize=8)
mpl.rcParams['figure.subplot.left'] = 0.05
mpl.rcParams['figure.subplot.top'] = 0.95

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


make_behav_figs=False
make_photo_sip_figs=False
make_photo_licks_figs=False
make_pref2_fig=False
make_pref3_fig=False
make_summary_fig=True

#sacc_behav_fig = pppfig.sacc_behav_fig(df_sacc_behav)


#fig, ax = plt.subplots(figsize=(4,3), ncols=2, sharey=True)
#
#pppfig.cond_licks_fig(ax[0], df_cond1_behav, 'NR')
#pppfig.cond_licks_fig(ax[1], df_cond1_behav, 'PR')
#ax[0].set_ylabel('Licks')

if make_behav_figs:
    fflicks_pref1_fig, ax = plt.subplots(figsize=(7.2, 2.5), ncols=4, sharey=False, sharex=False)
    fflicks_pref1_fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, wspace=0.65)
    pppfig.pref_behav_fig(ax, df_behav, df_photo, prefsession=1)
    fflicks_pref1_fig.savefig(savefolder + 'pref1_behav.eps')
    
#    fflicks_pref2_fig, ax = plt.subplots(figsize=(7.2, 2.5), ncols=4, sharey=False, sharex=False)
#    fflicks_pref2_fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, wspace=0.65)
#    pppfig.pref_behav_fig(ax, df_behav, df_photo, prefsession=2, dietswitch=True)
#    fflicks_pref2_fig.savefig(savefolder + 'pref2_behav.eps')
#    
#    fflicks_pref3_fig, ax = plt.subplots(figsize=(7.2, 2.5), ncols=4, sharey=False, sharex=False)
#    fflicks_pref3_fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, wspace=0.65)
#    pppfig.pref_behav_fig(ax, df_behav, df_photo, prefsession=3, dietswitch=True)
#    fflicks_pref3_fig.savefig(savefolder + 'pref3_behav.eps')

clims = [[-0.15,0.20], [-0.11,0.15]]

if make_photo_sip_figs:
    photo_pref1_sipper_fig = pppfig.mainphotoFig(df_reptraces_sip, df_heatmap_sip, df_photo, clims=clims,
                                                 keys_traces = ['pref1_cas_sip', 'pref1_malt_sip'],
                                                 keys_bars = ['pref1_cas_sip_peak', 'pref1_malt_sip_peak'],
                                                 keys_lats = ['pref1_cas_lats_all_fromsip', 'pref1_malt_lats_all_fromsip'],
                                                 event='Sipper')
    
    photo_pref1_sipper_fig.savefig(savefolder + 'pref1_sip_photo.pdf')

if make_photo_licks_figs:
    
    photo_pref1_fig = pppfig.mainphotoFig(df_reptraces, df_heatmap, df_photo, clims=clims,
                                          keys_traces = ['pref1_cas_licks_forced', 'pref1_malt_licks_forced'],
                                          keys_bars = ['pref1_cas_licks_peak', 'pref1_malt_licks_peak'],
                                          keys_lats = ['pref1_cas_lats_all', 'pref1_malt_lats_all'])
    
    photo_pref1_fig.savefig(savefolder + 'pref1_licks_photo.pdf')
    
if make_pref2_fig:
    pref2_fig = pppfig.reduced_photofig(df_photo, df_behav)
    pref2_fig.savefig(savefolder + 'pref2_licks_fig.pdf')
    

if make_pref3_fig:
    pref3_fig = pppfig.reduced_photofig(df_photo, df_behav, session=3,
                     keys_traces = ['pref3_cas_licks_forced', 'pref3_malt_licks_forced'],
                     keys_bars = ['pref3_cas_licks_peak', 'pref3_malt_licks_peak'])
    pref3_fig.savefig(savefolder + 'pref3_licks_fig.pdf')
##
##photo_pref2_fig = pppfig.mainphotoFig(df_reptraces, df_heatmap, df_photo,
##                                     session='pref2',
##                                     dietswitch=True,
##                                     clims=clims)
##
##photo_pref2_fig.savefig(savefolder + 'pref2_photo.pdf')
##
##photo_pref3_fig = pppfig.mainphotoFig(df_reptraces, df_heatmap, df_photo,
##                                     session='pref3',
##                                     dietswitch=True,
##                                     clims=clims)
##
##photo_pref3_fig.savefig(savefolder + 'pref3_photo.pdf')


if make_summary_fig:
    summaryFig = pppfig.makesummaryFig(df_behav, df_photo)
    summaryFig.savefig(savefolder + 'summaryfig.pdf')

# Fig for Preference Test 1
#keys_choicebars = ['ncas1', 'nmalt1']
#keys_traces = ['cas1_licks_forced', 'malt1_licks_forced']
#keys_photobars = ['cas1_licks_peak', 'malt1_licks_peak']

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