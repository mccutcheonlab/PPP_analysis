# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:16:41 2018

NEED TO RUN ppp1_grouped.py first to load data and certain functions into memory.
Trying to do this using import statement - but at the moment not importing modules.

@author: jaimeHP
"""

import os
os.chdir(os.path.dirname(__file__))
cwd = os.getcwd()

import sys
sys.path.insert(0,cwd)

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib as mpl

import JM_general_functions as jmf
import JM_custom_figs as jmfig
import ppp_pub_figs_fx as pppfig

savefigs=True
savefolder='R:\\DA_and_Reward\\gc214\\PPP_combined\\figs\\'
savefolder=usr + '\Dropbox\Publications in Progress\PPP Paper\Figs\\'

#Set general rcparams
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

summaryFig = pppfig.makesummaryFig2(df1, df4)
#summaryFig.savefig('R:/DA_and_Reward/es334/PPP1/figures/MMiN/summary.pdf')
    



fflicks_pref1_fig, ax = plt.subplots(figsize=(5, 2), ncols=2, sharey=True, sharex=False)
fflicks_pref1_fig.subplots_adjust(left=0.25, bottom=0.2)
pppfig.forcedandfreelicks(ax, df2, df3, prefsession=3)
fflicks_pref1_fig.savefig(savefolder + 'forcedandfree.eps')
#forcedandfreelicksfig.savefig(savepath + 'forcedandfree.eps')


# Fig for Preference Test 1
keys_choicebars = ['ncas1', 'nmalt1']
keys_traces = ['cas1_licks_forced', 'malt1_licks_forced']
keys_photobars = ['cas1_licks_peak', 'malt1_licks_peak']

pref1_photofig = pppfig.mainPhotoFig(df1, df4, keys_choicebars, keys_traces, keys_photobars)
#pref1_photofig.savefig(savepath + 'pref1_photofig.eps')

# Fig for Preference Test 2
keys_choicebars = ['ncas2', 'nmalt2']
keys_traces = ['cas2_licks_forced', 'malt2_licks_forced']
keys_photobars = ['cas2_licks_peak', 'malt2_licks_peak']


pref2_photofig = pppfig.mainPhotoFig(df1, df4, keys_choicebars, keys_traces, keys_photobars, dietswitch=True)
#pref2_photofig.savefig(savepath + 'pref2_photofig.eps')

# Fig for Preference Test 3
keys_choicebars = ['ncas3', 'nmalt3']
keys_traces = ['cas3_licks_forced', 'malt3_licks_forced']
keys_photobars = ['cas3_licks_peak', 'malt3_licks_peak']

pref3_photofig = pppfig.mainPhotoFig(df1, df4, keys_choicebars, keys_traces, keys_photobars, dietswitch=True)
#pref3_photofig.savefig(savepath + 'pref3_photofig.eps')


testfig, ax = plt.subplots(figsize=(2,2))
Ydata = df1['pref1']
Xdata = df4['pref1_peak_delta']

pppfig.behav_vs_photoFig(ax, Xdata, Ydata, df1['diet'])



testfig, ax = plt.subplots(figsize=(2,2))
Ydata = df1['pref2']
Xdata = df4['pref2_peak_delta']
pppfig.behav_vs_photoFig(ax, Xdata, Ydata, df1['diet'])

testfig, ax = plt.subplots(figsize=(2,2))
Ydata = df1['pref3']
Xdata = df4['pref3_peak_delta']
pppfig.behav_vs_photoFig(ax, Xdata, Ydata, df1['diet'])

if savefigs == True:
    forcedandfreelicksfig.savefig(savefolder + 'forcedandfree.eps')
    
    pref1_photofig.savefig(savefolder + 'pref1_photofig.eps')
    pref2_photofig.savefig(savefolder + 'pref2_photofig.eps')
    pref3_photofig.savefig(savefolder + 'pref3_photofig.eps')
