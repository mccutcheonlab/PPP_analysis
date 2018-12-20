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
import ppp_pub_figs_fx2 as pppfig2
from ppp_pub_figs_fx import almost_black

import dill

# Looks for existing data and if not there loads pickled file
try:
    pickle_folder = 'R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\'
    
#    pickle_in = open(pickle_folder + 'ppp_dfs_sacc.pickle', 'rb')
#    df_sacc_behav = dill.load(pickle_in)
#    
    pickle_in = open(pickle_folder + 'ppp_dfs_cond1.pickle', 'rb')
    df_cond1_behav, df_cond1_photo = dill.load(pickle_in)
    
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

mpl.rcParams['font.size'] = 8.0
mpl.rcParams['axes.labelsize'] = 'medium'
mpl.rcParams['ytick.labelsize'] = 'small'
mpl.rcParams['figure.subplot.bottom'] = 0.05

mpl.rcParams['errorbar.capsize'] = 5

mpl.rcParams['savefig.transparent'] = True

mpl.rcParams['axes.spines.top']=False
mpl.rcParams['axes.spines.right']=False

mpl.rc('lines', linewidth=0.5)
mpl.rc('axes', linewidth=1, edgecolor=almost_black, labelsize=6, labelpad=4)
mpl.rc('patch', linewidth=1, edgecolor=almost_black)
mpl.rc('font', family='Arial', size=6)
for tick,subtick in zip(['xtick', 'ytick'], ['xtick.major', 'ytick.major']):
    mpl.rc(tick, color=almost_black, labelsize=6)
    mpl.rc(subtick, width=1)
mpl.rc('legend', fontsize=8)
mpl.rcParams['figure.subplot.left'] = 0.05
mpl.rcParams['figure.subplot.top'] = 0.95


make_sacc_figs=False
make_cond_figs=False

make_fig1_behav=False
make_fig1_photo=False

make_fig2_behav=False
make_fig2_photo=False

make_fig3_summary=True

make_photo_sip_figs=False
make_photo_licks_figs=False

peaktype='auc'
epoch=[100,119]

#sacc_behav_fig = pppfig.sacc_behav_fig(df_sacc_behav)

if make_cond_figs:
    cond1_behav_fig, ax = plt.subplots(figsize=(4,3), ncols=2, sharey=True)
    
    pppfig.cond_licks_fig(ax[0], df_cond1_behav, 'NR')
    pppfig.cond_licks_fig(ax[1], df_cond1_behav, 'PR')
    ax[0].set_ylabel('Licks')
    
    cond1_behav_fig.savefig(savefolder + 'cond1_behav.pdf')

    keys=[['cond1_cas1_sip', 'cond1_cas2_sip'],
         ['cond1_malt1_sip', 'cond1_malt2_sip'],
         ['cond1_cas1_licks', 'cond1_cas2_licks'],
         ['cond1_malt1_licks', 'cond1_malt2_licks']]
    
    keysbars = [[['cond1_cas1_sip_peak', 'cond1_cas2_sip_peak'], ['cond1_malt1_sip_peak', 'cond1_malt2_sip_peak']],
                [['cond1_cas1_licks_peak', 'cond1_cas2_licks_peak'], ['cond1_malt1_licks_peak', 'cond1_malt2_licks_peak']]]
    
    cond1_photo_sip_fig, ax = plt.subplots(figsize=(6,4), ncols=3, nrows=2)

    pppfig.cond_photo_fig(ax[0][0], df_cond1_photo, 'NR', keys[0], event='Sipper')
    pppfig.cond_photo_fig(ax[0][1], df_cond1_photo, 'NR', keys[1], event='Sipper')
    pppfig.cond_photobar_fig(ax[0][2], df_cond1_photo, 'NR', keysbars[0])
    
    pppfig.cond_photo_fig(ax[1][0], df_cond1_photo, 'PR', keys[0], event='Sipper')
    pppfig.cond_photo_fig(ax[1][1], df_cond1_photo, 'PR', keys[1], event='Sipper')
    pppfig.cond_photobar_fig(ax[1][2], df_cond1_photo, 'PR', keysbars[0])
    
    
    cond1_photo_lick_fig, ax = plt.subplots(figsize=(6,4), ncols=3, nrows=2)

    pppfig.cond_photo_fig(ax[0][0], df_cond1_photo, 'NR', keys[2], event='Licks')
    pppfig.cond_photo_fig(ax[0][1], df_cond1_photo, 'NR', keys[3], event='Licks')
    pppfig.cond_photobar_fig(ax[0][2], df_cond1_photo, 'NR', keysbars[1])
    
    pppfig.cond_photo_fig(ax[1][0], df_cond1_photo, 'PR', keys[2], event='Licks')
    pppfig.cond_photo_fig(ax[1][1], df_cond1_photo, 'PR', keys[3], event='Licks')
    pppfig.cond_photobar_fig(ax[1][2], df_cond1_photo, 'PR', keysbars[1])
    
    cond1_photo_sip_fig.savefig(savefolder + 'cond1_photo_sip.pdf')
    cond1_photo_lick_fig.savefig(savefolder + 'cond1_photo_lick.pdf')


if make_fig1_behav:
    fflicks_pref1_fig, ax = plt.subplots(figsize=(7.2, 1.75), ncols=4, sharey=False, sharex=False)
    fflicks_pref1_fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, wspace=0.65)
    pppfig.pref_behav_fig(ax, df_behav, df_photo, prefsession=1,
                          barlabeloffset=[0.025, 0.035, 0.045, 0.07])
    fflicks_pref1_fig.savefig(savefolder + 'fig1_behav.pdf')

clims = [[-0.15,0.20], [-0.11,0.15]]

if make_fig1_photo:
    fig1_photo_NR = pppfig2.fig1_photo(df_heatmap, df_photo, 'NR', 'pref1', clims=clims[0],
                                          peaktype=peaktype, epoch=epoch,
                                          keys_traces = ['pref1_cas_licks_forced', 'pref1_malt_licks_forced'],
                                          keys_lats = ['pref1_cas_lats_all', 'pref1_malt_lats_all'])
    
    fig1_photo_PR = pppfig2.fig1_photo(df_heatmap, df_photo, 'PR', 'pref1', clims=clims[1],
                                          peaktype=peaktype, epoch=epoch,
                                          keys_traces = ['pref1_cas_licks_forced', 'pref1_malt_licks_forced'],
                                          keys_lats = ['pref1_cas_lats_all', 'pref1_malt_lats_all'])
    
    fig1_photo_NR.savefig(savefolder + 'fig1_photo_NR.pdf')
    fig1_photo_PR.savefig(savefolder + 'fig1_photo_PR.pdf')
    

if make_fig2_behav:
    pref2_behav_fig, ax = plt.subplots(figsize=(3.2, 3.2), ncols=2, nrows=2)
    pref2_behav_fig.subplots_adjust(left=0.20, right=0.95, bottom=0.15, wspace=0.65)
    pppfig.pref_behav_fig(ax, df_behav, df_photo, prefsession=2, dietswitch=True,
                          barlabeloffset=[0.03, 0.06, 0.045, 0.07])
    pref2_behav_fig.savefig(savefolder + 'fig2_pref2_behav.pdf')


    pref3_behav_fig, ax = plt.subplots(figsize=(3.2, 3.2), ncols=2, nrows=2)
    pref3_behav_fig.subplots_adjust(left=0.20, right=0.95, bottom=0.15, wspace=0.65)
    pppfig.pref_behav_fig(ax, df_behav, df_photo, prefsession=3, dietswitch=True,
                          barlabeloffset=[0.02, 0.04, 0.045, 0.07])
    pref3_behav_fig.savefig(savefolder + 'fig2_pref3_behav.pdf')
    
    
if make_fig2_photo:
    fig2_pref2_photo = pppfig2.fig2_photo(df_photo, peaktype=peaktype, epoch=epoch)
    fig2_pref2_photo.savefig(savefolder + 'fig2_pref2_photo.pdf')
    
    fig2_pref3_photo = pppfig2.fig2_photo(df_photo, peaktype=peaktype, epoch=epoch,
                                          keys_traces = ['pref3_cas_licks_forced', 'pref3_malt_licks_forced'])
    fig2_pref3_photo.savefig(savefolder + 'fig2_pref3_photo.pdf')



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


if make_fig3_summary:
    summaryFig = pppfig.makesummaryFig(df_behav, df_photo, peaktype=peaktype, epoch=epoch)
    summaryFig.savefig(savefolder + 'summaryfig.pdf')

#if savefigs == True:
#    forcedandfreelicksfig.savefig(savefolder + 'forcedandfree.eps')
#    
#    pref1_photofig.savefig(savefolder + 'pref1_photofig.eps')
#    pref2_photofig.savefig(savefolder + 'pref2_photofig.eps')
#    pref3_photofig.savefig(savefolder + 'pref3_photofig.eps')
