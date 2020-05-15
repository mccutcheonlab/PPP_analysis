# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:16:41 2018

Loads in dataframes from pickled files created by ppp_averages

@author: jaimeHP
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

# import JM_general_functions as jmf
# import JM_custom_figs as jmfig

from ppp_pub_figs_settings import *
from ppp_pub_figs_fx import *
from ppp_pub_figs_supp import *

make_fig1_behav=True
make_fig1_photo=False

make_fig2_behav=False
make_fig2_photo=False

make_fig3_summary=False

make_bwfood_figs=False

make_sacc_figs=False
make_cond_figs=False

make_photo_sip_figs=False
make_photo_licks_figs=False

supp_rep_trace = False
supp_heatmap = False

peaktype='auc'
epoch=[100,119]

if make_fig1_behav:

    fig1_behav, ax = plt.subplots(figsize=(7.2, 1.75), ncols=5, sharey=False, sharex=False,
                                      gridspec_kw = {'width_ratios':[1, 1, 1, 0.2, 0.6]})
    fig1_behav.subplots_adjust(left=0.1, right=0.83, bottom=0.15, wspace=0.65)
    pref_behav_fig(ax, df_behav, df_photo, prefsession=1,
                          barlabeloffset=[0.025, 0.035, 0.045, 0.07],
                          scattersize=scattersize)
    fig1_behav.savefig(savefolder + 'fig1_behav.pdf')

clims = [[-0.15,0.20], [-0.11,0.15]] # when using 'blue' signal

clims = [[-3,4], [-3,3]] # when using 'filt_z''

if make_fig1_photo:
    fig1_photo_NR = fig1_photo(df_heatmap, df_photo, 'NR', 'pref1', clims=clims[0],
                                          peaktype=peaktype, epoch=epoch,
                                          keys_traces = ['pref1_cas_licks_forced', 'pref1_malt_licks_forced'],
                                          keys_lats = ['pref1_cas_lats_all', 'pref1_malt_lats_all'],
                                          scattersize=scattersize)
    
    fig1_photo_PR = fig1_photo(df_heatmap, df_photo, 'PR', 'pref1', clims=clims[1],
                                          peaktype=peaktype, epoch=epoch,
                                          keys_traces = ['pref1_cas_licks_forced', 'pref1_malt_licks_forced'],
                                          keys_lats = ['pref1_cas_lats_all', 'pref1_malt_lats_all'],
                                          scattersize=scattersize)
    
    fig1_photo_NR.savefig(savefolder + 'fig1_photo_NR.pdf')
    fig1_photo_PR.savefig(savefolder + 'fig1_photo_PR.pdf')
    

if make_fig2_behav:

    pref2_behav_fig = plt.figure(figsize=(3.2, 3.2))
    pref2_behav_fig.subplots_adjust(left=0.20, right=0.95, bottom=0.15, wspace=0.65, hspace=0.5) 
    gs =  gridspec.GridSpec(2, 2, figure=pref2_behav_fig)
    
    ax = []
    ax.append(pref2_behav_fig.add_subplot(gs[0, 0]))
    ax.append(pref2_behav_fig.add_subplot(gs[0, 1]))
    ax.append(pref2_behav_fig.add_subplot(gs[1, 0]))
    
    pref_behav_fig(ax, df_behav, df_photo, prefsession=2, dietswitch=True,
                          barlabeloffset=[0.02, 0.02, 0.03, 0.07], gs=gs, f=pref2_behav_fig,
                          scattersize=scattersize)
    pref2_behav_fig.savefig(savefolder + 'fig2_pref2_behav.pdf')

    pref3_behav_fig = plt.figure(figsize=(3.2, 3.2))
    pref3_behav_fig.subplots_adjust(left=0.20, right=0.95, bottom=0.15, wspace=0.6, hspace=0.55)
    gs =  gridspec.GridSpec(2, 2, figure=pref3_behav_fig)

    ax = []
    ax.append(pref3_behav_fig.add_subplot(gs[0, 0]))
    ax.append(pref3_behav_fig.add_subplot(gs[0, 1]))
    ax.append(pref3_behav_fig.add_subplot(gs[1, 0]))
    
    pref_behav_fig(ax, df_behav, df_photo, prefsession=3, dietswitch=True,
                          barlabeloffset=[0.02, 0.02, 0.03, 0.07], gs=gs, f=pref3_behav_fig,
                          scattersize=scattersize)
    pref3_behav_fig.savefig(savefolder + 'fig2_pref3_behav.pdf')
    
    
if make_fig2_photo:
    fig2_pref2_photo = fig2_photo(df_photo, peaktype=peaktype, epoch=epoch, peakkey='peakdiff_2',
                                  scattersize=scattersize)
    fig2_pref2_photo.savefig(savefolder + 'fig2_pref2_photo.pdf')
    
    fig2_pref3_photo = fig2_photo(df_photo, peaktype=peaktype, epoch=epoch,
                                          keys_traces = ['pref3_cas_licks_forced', 'pref3_malt_licks_forced'],
                                          peakkey='peakdiff_3',
                                          scattersize=scattersize)
    fig2_pref3_photo.savefig(savefolder + 'fig2_pref3_photo.pdf')

if make_fig3_summary:
    summaryFig = makesummaryFig(df_behav, df_photo, peaktype=peaktype, epoch=epoch,
                                use_zscore_diff=True,
                                scattersize=scattersize)
    summaryFig.savefig(savefolder + 'summaryfig.pdf')

# For making supplemental Figures
#sacc_behav_fig = pppfig.sacc_behav_fig(df_sacc_behav)

if make_cond_figs:
    cond1_behav_fig, ax = plt.subplots(figsize=(4,3), ncols=2, sharey=True)
    cond1_behav_fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15, wspace=0.2)
    
    cond_licks_fig(ax[0], df_cond1_behav, 'NR')
    cond_licks_fig(ax[1], df_cond1_behav, 'PR')
    ax[0].set_ylabel('Licks', fontsize=8)
        
    cond1_behav_fig.savefig(savefolder + 'cond1_behav.pdf')
#
#    keys=[['cond1_cas1_sip', 'cond1_cas2_sip'],
#         ['cond1_malt1_sip', 'cond1_malt2_sip'],
#         ['cond1_cas1_licks', 'cond1_cas2_licks'],
#         ['cond1_malt1_licks', 'cond1_malt2_licks']]
#    
#    keysbars = [[['cond1_cas1_sip_peak', 'cond1_cas2_sip_peak'], ['cond1_malt1_sip_peak', 'cond1_malt2_sip_peak']],
#                [['cond1_cas1_licks_peak', 'cond1_cas2_licks_peak'], ['cond1_malt1_licks_peak', 'cond1_malt2_licks_peak']]]
#    
#    cond1_photo_sip_fig, ax = plt.subplots(figsize=(6,4), ncols=3, nrows=2)
#
#    pppfig.cond_photo_fig(ax[0][0], df_cond1_photo, 'NR', keys[0], event='Sipper')
#    pppfig.cond_photo_fig(ax[0][1], df_cond1_photo, 'NR', keys[1], event='Sipper')
#    pppfig.cond_photobar_fig(ax[0][2], df_cond1_photo, 'NR', keysbars[0])
#    
#    pppfig.cond_photo_fig(ax[1][0], df_cond1_photo, 'PR', keys[0], event='Sipper')
#    pppfig.cond_photo_fig(ax[1][1], df_cond1_photo, 'PR', keys[1], event='Sipper')
#    pppfig.cond_photobar_fig(ax[1][2], df_cond1_photo, 'PR', keysbars[0])
#    
#    
#    cond1_photo_lick_fig, ax = plt.subplots(figsize=(6,4), ncols=3, nrows=2)
#
#    pppfig.cond_photo_fig(ax[0][0], df_cond1_photo, 'NR', keys[2], event='Licks')
#    pppfig.cond_photo_fig(ax[0][1], df_cond1_photo, 'NR', keys[3], event='Licks')
#    pppfig.cond_photobar_fig(ax[0][2], df_cond1_photo, 'NR', keysbars[1])
#    
#    pppfig.cond_photo_fig(ax[1][0], df_cond1_photo, 'PR', keys[2], event='Licks')
#    pppfig.cond_photo_fig(ax[1][1], df_cond1_photo, 'PR', keys[3], event='Licks')
#    pppfig.cond_photobar_fig(ax[1][2], df_cond1_photo, 'PR', keysbars[1])
#    
#    cond1_photo_sip_fig.savefig(savefolder + 'cond1_photo_sip.pdf')
#    cond1_photo_lick_fig.savefig(savefolder + 'cond1_photo_lick.pdf')

if supp_rep_trace:

    
    figS2_rep_photo = figS2_rep(longtrace)  
    figS2_rep_photo.savefig(savefolder + 'figS2_rep_traces.pdf')

if supp_heatmap:
    print('Supp heatmap')
    figS2_photo_NR = fig1_photo(df_heatmap_sip, df_photo, 'NR', 'pref1', clims=clims[0],
                                          peaktype=peaktype, epoch=epoch,
                                          keys_traces = ['pref1_cas_sip', 'pref1_malt_sip'],
                                          keys_lats = ['pref1_cas_lats_all_fromsip', 'pref1_malt_lats_all_fromsip'],
                                          event='Sipper')
    
    figS2_photo_PR = fig1_photo(df_heatmap, df_photo, 'PR', 'pref1', clims=clims[1],
                                          peaktype=peaktype, epoch=epoch,
                                          keys_traces = ['pref1_cas_sip', 'pref1_malt_sip'],
                                          keys_lats = ['pref1_cas_lats_all_fromsip', 'pref1_malt_lats_all_fromsip'],
                                          event='Sipper')
    
    figS2_photo_NR.savefig(savefolder + 'figS2_photo_NR.pdf')
    figS2_photo_PR.savefig(savefolder + 'figS2_photo_PR.pdf')

#if make_photo_sip_figs:
#    photo_pref1_sipper_fig = pppfig.mainphotoFig(df_reptraces_sip, df_heatmap_sip, df_photo, clims=clims,
#                                                 keys_traces = ['pref1_cas_sip', 'pref1_malt_sip'],
#                                                 keys_bars = ['pref1_cas_sip_peak', 'pref1_malt_sip_peak'],
#                                                 keys_lats = ['pref1_cas_lats_all_fromsip', 'pref1_malt_lats_all_fromsip'],
#                                                 event='Sipper')
#    
#    photo_pref1_sipper_fig.savefig(savefolder + 'pref1_sip_photo.pdf')
#
#if make_photo_licks_figs:
#    
#    photo_pref1_fig = pppfig.mainphotoFig(df_reptraces, df_heatmap, df_photo, clims=clims,
#                                          keys_traces = ['pref1_cas_licks_forced', 'pref1_malt_licks_forced'],
#                                          keys_bars = ['pref1_cas_licks_peak', 'pref1_malt_licks_peak'],
#                                          keys_lats = ['pref1_cas_lats_all', 'pref1_malt_lats_all'])
#    
#    photo_pref1_fig.savefig(savefolder + 'pref1_licks_photo.pdf')