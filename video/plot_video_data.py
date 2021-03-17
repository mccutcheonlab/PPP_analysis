#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:09:40 2020

@author: jaime
"""
import dill
import numpy as np
# import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy.stats.distributions import norm
import pandas as pd

import trompy as tp

from scipy import stats
from subprocess import PIPE, run

Rscriptpath = 'C:\\Program Files\\R\\R-4.0.3\\bin\\Rscript'
statsfolder = 'C:\\Github\\PPP_analysis\\stats\\'

# from sympy import *
# from sympy.geometry import *

# from ppp_pub_figs_settings import *

almost_black = mpl.colors.to_rgb('#262626')

## Colour scheme
col={}
col['nr_cas'] = 'xkcd:silver'
col['nr_malt'] = 'white'

col['pr_cas'] = 'xkcd:blue'
col['pr_malt'] = 'xkcd:sky blue'


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

def get_good_xys(df, threshold=0.9, verbose=False):
    """
    Returns xy coordinates from deeplabcut only when high likelihood.

    Parameters
    ----------
    df : Dataframe
        Includes xy coordinates and likelihood column.
    threshold : Float, optional
        Likelihood threshold for inclusion. The default is 0.9.
    verbose : Bool, optional
        Prints number of frames excluded. The default is False.

    Returns
    -------
    None.

    """
    
    L = df['likelihood'] > threshold
    F = df.index[df['likelihood'] > threshold].tolist()
    df_hi = df[L]
    
    x = [int(xval) for xval in df_hi['x']]
    y = [int(yval) for yval in df_hi['y']]
    if verbose:
        print(f"There are {sum(L)} good frames out of a total of {len(df)}.")
    
    return (x, y, F)

def plot_heatmap(data, threshold=0.01, ax=None, img=None, opacity=1):
    """
    Plots heatmap showing position of rat in chamber for whole session and cues.
    
    Parameters
    ----------
    data : Dictionary
        Dictionary from video data including position of nose and cues.
    ax : Axis object, optional
        Axis from matplotlib. The default is None.
    img : String
        Path to image file to be used as background. The default is None.
    opacity : Float
        Opacity of heatmap, can be changed whern image used. The default is 1.

    Returns
    -------
    None.

    """

    if ax == None:
        f, ax = plt.subplots()
        
    if img==None:
        ax.plot([0, 640, 640, 0, 0], [480, 480, 0, 0, 480], color='k', linewidth=3)
        
    if img != None:
        try:
            img=mpimg.imread(img)
            ax.imshow(img)
        except: pass
    
    (x, y, F) = get_good_xys(data['nose'], threshold=threshold)
        
    sns.kdeplot(x, y, cmap="Reds", shade=True, bw = 25, shade_lowest=False, n_levels = 50, ax=ax, alpha=opacity)
    # ax = sns.kdeplot(x, y, kernel="gau", bw = 25, cmap="Reds", n_levels = 50, shade=True, shade_lowest=False, gridsize=100)
    ax.set_frame_on(False)
    plot_cues(ax, data, scale=1)

    # ax.plot(np.median(x), np.median(y), marker='o', color='k', markersize=20)
    
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_xticks([])
    ax.set_yticks([])
  
def plot_cues(ax, xydict, scale=8):
    casx, casy = [coord/scale for coord in xydict['cas_pos']]
    maltx, malty = [coord/scale for coord in xydict['malt_pos']]
    
    # ax.plot(casx, casy, marker='o', markerfacecolor='k', markersize=3, markeredgecolor='white')
    # ax.plot(maltx, malty, marker='o', markerfacecolor='grey', markersize=3, markeredgecolor='white')
    
    ax.plot(casx, casy, marker='*', markerfacecolor='white', markeredgecolor='white', markersize=4)
    ax.plot(maltx, malty, marker='+', markerfacecolor='white', markeredgecolor='white', markersize=4)

def calc_dist(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def convertfrompos2dist(data, threshold=0.95):
    (x, y, F) = get_good_xys(data['nose'], threshold=threshold)
    
    cas_dist, malt_dist = [], []
    for X, Y in zip(x, y):
        cas_dist.append(calc_dist((X, Y), data['cas_pos']))
        malt_dist.append(calc_dist((X, Y), data['malt_pos']))
            
    return cas_dist, malt_dist
    
def plot_median_dists(data, ax=None):
    # Initializes dictionaries to store data relating to casein and maltodextrin
    cas_dist_med, malt_dist_med = {}, {}
    
    for d in [cas_dist_med, malt_dist_med]:
        d['NR'] = []
        d['PR'] = []
        
    for d in data:
        
        diet = d['diet']
        
        cas_dist, malt_dist = convertfrompos2dist(d)
        # (x, y, F) = get_good_xys(d['nose'], threshold=0.95)
        # cas_dist, malt_dist = [], []
        # for X, Y in zip(x, y):
        #     cas_dist.append(calc_dist((X, Y), d['cas_pos']))
        #     malt_dist.append(calc_dist((X, Y), d['malt_pos']))
        
        cas_dist_med[diet].append(np.median(cas_dist))
        malt_dist_med[diet].append(np.median(malt_dist))
    
    NR = [cas_dist_med['NR'], malt_dist_med['NR']]
    PR = [cas_dist_med['PR'], malt_dist_med['PR']]
    
    if ax == None:
        f, ax = plt.subplots()
        
    tp.barscatter(tp.data2obj2D([NR, PR]), ax=ax, paired=True,
                  barfacecoloroption="individual",
                  barfacecolor=["grey", "w", "blue", "xkcd:light blue"],
                  barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                  grouplabel=['NR', 'PR'],
                  grouplabeloffset=-0.03
                  )
    
    ax.set_ylabel("Median distance from sipper (pixels)")

def get_kde_estimate(data, lim):
    
    d = np.array(data, dtype=float)
    dvals = np.linspace(0, lim)
    
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(d[:, None])
    log_prob = kde.score_samples(dvals[:, None])
    
    return np.argmax(log_prob)*lim/50

def get_max_fromkdeplot(ax):
    
    xy = []

    for line in ax.lines:
        x = line.get_xdata() # Get the x data of the distribution
        y = line.get_ydata() # Get the y data of the distribution
    
        maxid = np.argmax(y) # The id of the peak (maximum of y data)
        
        xy.append([x[maxid], y[maxid]])
    
    return xy

def plot_kde_dists(data, ax=None, threshold=0.01):
    # Initializes dictionaries to store data relating to casein and maltodextrin
    cas_dist, malt_dist = {}, {}
    
            
    if ax == None:
        f, ax = plt.subplots()
    
    for d in [cas_dist, malt_dist]:
        d['NR'] = []
        d['PR'] = []
        
    for d in data:
        
        diet = d['diet']
        
        (x, y, F) = get_good_xys(d['nose'], threshold=threshold)
        
        (X, Y) = (get_kde_estimate(x, 640), get_kde_estimate(y, 480))
        
        cas_dist[diet].append(calc_dist((X, Y), d['cas_pos']))
        
        malt_dist[diet].append(calc_dist((X, Y), d['malt_pos']))

    
    NR = [cas_dist['NR'], malt_dist['NR']]
    PR = [cas_dist['PR'], malt_dist['PR']]

    tp.barscatter(tp.data2obj2D([NR, PR]), ax=ax, paired=True,
                  barfacecoloroption="individual",
                  barfacecolor=[col['nr_cas'], col['nr_malt'], col['pr_cas'], col['pr_malt']],
                  barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                  grouplabel=['NR', 'PR'],
                  grouplabeloffset=-0.04,
                  scattersize=40,
                  xfontsize=6
                  
                  )
    
    ax.set_ylabel("Average distance from sipper (pixels)")
    
    
    
    return [NR, PR]

def plot_kde_representative(data, ax=None, colors=['blue', 'orange'], labels=True, threshold=0.01):
        
    if ax == None:
        f, ax = plt.subplots()
    
    c, m = convertfrompos2dist(data, threshold=threshold)
    
    sns.kdeplot(c, ax=ax, bw=10, color=colors[0], label='Casein')
    sns.kdeplot(m, ax=ax, bw=10, color=colors[1], linestyle='dashed',label='Maltodextrin')
    
    ax.legend(loc='upper right', fontsize=6, frameon=False)
    
    ax.set_ylabel('Density')
    ax.set_xlabel('Distance from sipper (pixels)')
    
    if labels:
        plot_vline_and_text(ax, 0, color=colors[0])
        plot_vline_and_text(ax, 1, color=colors[1])
    
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # ax.legend(('Casein', 'Malt'), loc='upper center')


def calc_total_dist(data, threshold=0.01, calibration=None):
    
    (x, y, F) = get_good_xys(data['nose'], threshold=threshold)
    
    dist = []
    xn, yn = x[0], y[0]
    
    for idx in range(1, len(F)):
        dist.append(calc_dist((xn, yn), (x[idx], y[idx])))
        xn, yn = x[idx], y[idx]
        
        
    try:
        dist = np.divide(dist, calibration)
    except: pass

    total_dist = sum(dist)
        
    print(f'Total distance (in pixels) was {total_dist}')
    
    return dist, total_dist

def plot_vline_and_text(ax, line_number, color='blue', xoffset=5):
    
    x, y = get_max_fromkdeplot(ax)[line_number]
    
    ax.axvline(x, linestyle='dashed', color=color)
    ax.annotate(f"{x:.2f} px", (x+xoffset, y))

MASTERFOLDER = "./"

pickle_in = MASTERFOLDER + "PPP_video_data.pickle"
with open(pickle_in, "rb") as dill_file:
    PPP_video_data = dill.load(dill_file)

plot_all_heat_maps = False

if plot_all_heat_maps:
    f1, ax = plt.subplots(ncols=14, figsize=(16, 2))
    for idx, data in enumerate(PPP_video_data):
        (x, y, F) = get_good_xys(data['nose'], threshold=0.95)
        
        axis = ax[idx]
        sns.kdeplot(x, y, cmap="Reds", shade=True, bw = 25, shade_lowest=False, n_levels = 50, ax=axis)
        # ax = sns.kdeplot(x, y, kernel="gau", bw = 25, cmap="Reds", n_levels = 50, shade=True, shade_lowest=False, gridsize=100)
        axis.set_frame_on(False)
        plot_cues(axis, data, scale=1)
        axis.set_xlim(0, 640)
        axis.set_ylim(480, 0)
        axis.set_xticks([])
        axis.set_yticks([])
        ax.plot([0, 640, 640, 0, 0], [480, 480, 0, 0, 480], color='k', linewidth=3)
        
        ax.set_title(data['rat'])

    f1.savefig(MASTERFOLDER + "heatmaps.pdf")

def plot_dist_moved(data, ax=None, threshold=0.01, calibration=1):
    
    print('Yo')
    
    if ax == None:
        f, ax = plt.subplots()
    
    NR, PR = [], []
    
    for d in data:
        

        
        _, total_dist = calc_total_dist(d, threshold=threshold)
        
        total_dist = total_dist / calibration
        
        if d['diet'] == 'NR':
            NR.append(total_dist)
        else:
            PR.append(total_dist)

    tp.barscatter([NR, PR], ax=ax,
                   barfacecoloroption="individual",
                   barfacecolor=[col['nr_cas'], col['pr_cas']],
                   # barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                   grouplabel=['NR', 'PR'],
                  # grouplabeloffset=-0.04,
                  scattersize=20,
                  xfontsize=6
                  
                  )
    
    ax.set_ylabel("Total distance moved (pixels)")
    
    return NR, PR


THRESHOLD=0.95

make_pos_fig=False

if make_pos_fig:
    f = plt.figure(figsize=(4, 5), constrained_layout=False)
    gs_top=f.add_gridspec(nrows=2, ncols=2, left=0.13, right=0.9, top=0.95, bottom=0.45, wspace=0.215, hspace=0.4)
    
    
    img="C:/Github/PPP_video/PPP1-171017-081744_Eelke-171027-111329_Cam2.jpg"
    
    ax1 = f.add_subplot(gs_top[0,0])
    plot_heatmap(PPP_video_data[4], ax=ax1, img=img, opacity=0.75, threshold=THRESHOLD)
    ax1.set_title('Non restricted (NR)')    
    
    ax2 = f.add_subplot(gs_top[1,0])
    plot_heatmap(PPP_video_data[0], ax=ax2, img=img, opacity=0.75, threshold=THRESHOLD)
    ax2.set_title('Protein restricted (PR)')
    
    ax3 = f.add_subplot(gs_top[0,1])
    plot_kde_representative(PPP_video_data[4], ax=ax3, colors=['k', 'xkcd:grey'], labels=False, threshold=THRESHOLD)
    ax3.set_xlabel("")
    
    ax4 = f.add_subplot(gs_top[1,1])
    plot_kde_representative(PPP_video_data[0], ax=ax4, colors=['k', 'grey'], labels=False, threshold=THRESHOLD)
    ax4.set_xlabel("Distance from sipper (mm)")
    
    gs_bottom = f.add_gridspec(nrows=1, ncols=2, left=0.15, right=0.85, top=0.35, wspace=0.5, width_ratios=[1, 0.5])
    
    ax5 = f.add_subplot(gs_bottom[0,0])
    kde_dists_NR, kde_dists_PR = plot_kde_dists(PPP_video_data, ax=ax5, threshold=THRESHOLD)
    ax5.set_ylabel("Average distance from sipper (mm)")
    
    ax6 = f.add_subplot(gs_bottom[0,1])
    total_dist_NR, total_dist_PR = plot_dist_moved(PPP_video_data, ax=ax6, threshold=THRESHOLD, calibration=1000)
    ax6.set_ylabel("Total distance moved (m)")
    
    
    f.savefig("C:\\Users\\jmc010\\Dropbox\\Publications in Progress\\PPP Paper\\04_JNS\\Figs\\figS2_tracking.pdf")

# data = plot_kde_dists(PPP_video_data)


def ppp_2wayANOVA(df, csvfile):
    
    # df = extractandstack(df, cols, new_cols=['rat', 'diet', 'substance', 'licks'])
    df.to_csv(csvfile)
    result = run([Rscriptpath, "--vanilla", "ppp_licksANOVA.R", csvfile], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    print(result.returncode, result.stderr, result.stdout)
    return result



do_stats = True
if do_stats:
    
    kde_dists_NR, kde_dists_PR = plot_kde_dists(PPP_video_data, threshold=THRESHOLD)
    
    ratkey = np.arange(0,14).tolist() * 2
    dietkey = ['NR']*6 + ['PR']*8 + ['NR']*6 + ['PR']*8
    subskey = ['kde_cas'] * 14 + ['kde_malt'] * 14
    datakey = kde_dists_NR[0] + kde_dists_PR[0] + kde_dists_NR[1] + kde_dists_PR[1]
    

    kde_stats_df = pd.DataFrame({'rat': ratkey, 'diet': dietkey, 'substance': subskey, 'licks': datakey})
    
    ppp_2wayANOVA(kde_stats_df, "C:\\Github\\PPP_video\\kde_stats.csv")
    
    
    
    total_dist_NR, total_dist_PR = plot_dist_moved(PPP_video_data, threshold=THRESHOLD, calibration=1000)
    print(stats.ttest_ind(total_dist_NR, total_dist_PR))


# dist, total_dist = calc_total_dist(PPP_video_data[0], calibration = 1000)

# dist, total_dist = calc_total_dist(PPP_video_data[0])

def make_likelihood_fig(df, threshold=0.01, ax=None):
    
    if ax == None:
        f, ax = plt.subplots()
    
    data = np.array([d for d in df['nose']['likelihood'] if d > threshold])
    
    sns.kdeplot(data, ax=ax, bw=0.01)
    ax.set_xlim([-0.1, 1.1])
    
def plot_tracking(x, y, ax=None):
    
    if ax == None:
        f, ax = plt.subplots()
    
    ax.plot(x, y)
        
    
def plot_distance_hist(x, y, ax=None):
    
    if ax == None:
        f, ax = plt.subplots()

    dist = []
    xn, yn = x[0], y[0]
    
    for idx in range(1, len(x)):
        dist.append(calc_dist((xn, yn), (x[idx], y[idx])))
        xn, yn = x[idx], y[idx]

    total_dist = sum(dist)

    sns.kdeplot(dist, ax=ax)
    
    ax.text(0.95, 0.95, f"Dist. = {total_dist:.2f}", transform=ax.transAxes, ha='right')
    


def make_tracking_summary(df, threshold=0.01, axes=None):
    
    (x, y, F) = get_good_xys(df['nose'], threshold=threshold)
    
    if axes == None:
        f, ax = plt.subplots(ncols=4, figsize=(6,2))
        
    make_likelihood_fig(df, ax=ax[0], threshold=threshold)
    
    plot_tracking(x, y, ax=ax[1])
    
    plot_heatmap(df, ax=ax[2], threshold=threshold)
    
    plot_distance_hist(x, y, ax=ax[3])
    
    


# make_tracking_summary(PPP_video_data[0])

# make_tracking_summary(PPP_video_data[0], threshold=0.9)

# make_tracking_summary(PPP_video_data[0], threshold=0.99)


def make_distance_df(df, threshold=0.01):
    
    (x, y, F) = get_good_xys(df['nose'], threshold=threshold)
    
    dist, _ = calc_total_dist(df, threshold=threshold)
    
    return pd.DataFrame({'F': F, 'x': x, 'y': y, 'dist': dist+[0]})
    
        
# distdf = make_distance_df(PPP_video_data[0], threshold=0.9)
    

# data = PPP_video_data[0]
# (x, y, F) = get_good_xys(data['nose'], threshold=0.95)

# x = np.array(x, dtype=float)
# y = np.array(x, dtype=float)

# xvals = np.linspace(0, 640)
# yvals = np.linspace(0, 480)



# kde = KernelDensity(bandwidth=0.2, kernel='gaussian')

# kde.fit((x[:, np.newaxis]))
# log_prob = kde.score_samples(xvals[:, np.newaxis])
# print(np.argmax(log_prob)*640/50) # USE THIS FOR X AND Y DIMENSION TO CALCULATE MOST LIKELY PLACE, THEN CALC DIST TO EACH CUE
# pde = np.exp(log_prob)

# xy = [(x,y) for x, y in zip(x, y)]
# xyvals = [(x,y) for x, y in zip(xvals, yvals)]

# kde = KernelDensity(bandwidth=0.2, kernel='gaussian').fit(xy)
# log_prob = kde.score_samples(xyvals)
# print(np.argmax(log_prob))

# fig, ax = plt.subplots(ncols=2)
# fig.subplots_adjust(wspace=0.05)

# bins = range(0, 500, 20)

# cas_hist = np.histogram(cas_dist, bins=bins)
# malt_hist = np.histogram(malt_dist, bins=bins)
# # malt_bins = [-b for b in bins]
# ax[0].set_xlim([500, 0])
# ax[1].set_xlim([0, 500])
# ax[0].plot(bins[:-1], cas_hist[0])
# ax[1].plot(bins[:-1], malt_hist[0])

# ax[1].set_yticks([])
# ax[1].spines['left'].set_visible(False)

# for axis in [ax[0], ax[1]]:
#     axis.set_ylim([-20, 4200])
#     axis.spines['right'].set_visible(False)
#     axis.spines['top'].set_visible(False)
    
# ax[0].set_xlabel('Distance from Casein cue (px)')
# ax[1].set_xlabel('Distance from Maltodextrin cue (px)')
# ax[0].set_ylabel('Frequency')


        
        
        
        # x = np.array(x, dtype=float)
        # y = np.array(x, dtype=float)
        
        # xvals = np.linspace(0, 640)
        # yvals = np.linspace(0, 480)
        
        # kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
        # x_d = np.linspace(0, 640)
        # kde.fit(x[:, None])
        # log_prob = kde.score_sampls(x_d[:, None])
        
        # data = PPP_video_data[0]
        # (x, y, F) = get_good_xys(data['nose'], threshold=0.95)
        
        # x = np.array(x, dtype=float)
        # y = np.array(x, dtype=float)
        
        # xvals = np.linspace(0, 640)
        # yvals = np.linspace(0, 480)
        
        
        
        # kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
        
        # kde.fit((x[:, np.newaxis]))
        # log_prob = kde.score_samples(xvals[:, np.newaxis])
        # print(np.argmax(log_prob)*640/50) # USE THIS FOR X AND Y DIMENSION TO CALCULATE MOST LIKELY PLACE, THEN CALC DIST TO EACH CUE
        # pde = np.exp(log_prob)

        
        
        # cas_dist, malt_dist = [], []
        # for X, Y in zip(x, y):
        #     cas_dist.append(calc_dist((X, Y), d['cas_pos']))
        #     malt_dist.append(calc_dist((X, Y), d['malt_pos']))
        
        # cas_dist_med[diet].append(np.median(cas_dist))
        # malt_dist_med[diet].append(np.median(malt_dist))

# # https://zbigatron.com/generating-heatmaps-from-coordinates/
