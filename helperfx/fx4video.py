# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:03:27 2021

@author: admin
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

from settings4figs import *

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
    

    sns.kdeplot(x, y=y, cmap="Reds", fill=True, bw_method="scott", bw_adjust= 1.7, thresh=0.1, n_levels = 50, ax=ax, alpha=opacity)
    ax.set_frame_on(False)
    plot_cues(ax, data, scale=1)
    
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_xticks([])
    ax.set_yticks([])
  
def plot_cues(ax, xydict, scale=8):
    casx, casy = [coord/scale for coord in xydict['cas_pos']]
    maltx, malty = [coord/scale for coord in xydict['malt_pos']]
    
    ax.plot(casx, casy, marker='*', markerfacecolor='white', markeredgecolor='white', markersize=3)
    ax.plot(maltx, malty, marker='+', markerfacecolor='white', markeredgecolor='white', markersize=3)

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

    
    NR = [malt_dist['NR'], cas_dist['NR']]
    PR = [malt_dist['PR'], cas_dist['PR']]

    tp.barscatter(tp.data2obj2D([NR, PR]), ax=ax, paired=True,
                  barfacecoloroption="individual",
                  barfacecolor=[col['nr_malt'], col['nr_cas'], col['pr_malt'], col['pr_cas']],
                  barlabels=['Malt', 'Cas', 'Malt', 'Cas'],
                  grouplabel=['NR', 'PR'],
                  grouplabeloffset=-0.07,
                  scattersize=30,
                  xfontsize=5              
                  )
    
    ax.set_ylabel("Average distance from sipper (pixels)")

    return [NR, PR]

def plot_kde_representative(data, ax=None, colors=['blue', 'orange'], labels=True, threshold=0.01):
        
    if ax == None:
        f, ax = plt.subplots()
    
    c, m = convertfrompos2dist(data, threshold=threshold)
    
    sns.kdeplot(c, bw_method="scott", bw_adjust= 1.7, thresh=0.1, ax=ax, color=colors[0], label='Casein')
    sns.kdeplot(m, bw_method="scott", bw_adjust= 1.7, thresh=0.1, ax=ax, color=colors[1], linestyle='dashed',label='Maltodextrin')
    
    ax.legend(loc=(0.4,0.5), fontsize=6, frameon=False)
    
    ax.set_ylabel('Density')
    ax.set_xlabel('Distance from sipper (pixels)')
    
    if labels:
        plot_vline_and_text(ax, 0, color=colors[0])
        plot_vline_and_text(ax, 1, color=colors[1])
    
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xlim([-90, 520])

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
    
def plot_dist_moved(data, ax=None, threshold=0.01, calibration=1):
    
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
                  scattersize=10,
                  xfontsize=6
                  
                  )
    
    ax.set_ylabel("Total distance moved (pixels)")
    
    return NR, PR

if __name__ == "__main__":
    
    pickle_in = "..\\video\\PPP_video_data.pickle"
    with open(pickle_in, "rb") as dill_file:
        PPP_video_data = dill.load(dill_file)
    
    THRESHOLD=0.95
    
    fig2C = plt.figure(figsize=(2.7, 2.3), constrained_layout=False)
    gs=fig2C.add_gridspec(nrows=2, ncols=2, left=0.13, right=0.9, top=0.9, bottom=0.2, wspace=0.215, hspace=0.4)
    
    img="..\\video\\PPP1-171017-081744_Eelke-171027-111329_Cam2.jpg"
    
    # ax1 = fig2C.add_subplot(gs[0,0])
    # plot_heatmap(PPP_video_data[4], ax=ax1, img=img, opacity=0.75, threshold=THRESHOLD)
    # # ax1.set_title('Non restricted (NR)')  
    
    # ax2 = fig2C.add_subplot(gs[1,0])
    # plot_heatmap(PPP_video_data[0], ax=ax2, img=img, opacity=0.75, threshold=THRESHOLD)
    # # ax2.set_title('Protein restricted (PR)')
    
    ax3 = fig2C.add_subplot(gs[0,1])
    plot_kde_representative(PPP_video_data[4], ax=ax3, colors=['k', 'xkcd:grey'], labels=False, threshold=THRESHOLD)
    ax3.set_xlabel("")
    
    ax4 = fig2C.add_subplot(gs[1,1])
    plot_kde_representative(PPP_video_data[0], ax=ax4, colors=['k', 'grey'], labels=False, threshold=THRESHOLD)
    ax4.set_xlabel("Distance from sipper (mm)")
    
    fig2D, ax = plt.subplots(figsize=(1.15, 1.5), gridspec_kw={"left": 0.35, "right": 0.95, "bottom": 0.2, "top": 0.80})
    
    kde_dists_NR, kde_dists_PR = plot_kde_dists(PPP_video_data, ax=ax, threshold=THRESHOLD)
    ax.set_ylabel("Average distance from sipper (mm)")
    
    fig2D.savefig(savefolder + "fig2D.pdf")