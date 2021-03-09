# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:15:46 2021

@author: admin
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.transforms as transforms

# import pandas as pd

from ppp_pub_figs_settings import *
# from ppp_pub_figs_fx import *
# from ppp_pub_figs_supp import *


# import dabest as db

import trompy as tp

# import scipy.io as sio
# import os
# import string
import numpy as np

import dill
import tdt


from scipy.stats import linregress
from scipy.stats import gaussian_kde
from scipy.stats import mannwhitneyu

# Looks for existing data and if not there loads pickled file
try:
    type(sessions)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    try:
        pickle_in = open('C:\\Github\\PPP_analysis\\data\\ppp_pref.pickle', 'rb')
    except FileNotFoundError:
        print('Cannot access pickled file')
    sessions, rats = dill.load(pickle_in)

pref_sessions = {}
for session in sessions:
    x = sessions[session]
    try:
        len(x.data)
        pref_sessions[x.sessionID] = x
    except AttributeError:
        pass

def findpeak(snips):
    time_to_peak = []
    for snip in snips:
        time_to_peak.append(np.argmax(snip[100:])/10)
        
    return(time_to_peak)

def scatter(x, y, ax, color="black"):
    # converts input lists into arrays
    x = np.array(x)
    y = np.array(y)
    
    # removes trials with latencies greater than 20 s (time of maximum peak latency)
    y = y[x<20]
    x = x[x<20]
    
    # ax.scatter(x, y, marker="o", edgecolor=color, facecolor="none")
    ax.scatter(x, y, marker="o", edgecolor="none", facecolor=color, alpha=0.1)
    
    slope, intercept, r, p, se = linregress(x, y)
    
    x_line = ax.get_xlim()
    y_line = [slope*x + intercept for x in x_line]
    ax.plot(x_line, y_line, color=color)
    
    print(r, p)

NR_cas ={"time2peak": [], "latency": []}
NR_malt ={"time2peak": [], "latency": []}
PR_cas ={"time2peak": [], "latency": []}
PR_malt ={"time2peak": [], "latency": []}

for key in pref_sessions.keys():
    s = pref_sessions[key]
    if s.session == "s10":
        
        casdata = s.cas["snips_sipper"]
        maltdata = s.malt["snips_sipper"]
        
        if s.diet == "NR":
            NR_cas["time2peak"].append(findpeak(casdata["filt_z"]))
            NR_cas["latency"].append(casdata["latency"])
            
            NR_malt["time2peak"].append(findpeak(maltdata["filt_z"]))
            NR_malt["latency"].append(maltdata["latency"])
        elif s.diet == "PR":
            PR_cas["time2peak"].append(findpeak(casdata["filt_z"]))
            PR_cas["latency"].append(casdata["latency"])
            
            PR_malt["time2peak"].append(findpeak(maltdata["filt_z"]))
            PR_malt["latency"].append(maltdata["latency"])

x1data = tp.flatten_list(NR_cas["latency"])
y1data = tp.flatten_list(NR_cas["time2peak"])

x2data = tp.flatten_list(NR_malt["latency"])
y2data = tp.flatten_list(NR_malt["time2peak"])

x3data = tp.flatten_list(PR_cas["latency"])
y3data = tp.flatten_list(PR_cas["time2peak"])

x4data = tp.flatten_list(PR_malt["latency"])
y4data = tp.flatten_list(PR_malt["time2peak"])
# f, ax = plt.subplots(ncols=2, figsize=(6, 2), sharey=True)
# f.subplots_adjust(left=0.15, bottom=0.15)


# # ax[0].scatter(x1data, y1data, color=col["nr_cas"])
# # ax[0].scatter(x2data, y2data, color="black")
# scatter(x1data, y1data, ax[0], color="black")
# scatter(x2data, y2data, ax[0], color="grey")



# scatter(x3data, y3data, ax[1], color=col["pr_cas"])
# scatter(x4data, y4data, ax[1], color=col["pr_malt"])

# ax[0].set_ylabel("Time to peak (s)")
# ax[0].set_xlabel("Latency (s)")
# ax[1].set_xlabel("Latency (s)")

# ax[0].set_yticks([0, 5, 10, 15, 20])

# f.savefig("C:\\Users\\jmc010\\Dropbox\\Publications in Progress\\PPP Paper\\04_JNS\\02_revision 1\\revision figs\\peak-vs-latency.jpg")

def remove_long_lats(x, y, threshold=20):
    
    # Checks for long latencies and removes all these entries from both x and y
    
    # converts input lists into arrays
    x = np.array(x)
    y = np.array(y)
    
    # removes trials with latencies greater than 20 s (time of maximum peak latency)
    y = y[x < threshold]
    x = x[x < threshold]
    
    return x, y


def scatter_plus_density(x1, y1, x2, y2, colors=["red", "black"]):

    # Tidies up data by removing long latencies
    x1, y1 = remove_long_lats(x1, y1)
    x2, y2 = remove_long_lats(x2, y2)
    
    # Set up figure grid
    gs = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.1,
                           bottom=0.15, left=0.15, right=0.85, top=0.85,
                           width_ratios=[1,0.2], height_ratios=[0.2,1])
    
    #Initialize figure
    f = plt.figure(figsize=(3,3))
    
    # Create main axis
    main_ax = f.add_subplot(gs[1, 0])
    
    scatter(x1, y1, main_ax, color=colors[0])
    scatter(x2, y2, main_ax, color=colors[1])
    
    main_ax.set_ylabel("Time to peak (s)", fontsize=8)
    main_ax.set_xlabel("Latency (s)", fontsize=8)
    
    # main_ax.tick
    
    # Create axis for latency density plot
    lat_ax = f.add_subplot(gs[0, 0], sharex=main_ax)
    lat_ax.tick_params(labelbottom=False)
    lat_ax.set_yticks([])
    lat_ax.set_ylabel("Density", fontsize=8)
    
    density1 = gaussian_kde(x1)
    density2 = gaussian_kde(x2)
    
    xs = np.linspace(0, lat_ax.get_xlim()[1])
    
    lat_ax.plot(xs, density1(xs), color=colors[0])
    lat_ax.plot(xs, density2(xs), color=colors[1])
    
    lat_ax.axvline(np.median(x1), color=colors[0], linestyle="--")
    lat_ax.axvline(np.median(x2), color=colors[1], linestyle="--")
    
    # Create axis for peak density plot
    peak_ax = f.add_subplot(gs[1, 1], sharey=main_ax)
    peak_ax.tick_params(labelleft=False)
    peak_ax.set_xticks([])
    peak_ax.set_xlabel("Density", fontsize=8)
    
    density1 = gaussian_kde(y1)
    density2 = gaussian_kde(y2)
    
    ys = np.linspace(0, peak_ax.get_ylim()[1])
    
    peak_ax.plot(density1(ys), ys, color=colors[0])
    peak_ax.plot(density2(ys), ys, color=colors[1])
    
    peak_ax.axhline(np.median(y1), color=colors[0], linestyle="--")
    peak_ax.axhline(np.median(y2), color=colors[1], linestyle="--")
    
    label_ax = f.add_subplot(gs[0,1])
    tp.invisible_axes(label_ax)

    label_ax.text(0,0.5,"Maltodextrin", color=colors[1], fontsize=8)
    label_ax.text(0,0.2,"Casein", color=colors[0], fontsize=8)
    
    label_ax.set_ylim([0,1])
    
    return f

f_NR = scatter_plus_density(x1data, y1data, x2data, y2data)
f_PR = scatter_plus_density(x3data, y3data, x4data, y4data, colors=["blue", "green"])

f_NR.savefig("C:\\Users\\jmc010\\Dropbox\\Publications in Progress\\PPP Paper\\04_JNS\\02_revision 1\\revision figs\\peak-vs-latency_NR.pdf")

f_PR.savefig("C:\\Users\\jmc010\\Dropbox\\Publications in Progress\\PPP Paper\\04_JNS\\02_revision 1\\revision figs\\peak-vs-latency_PR.pdf")

u, p = mannwhitneyu(x1data, x2data, alternative="two-sided")
print(u,p)
u, p = mannwhitneyu(y1data, y2data, alternative="two-sided")
print(u,p)

u, p = mannwhitneyu(x3data, x4data, alternative="two-sided")
print(u,p)

u, p = mannwhitneyu(y3data, y4data, alternative="two-sided")
print(u,p)


