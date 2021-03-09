# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:43:27 2021

@author: admin
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.lines as mlines
# import matplotlib.transforms as transforms

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

from scipy.stats import ttest_ind

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
         

start_times = {"PPP1-1_s10": 62052,
               "PPP1-2_s10": 175600,
               "PPP1-3_s10": 99691,
               "PPP1-4_s10": 269571,
               "PPP1-5_s10": 105794,
               "PPP1-6_s10": 258890,
               "PPP1-7_s10": 110880,
               "PPP3-2_s10": 263458,
               "PPP3-3_s10": 78328,
               "PPP3-4_s10": 180053,
               "PPP3-5_s10": 119018,
               "PPP3-8_s10": 109863,
               "PPP4-1_s10": 1017,
               "PPP4-4_s10": 1017,
               "PPP4-6_s10": 92570}

def get_auc(key, makefig=False, figfile=""):

    print("Analysing", key)
    s = pref_sessions[key]
    
    sip0 = min(s.cas["sipper"][0], s.malt["sipper"][0])
    
    start_t = start_times[key] / s.fs
    
    print("Total time analysed is",  sip0-start_t)
    
    data = s.data_filt[int(start_t*s.fs):int(sip0*s.fs)]
    
    if makefig:
        data2plot = data+np.abs(np.min(data)*2)
        if s.diet == "NR":
            color=col["nr_cas"]
        elif s.diet == "PR":
            color = col["pr_cas"]
        f, ax = plt.subplots(figsize=(1.5,0.5))
        ax.plot(data2plot, color=color)
        # ax.text(0,0,key)
        tp.invisible_axes(ax)
        ax.plot([0, 0], [0, 0.1], color="k")
        ax.plot([0, s.fs*5], [0, 0], color="k")
        try:
            f.savefig(figfile)
        except:
            pass

    min_value = np.min(data)
    data = data + np.abs(min_value)
    
    return np.mean(data)

savefolder = "C:\\Users\\jmc010\\Dropbox\\Publications in Progress\\PPP Paper\\04_JNS\\02_revision 1\\revision figs\\"

NR_aucs = []
PR_aucs = []

for key in start_times.keys():
    auc = get_auc(key)
    diet = pref_sessions[key].diet
    if diet == "NR":
        NR_aucs.append(auc)
    elif diet == "PR":
        PR_aucs.append(auc)
    else:
        print("problem assigning AUC to group")
        
f, ax = plt.subplots(figsize=(1.5,2))
f.subplots_adjust(left=0.25, bottom=0.15)
tp.barscatter([NR_aucs, PR_aucs],
              barfacecolor=[col["nr_cas"], col["pr_cas"]],
              barfacecoloroption="individual",
              barlabels = ["NR", "PR"],
              scattersize=20,
              ax=ax)

ax.set_ylabel("Baseline (AUC)")

stats = ttest_ind(NR_aucs, PR_aucs)

f.savefig(savefolder + "baseline.pdf")


### For saving representative figs for revision
# get_auc("PPP1-1_s10", makefig=True,
#         figfile=savefolder + "PPP1-1_trans.pdf")

# get_auc("PPP1-5_s10", makefig=True,
#         figfile=savefolder + "PPP1-5_trans.pdf")



### This code was used to determine start times for each rat by looking for
# first sipper entry and time at which power to LED was stable

# s = pref_sessions["PPP4-6_s10"]

# # Sets indices for power parameters
# if s.box == "1.0":
#     power_index = [2, 6]
# else:
#     power_index = [10, 14]

# # Gets time that first trial starts
# sip0 = min(s.cas["sipper"][0], s.malt["sipper"][0])

# # Extracts parameters from TDT file
# pars = tdt.read_block(s.tdtfile, evtype=['scalars']).scalars.Pars

# # Gets index of first trial start in parameters array
# sip0_pars_index = np.argmin([np.abs(t-sip0) for t in pars.ts])

# f, ax = plt.subplots(nrows=3)
# ax[0].plot(s.data_filt[:int(sip0*s.fs)])
# ax[1].plot(pars.ts[:sip0_pars_index], pars.data[power_index[0]][:sip0_pars_index])
# ax[2].plot(pars.ts[:sip0_pars_index], pars.data[power_index[1]][:sip0_pars_index])


# start_inds=[]
# for i in power_index:
#     presip_power = pars.data[i][:sip0_pars_index]
#     L_samepower = presip_power == presip_power[-1] # creates boolean array of all powers that are same as final power
#     start_inds.append(np.argmax(L_samepower))
    
# start_t = pars.ts[np.max(start_inds)] # finds time 

# total_time = sip0-start_t
# print(total_time)
# data = s.data_filt[int(start_t*s.fs):int(sip0*s.fs)]

# plt.plot(data)
