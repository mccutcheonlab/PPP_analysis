# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:12:59 2020

@author: admin
"""

import dill
import numpy as np
import pandas as pd

import trompy as tp

import matplotlib.pyplot as plt

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
    
def get_snips(sessions, diet, prefsession, sol, event, datatype):
    
    eventkey = 'snips_' + event

    list_of_snips = []
    for key in sessions.keys():
        
        s = sessions[key]
        if s.session == prefsession and s.diet == diet:
            snip_dict = getattr(s, sol)[eventkey]
            try:
                noise = snip_dict['noise']
                snips = [snip for snip, noise in zip(snip_dict[datatype], noise) if not noise]
                snips = tp.resample_snips(snips)
                list_of_snips.append(snips)
            except:
                print(f'Could not extract snips for {key}, {event}')
    
    print(eventkey, diet, prefsession, np.shape(list_of_snips))

    return list_of_snips

def combine_auc_and_latency(session, sol, event, datatype="filt_z", epoch=[100, 149]):
    
    eventkey = "snips_" + event
    start, stop = epoch[0], epoch[1]

    data = getattr(session, sol)[eventkey]
    
    aucs = [np.trapz(snip[start:stop]) for snip, noise in zip(data[datatype], data["noise"]) if not noise]
    lats = [lat for lat, noise in zip(data["latency"], data["noise"]) if not noise]

    return aucs, lats 
    
    

    
s = sessions["PPP1-1_s10"]

aucs, lats = combine_auc_and_latency(s, "cas", "sipper")

dietcol, solcol, sessioncol, auc, lat = [], [], [], [], []

for key in sessions.keys():
    s = sessions[key]

    
    aucs, lats = combine_auc_and_latency(s, "cas", "sipper")
    auc.append(aucs)
    lat.append(lats)
    dietcol.append([s.diet]*len(aucs))
    solcol.append(["cas"]*len(aucs))
    sessioncol.append([s.session]*len(aucs))
    
    aucs, lats = combine_auc_and_latency(s, "malt", "sipper")
    auc.append(aucs)
    lat.append(lats)
    dietcol.append([s.diet]*len(aucs))
    solcol.append(["malt"]*len(aucs))
    sessioncol.append([s.session]*len(aucs))
    

df = pd.DataFrame()
df["diet"] = tp.flatten_list(dietcol)
df["sol"] = tp.flatten_list(solcol)
df["session"] = tp.flatten_list(sessioncol)
df["AUC"] = tp.flatten_list(auc)
df["latency"] = tp.flatten_list(lat)


df2plot = df[(df["session"] == "s10") & (df["diet"] == "PR") & (df["sol"] == "cas")]


f, ax = plt.subplots()
ax.scatter(df2plot["latency"], df2plot["AUC"])


df2plot = df[(df["session"] == "s10") & (df["diet"] == "PR") & (df["sol"] == "malt")]
    
ax.scatter(df2plot["latency"], df2plot["AUC"], color="green")
    
