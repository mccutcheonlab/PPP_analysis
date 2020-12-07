# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:23:18 2020

@author: admin
"""

import scipy.io as sio
import os
import string
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import trompy as tp

import dill


def variablesnipper(data_filt, fs, events, all_events_baseline,
                                      trialLength=30,
                                      snipfs=10,
                                      preTrial=10,
                                      threshold=8,
                                      peak_between_time=[0, 1],
                                      latency_events=[],
                                      latency_direction='pre',
                                      max_latency=30,
                                      verbose=True,
                                      removenoisefromaverage=True,
                                      **kwargs):
    
   # Parse arguments relating to trial length, bins etc

    if preTrial > trialLength:
        baseline = trialLength / 2
        print("preTrial is too long relative to total trialLength. Changing to half of trialLength.")
    else:
        baseline = preTrial
    
    if 'bins' in kwargs.keys():
        bins = kwargs['bins']
        print(f"bins given as kwarg. Fs for snip will be {trialLength/bins} Hz.")
    else:
        if snipfs > fs-1:
            print('Snip fs is too high, reducing to data fs')
            bins = 0
        else:
            bins = int(trialLength * snipfs)
    
    print('Number of bins is:', bins)
    
    if 'baselinebins' in kwargs.keys():
        baselinebins = kwargs['baselinebins']
        if baselinebins > bins:
            print('Too many baseline bins for length of trial. Changing to length of baseline.')
            baselinebins = int(baseline*snipfs)
        baselinebins = baselinebins
    else:
        baselinebins = int(baseline*snipfs)
    
    if len(events) < 1:
        print('Cannot find any events. All outputs will be empty.')
        return {}
    else:
        if verbose: print('{} events to analyze.'.format(len(events)))

    print(events, all_events_baseline)
    
    # find closest baseline event for each timelocked event
    events_baseline = []
    for event in events:
        events_baseline.append([e for e in all_events_baseline if e < event][-1])
        print(events_baseline[-1], event)
        
    # calculate mean and SD of baseline by taking 10s window butting in 100 bins and working out
    baseline_snips,_ = tp.snipper(data_filt, events_baseline,
                           fs=fs,
                           bins=100,
                           preTrial=10,
                           trialLength=10,
                           adjustBaseline=False)
    
    baseline_mean = np.mean(baseline_snips, axis=1)
    baseline_sd = np.std(baseline_snips, axis=1)
    
    filt_snips,_ = tp.snipper(data_filt, events,
                                   fs=fs,
                                   bins=bins,
                                   preTrial=baseline,
                                   trialLength=trialLength,
                                   adjustBaseline=False)
    
    filt_snips_z = []
    for snip, mean, sd in zip(filt_snips, baseline_mean, baseline_sd):
        filt_snips_z.append([(x-mean)/sd for x in snip])
      
    return filt_snips_z
    

def addvariablebaseline_Z(x):
    
    print(x.rat, x.session)
    
    forced_licks = [licks for licks in x.cas['lickdata']['rStart'] if licks in x.cas['licks-forced']]

    cas = variablesnipper(x.data_filt, x.fs, forced_licks, x.cas["sipper"])
    
    forced_licks = [licks for licks in x.malt['lickdata']['rStart'] if licks in x.malt['licks-forced']]

    malt = variablesnipper(x.data_filt, x.fs, forced_licks, x.malt["sipper"])
    
    return cas, malt


#    Looks for existing data and if not there loads pickled file
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
        x.cas["snips_licks_forced"]["snips_filt_z"], x.malt["snips_licks_forced"]["snips_filt_z"] = addvariablebaseline_Z(x)
    except AttributeError:
        print(x, "no variable snips")
        pass


savefile=True
if savefile == True:
    pickle_out = open('C:\\Github\\PPP_analysis\\data\\ppp_pref.pickle', 'wb')
    dill.dump([sessions, rats], pickle_out)
    pickle_out.close()

# pickle_in = open('C:\\Github\\PPP_analysis\\data\\ppp_test.pickle', 'rb')
# sessions, rats = dill.load(pickle_in)


# s = sessions["PPP4-6_s16"]

# print(s.malt["sipper"])

# forced_licks = [licks for licks in x.cas['lickdata']['rStart'] if licks in s.cas['licks-forced']]

# s.cas["snips_licks_forced"]["snips_filt_z"] = variablesnipper(s.data_filt, s.fs, forced_licks, s.cas["sipper"])

# forced_licks = [licks for licks in x.cas['lickdata']['rStart'] if licks in s.malt['licks-forced']]

# s.malt["snips_licks_forced"]["snips_filt_z"] = variablesnipper(s.data_filt, s.fs, forced_licks, s.malt["sipper"])
    
# x = sessions['PPP1-1_s10']

# forced_licks = [licks for licks in x.cas['lickdata']['rStart'] if licks in x.cas['licks-forced']]

# e = variablesnipper(x.data_filt, x.fs, forced_licks, x.cas["sipper"])


    