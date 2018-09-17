# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:54:20 2018

@author: jaimeHP
"""
import dill

import pandas as pd
import numpy as np

import JM_general_functions as jmf
# Looks for existing data and if not there loads pickled file
try:
    type(sessions)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    try:
        pickle_in = open('R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_sacc.pickle', 'rb')
    except FileNotFoundError:
        print('Cannot access pickled file')
    sessions, rats = dill.load(pickle_in)

#sacc_sessions = {}
#for session in sessions:
#    x = sessions[session]
#    try:
#        len(x.data)
#        sacc_sessions[x.sessionID] = x
#        
#    except AttributeError:
#        pass
#
    
def compile_lats(x):
    lats = {}
    try:
        lats['all'] = x.left['lats'] + x.right['lats']
        lats['nans'] = len([n for n in lats['all'] if np.isnan(n)])
        lats['mean'] = np.nanmean(lats['all'])
    except KeyError:
        print('No latencies found for ', x.sessionID)
        print(len(x.left['licks']))
    return lats
    
sacc_sessions = {}
for session in sessions:
    x = sessions[session]
    try:
        len(x.data)
        sacc_sessions[x.sessionID] = x
        
    except AttributeError:
        pass

rats = {}
included_sessions = []
for session in sacc_sessions:
    x = sacc_sessions[session]
    if x.rat not in rats.keys():
        rats[x.rat] = x.diet
    if x.session not in included_sessions:
        included_sessions.append(x.session)
        
for session in sacc_sessions:
    x = sacc_sessions[session]
    x.latency = compile_lats(x)        
# this block will run any fx that are specific to certain sessions, e.g.
#    x.choices = choicetest(x)
#    x.pref = prefcalc(x)
    
    
df_sacc_behav = pd.DataFrame([x for x in rats], columns=['rat'])
df_sacc_behav['diet'] = [rats.get(x) for x in rats]
df_sacc_behav.set_index(['rat', 'diet'], inplace=True)
# latency
    
x = sessions['PPP1-3_s3']

jmf.latencyCalc(x.left['lickdata']['licks'], x.left['sipper'], cueoff=x.left['sipper_off'], lag=0)