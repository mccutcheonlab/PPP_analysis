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
    
def compile_lats(x):
    lats = {}
    
    for side in [x.left, x.right]:
        if 'lats' not in side.keys():
            side['lats'] = []   
    try:
        lats['all'] = x.left['lats'] + x.right['lats']
        lats['nans'] = len([n for n in lats['all'] if np.isnan(n)])
        lats['mean'] = np.nanmean(lats['all'])
    except KeyError:
        print('No latencies found for ', x.sessionID)
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
    x.lats = compile_lats(x)        
# this block will run any fx that are specific to certain sessions, e.g.
#    x.choices = choicetest(x)
#    x.pref = prefcalc(x)
    
    
df_sacc_behav = pd.DataFrame([x for x in rats], columns=['rat'])
df_sacc_behav['diet'] = [rats.get(x) for x in rats]
df_sacc_behav.set_index(['rat', 'diet'], inplace=True)

for j, lats, latmean, missed in zip(included_sessions,
                                    ['lats1', 'lats2', 'lats3'],
                                    ['latx1', 'latx2', 'latx3'],
                                    ['missed1', 'missed2', 'missed3']):
    df_sacc_behav[lats] = [sacc_sessions[x].lats['all'] for x in sacc_sessions if sacc_sessions[x].session == j]
    df_sacc_behav[latmean] = [sacc_sessions[x].lats['mean'] for x in sacc_sessions if sacc_sessions[x].session == j]
    df_sacc_behav[missed] = [sacc_sessions[x].lats['nans'] for x in sacc_sessions if sacc_sessions[x].session == j]

pickle_out = open('R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_dfs_sacc.pickle', 'wb')
dill.dump(df_sacc_behav, pickle_out)
pickle_out.close()