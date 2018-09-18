# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:40:30 2018

@author: James Rig
"""

import dill
import pandas as pd

import matplotlib.pyplot as plt
import JM_custom_figs as jmfig

import ppp_pub_figs_fx as pppfig


try:
    type(sessions)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    try:
        pickle_in = open('R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_cond1.pickle', 'rb')
    except FileNotFoundError:
        print('Cannot access pickled file')
    sessions, rats = dill.load(pickle_in)
    
cond1_sessions = {}
for session in sessions:
    x = sessions[session]
    try:
        len(x.data)
        cond1_sessions[x.sessionID] = x
    except AttributeError:
        pass

rats = {}
included_sessions = []
for session in cond1_sessions:
    x = cond1_sessions[session]
    if x.rat not in rats.keys():
        rats[x.rat] = x.diet

for session in cond1_sessions:
    x = cond1_sessions[session]          
    
df_cond1_behav = pd.DataFrame([x for x in rats], columns=['rat'])
df_cond1_behav['diet'] = [rats.get(x) for x in rats]
df_cond1_behav.set_index(['rat', 'diet'], inplace=True)

for cas, malt in zip(['cond1-cas1', 'cond1-cas2'],
                     ['cond1-malt1', 'cond1-malt2']):

    df_cond1_behav[cas] = [cond1_sessions[x].cas['lickdata']['total'] for x in cond1_sessions if cond1_sessions[x].sessiontype == cas]
    df_cond1_behav[malt] = [cond1_sessions[x].malt['lickdata']['total'] for x in cond1_sessions if cond1_sessions[x].sessiontype == malt]

df_cond1_behav['cond1-cas-all'] = df_cond1_behav['cond1-cas1'] + df_cond1_behav['cond1-cas2']
df_cond1_behav['cond1-malt-all'] = df_cond1_behav['cond1-malt1'] + df_cond1_behav['cond1-malt2']

df_cond1_photo = pd.DataFrame([x for x in rats], columns=['rat'])
df_cond1_photo['diet'] = [rats.get(x) for x in rats]
df_cond1_photo.set_index(['rat', 'diet'], inplace=True)




pickle_out = open('R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_dfs_cond1.pickle', 'wb')
dill.dump([df_cond1_behav, df_cond1_photo], pickle_out)
pickle_out.close()
