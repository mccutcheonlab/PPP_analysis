# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:04:03 2018

For analysis of conditioning sessions

@author: James Rig
"""

import scipy.io as sio
import JM_general_functions as jmf
import JM_custom_figs as jmfig

import os
import string
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import dill

try:
    type(sessions)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    try:
        pickle_in = open('R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_rats_cond1.pickle', 'rb')
    except FileNotFoundError:
        print('Cannot access pickled file')
    sessions, rats = dill.load(pickle_in)
    
cond_sessions = {}
for session in sessions:
    x = sessions[session]
    try:
        len(x.data)
        cond_sessions[x.sessionID] = x
        
    except AttributeError:
        pass

rats = {}

for session in cond_sessions:
    x = cond_sessions[session]
    if x.rat not in rats.keys():
        rats[x.rat] = x.diet

cas_sessions = ['cond1-cas1', 'cond1-cas2']
malt_sessions = ['cond1-malt1', 'cond1-malt2']  

df1 = pd.DataFrame([x for x in rats])
df1.insert(1,'diet', [rats.get(x) for x in rats])

for cas, malt, n in zip(cas_sessions, malt_sessions, [2,4]):
    df1.insert(n, cas, [cond_sessions[x].cas['lickdata']['total'] for x in cond_sessions if cond_sessions[x].sessiontype == cas])
    df1.insert(n+1, malt, [cond_sessions[x].malt['lickdata']['total'] for x in cond_sessions if cond_sessions[x].sessiontype == malt])
