# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:20:24 2020

@author: admin
"""

import dill

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
    
for key in sessions.keys():
    s = sessions[key]
    s.data = 'removed'
    s.dataUV = 'removed'
    s.data_filt = 'removed'
    
pickle_out = open('C:\\Github\\PPP_analysis\\data\\ppp_pref_reduced.pickle', 'wb')
dill.dump([sessions, rats], pickle_out)
pickle_out.close()
