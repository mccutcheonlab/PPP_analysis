# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:54:20 2018

@author: jaimeHP
"""
import dill
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
#df_sacc