# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:09:09 2021

@author: admin
"""

import dill

try:
    pickle_in = open('C:\\Github\\PPP_analysis\\data\\ppp_cond.pickle', 'rb')
except FileNotFoundError:
    print('Cannot access pickled file')

cond_sessions, rats = dill.load(pickle_in)