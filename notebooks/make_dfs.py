# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:07:59 2020

@author: admin
"""

import dill

pickle_in = open('C:\\Github\\PPP_analysis\\data\\ppp_pref.pickle', 'rb')
sessions, rats = dill.load(pickle_in)
