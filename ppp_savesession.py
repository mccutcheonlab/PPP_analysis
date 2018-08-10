# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:55:41 2017

@author: jaimeHP
"""

import dill

pickle_out = open('R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_rats.pickle', 'wb')
dill.dump(sessions, pickle_out)
pickle_out.close()