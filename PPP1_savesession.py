# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:55:41 2017

@author: jaimeHP
"""

import dill

pickle_out = open('C:\\Users\\James Rig\\Documents\\rats.pickle', 'wb')
dill.dump(rats, pickle_out)
pickle_out.close()