# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:39:12 2018

@author: jaimeHP
"""

x = sessions['PPP1-1_s7']



x.matlabfile
#
#a = sio.loadmat(x.matlabfile, squeeze_me=True, struct_as_record=False)

a = sio.loadmat(x.matlabfile) 