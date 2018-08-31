# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:39:12 2018

@author: jaimeHP
"""
import scipy.io as sio
#x = sessions['PPP1-1_s7']



matlabfile = 'R:\\DA_and_Reward\\kp259\\CONVERSION_14th Aug\\NAPH matfiles\\NAPH02_habituation.mat'
#
#a = sio.loadmat(x.matlabfile, squeeze_me=True, struct_as_record=False)

a = sio.loadmat(matlabfile, squeeze_me=True, struct_as_record=False) 


matlabfile = 'R:\\DA_and_Reward\\kp259\\THPH2\\22ndSept\\thph2.3thph2.4distraction.mat'

b = sio.loadmat(matlabfile, squeeze_me=True, struct_as_record=False) 


c = a['output']

d = b['output']

