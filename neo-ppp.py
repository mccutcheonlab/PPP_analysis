# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:04:02 2019

@author: James Rig
"""

import neo

from neo.io import TdtIO
from neo.io import NeoMatlabIO

matfile = 'R:\\DA_and_Reward\\es334\\PPP1\\matfiles\\PPP1-5_s15.mat'

data = NeoMatlabIO(filename=matfile)