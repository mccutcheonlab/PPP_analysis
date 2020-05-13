# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:51:21 2019

@author: jmc010
"""

import numpy as np


def latencyCalc(licks, cueon, cueoff=10, nolat=np.nan, lag=3):
    if type(cueoff) == int:
        cueoff = [i+cueoff for i in cueon]
    lats=[]
    for on,off in zip(cueon, cueoff): 
        try:
            currentlat = [i-(on+lag) for i in licks if (i>on) and (i<off)][0]
        except IndexError:
            currentlat = nolat
        lats.append(currentlat)

    return(lats)


