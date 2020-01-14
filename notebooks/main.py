# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:14:26 2020

@author: admin
"""

import fx4assembly

test_sessions = metafile2sessions("..\\data\\test.xlsx",
                                  "..\\data\\test",
                                  "..\\data\\",
                                  "..\\output\\")

s = test_sessions['PPP1-7_s10']

process_rat(s)



