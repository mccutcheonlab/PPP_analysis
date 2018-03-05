# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:39:12 2018

@author: jaimeHP
"""

class Foo:
    right = [1,2,3,4,5,6,7]
    left = [0,2.5,4.5,6,7]
    
    first = (lambda right=right: [i for i,x in enumerate(left) if x in right])()
    print(first)