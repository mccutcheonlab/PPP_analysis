# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:39:12 2018

@author: jaimeHP
"""





def dividelicks(licks, time):
    before = [x for x in licks if x < time]
    after = [x for x in licks if x > time]
    
    return before, after

x.left['licks-forced'], x.left['licks-free'] = dividelicks(x.left['licks'], x.both['sipper'][0])


print(len(x.left['licks-forced']))

print(len(x.left['licks-free']))

x.right['licks-forced'], x.right['licks-free'] = dividelicks(x.right['licks'], x.both['sipper'][0])

print(len(x.right['licks-forced']))

print(len(x.right['licks-free']))