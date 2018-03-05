# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:41:24 2018

@author: jaimeHP
"""
def dividelicks(licks, time):
    before = [x for x in licks if x < time]
    after = [x for x in licks if x > time]
    
    return before, after



for i in rats:
    for j in ['s10']:
        
        x = rats[i].sessions[j]

        x.left['licks-forced'], x.left['licks-free'] = dividelicks(x.left['licks'], x.both['sipper'][0])
        x.right['licks-forced'], x.right['licks-free'] = dividelicks(x.right['licks'], x.both['sipper'][0])

df = pd.DataFrame([x for x in rats])
