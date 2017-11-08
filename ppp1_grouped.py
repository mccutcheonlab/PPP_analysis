# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 08:47:56 2017

@author: Jaime
"""

# Analysis of PPP1 grouped data
# Need to run PPP1_analysis first to load sessions into
# Choice data
import string

def choicetest(x):
    choices = []
    for trial, trial_off in zip(x.trialsboth, x.trialsboth_off):
        leftlick = [x for x in x.licksL if (x > trial) and (x < trial_off)]
        rightlick = [x for x in x.licksR if (x > trial) and (x < trial_off)]
        if len(leftlick) > 0:
            if len(rightlick) > 0:
                if leftlick < rightlick:
                    choices.append(x.bottleL[:3])
                else:
                    choices.append(x.bottleR[:3])
            else:
                choices.append(x.bottleL[:3])
        elif len(rightlick) > 0:
            choices.append(x.bottleR[:3])
        else:
            choices.append('missed')
    
    return choices

def prefcalc(x):
    cas = sum([1 for trial in x.choices if trial == 'cas'])
    malt = sum([1 for trial in x.choices if trial == 'mal'])
    pref = cas/(cas+malt)
    print(x.rat)
    print(pref)
    
    return pref

pref1_data = {}
  
for i in rats:
    for j in ['s11']:
        x = rats[i].sessions[j]
        ratkey = i
              
        x.choices = choicetest(x)
        pref1_data[i]['pref'].append(prefcalc(x))
        
        pref1_data[i]['casmean']



