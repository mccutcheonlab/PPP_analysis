# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:17:03 2018

@author: James Rig
"""

trial = x.left['snips_sipper']

for i,_ in enumerate(x.left['snips_sipper']['blue']):
   
    plt.figure()
    ax = plt.subplot()
    jmfig.trialsMultFig(ax, [trial['blue'][i], trial['uv'][i]], pps=1, preTrial=10, scale=5, 
              linecolor=['m', 'b'],)
    ax.text(25,.1,i)


# Code to make best trial into figure and save as eps

i=13

plt.figure()
ax = plt.subplot()
jmfig.trialsMultFig(ax, [trial['uv'][i], trial['blue'][i]], pps=1, preTrial=10, scale=5, 
          linecolor=['m', 'b'],)


plt.savefig('R:/DA_and_Reward/es334/PPP1/output/' + x.rat + 'trial ' + str(i) + '.eps', format='eps', dpi=1000)


# To make eps of multiple trials
plt.figure()
ax = plt.subplot()
jmfig.trialsMultShadedFig(ax, [x.left['snips_sipper']['uv'], x.left['snips_sipper']['blue']],
                                  x.pps,
                                  noiseindex = x.left['snips_sipper']['noise'])

plt.savefig('R:/DA_and_Reward/es334/PPP1/output/' + x.rat + 'all_LEFT_trials.eps', format='eps', dpi=1000)