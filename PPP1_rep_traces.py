# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 13:42:25 2018

@author: jaimeHP
"""

s = 's10'
rat = 'PPP1.7'
x = rats[rat].sessions[s]
total_trials = len(x.malt['snips_licks_forced']['blue'])

trials_to_plot = np.arange(total_trials)
#trials_to_plot = [6, 20 ]

for x in trials_to_plot:
    f,ax = plt.subplots()
    data = (rat, x)
    repFig(ax, data, sub='malt')
    plt.title('Trial {}'.format(x))
    

#s10 - casein
#PPP1.1
    # trials [0]
#PPP1.2
    #none
#PPP1.3
    # trials [0,10, 16, 18, 20, 22]
#PPP1.4
    # trials [6, 20 ]
    
#PPP1.5
    #trials [3, 21]
#PPP1.6
    #none
#PPP1.7
    #trials [ 0,6,10, 16 ]
    
#s10 - maltodextrin
#PPP1.4
    #trials [0, 4, 7, 11, 13, 15]
#PPP1.7
    #trials [1, 7, 11, 14, 17, 18, 19]
    
    