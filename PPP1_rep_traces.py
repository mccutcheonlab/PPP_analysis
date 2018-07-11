# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 13:42:25 2018

@author: jaimeHP
"""

s = 's16'
rat = 'PPP1.4'
x = rats[rat].sessions[s]
total_trials = len(x.cas['snips_licks_forced']['blue'])

trials_to_plot = np.arange(total_trials-1)
trials_to_plot = [8, 10, 14, 15, 17]

for x in trials_to_plot:
    f,ax = plt.subplots()
    data = (rat, x)
    repFig(ax, data, sub='malt')
    plt.title('Trial {}'.format(x))
    

##Casein
#PPP1.1
    # trials [0]
#PPP1.2
    #none
#PPP1.3
    # trials [0,10, 16, 18, 20, 22]
#PPP1.4
    #S10 = [6, 20 ]
    #S11 = [5, 7, 10, 12, 16, 18, 20] *10
    #s16 = [0, 3, 6, 9, 11, 14, 16]
    
#PPP1.5
    #trials [3, 21]
#PPP1.6
    #none
#PPP1.7
    #s10 = [ 0,6,10, 16 ], s11=[4,5, 17, 19]
    #s11 = [4, 5, 7, 15, 17, 21]
    #s16 = [7, 14, 15] *14

##Maltodextrin
#PPP1.4
    #s10 = [0, 4, 7, 11, 13, 15]
    #s11 = [5, 15] *15
    #s16 = [8, 10, 14, 15, 17] 10 or 14
#PPP1.7
    #s10 = [1, 7, 11, 14, 17, 18, 19]
    #s11 = [4,5, 17, 19], [5] best
    #s16 = [0, 6, 14, 15, 17]
    
    