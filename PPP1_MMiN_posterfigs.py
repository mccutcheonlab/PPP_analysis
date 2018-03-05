# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:16:41 2018

@author: jaimeHP
"""
def inch(mm):
    result = mm*0.0393701
    return result

def doublesnipFig(ax1, ax2, df, diet, factor1, factor2):
    dietmsk = df.diet == diet    
    ax1.axis('off')
    ax2.axis('off')

    shadedError(ax1, df[factor1][dietmsk], linecolor='black')
    ax1 = shadedError(ax1, df[factor2][dietmsk], linecolor='xkcd:bluish grey')
    ax1.plot([50,50], [0.02, 0.04], c='k')
    ax1.text(45, 0.03, '2% \u0394F', verticalalignment='center', horizontalalignment='right')
    
    shadedError(ax2, df[factor1][~dietmsk], linecolor='xkcd:kelly green')
    ax2 = shadedError(ax2, df[factor2][~dietmsk], linecolor='xkcd:light green')
    ax2.plot([250,300], [-0.03, -0.03], c='k')
    ax2.text(275, -0.035, '5 s', verticalalignment='top', horizontalalignment='center')

def singletrialFig(ax, blue, uv, licks, color='k'):
 
    ax.plot(uv, c=color, alpha=0.4)    
    ax.plot(blue, c=color)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    #Makes lick scatters
    xvals = [(x+10)*10 for x in licks]
    yvals = [ax.get_ylim()[1]]*len(licks)
    ax.plot(xvals,yvals,marker='|')        
#    scalebar = scale * pps

#    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
#    scalebary = (yrange / 10) + ax.get_ylim()[0]
#    scalebarx = [ax.get_xlim()[1] - scalebar, ax.get_xlim()[1]]
#    
#    ax.plot(scalebarx, [scalebary, scalebary], c='k', linewidth=2)
#    ax.text((scalebarx[0] + (scalebar/2)), scalebary-(yrange/50), str(scale) +' s', ha='center',va='top')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
#    xevent = pps * preTrial  
#    ax.plot([xevent, xevent],[ax.get_ylim()[0], ax.get_ylim()[1] - yrange/20],'--')
#    ax.text(xevent, ax.get_ylim()[1], eventText, ha='center',va='bottom')
    
    return ax

def mainFig(rep_nr_cas, rep_nr_malt, rep_pr_cas, rep_pr_malt):

    f = plt.figure(figsize=(inch(520), inch(120)))
    ax = f.add_subplot(151)
    
    # rep trace NR casein
    x = rats[rep_nr_cas[0]].sessions[s]    
    trial = x.cas['snips_licks_forced']
    n = rep_nr_cas[1]
    run = x.cas['lickdata']['rStart'][n]
    licks = [l-run for l in x.cas['licks'] if (l>run-10) and (l<run+20)]
    print(licks)
    singletrialFig(ax, trial['blue'][n], trial['uv'][n], licks)
    f.show()
    
    print(srt)

# Data, choices for preference session 1 ['s10']
s = 's10'
rep_nr_cas = ('PPP1.4', 4)
rep_nr_malt = ('PPP1.4', 5)
rep_pr_cas = ('PPP1.4', 6)
rep_pr_malt = ('PPP1.4', 7)

mainFig(rep_nr_cas, rep_nr_malt, rep_pr_cas, rep_pr_malt)



