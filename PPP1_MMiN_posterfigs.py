# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:16:41 2018

@author: jaimeHP
"""
import matplotlib.gridspec as gridspec

import timeit
tic = timeit.default_timer()

def toc():
    tc = timeit.default_timer()
    print(tc-tic)

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

def singletrialFig(ax, blue, uv, licks, color='k', xscale=True):
 
    ax.plot(uv, c=color, alpha=0.3)    
    ax.plot(blue, c=color)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    #Makes lick scatters
    xvals = [(x+10)*10 for x in licks]
    yvals = [ax.get_ylim()[1]]*len(licks)
    ax.plot(xvals,yvals,marker='|')        
    if xscale == True:
        y = ax.get_ylim()[0]
        ax.plot([251,300], [y, y], c='k', linewidth=2)
#        ax.text(276, y-0.01, '5 s', ha='center',va='top')
        ax.annotate('5 s', xy=(276,y), xycoords='data',
                    xytext=(0,-5), textcoords='offset points',
                    ha='center',va='top')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
#    xevent = pps * preTrial  
#    ax.plot([xevent, xevent],[ax.get_ylim()[0], ax.get_ylim()[1] - yrange/20],'--')
#    ax.text(xevent, ax.get_ylim()[1], eventText, ha='center',va='bottom')
    
    return ax

def averagetrace(ax, diet, color=['black', 'xkcd:bluish grey']):
    dietmsk = df.diet == diet

    shadedError(ax, df['cas1_licks_forced'][dietmsk], linecolor=color[0])
    ax = shadedError(ax, df['malt1_licks_forced'][dietmsk], linecolor=color[1])
    
    ax.axis('off')
    
#    ax.plot([50,50], [0.02, 0.04], c='k')
#    ax.text(45, 0.03, '2% \u0394F', verticalalignment='center', horizontalalignment='right')

    y = [y for y in ax.get_yticks() if y>0][:2]
    l = y[1] - y[0]
    scale_label = '{0:.0f}% \u0394F'.format(l*100)
    ax.plot([50,50], [y[0], y[1]], c='k')
    ax.text(45, y[0]+(l/2), scale_label, va='center', ha='right')
   
    y = ax.get_ylim()[0]
    ax.plot([251,300], [y, y], c='k', linewidth=2)
    ax.annotate('5 s', xy=(276,y), xycoords='data',
                xytext=(0,-5), textcoords='offset points',
                ha='center',va='top')
    
def repFig(ax, data, sub, color='k', yscale=True):
    x = rats[data[0]].sessions[s]
    n = data[1]
    
    if sub == 'cas':
        trial = x.cas[event]    
        run = x.cas['lickdata']['rStart'][n]
        all_licks = x.cas['licks']
    else:
        trial = x.malt[event]    
        run = x.malt['lickdata']['rStart'][n]
        all_licks = x.malt['licks']
 
    licks = [l-run for l in all_licks if (l>run-10) and (l<run+20)]
    singletrialFig(ax, trial['blue'][n], trial['uv'][n], licks, color=color)
    
    if yscale == True:
        y = [y for y in ax.get_yticks() if y>0][:2]
        l = y[1] - y[0]
        scale_label = '{0:.0f}% \u0394F'.format(l*100)
        ax.plot([50,50], [y[0], y[1]], c='k')
        ax.text(45, y[0]+(l/2), scale_label, verticalalignment='center', horizontalalignment='right')

def peakbargraph(ax, diet, keys):
    dietmsk = df.diet == diet
    a = [df[keys[0]][dietmsk], df[keys[1]][dietmsk]]

    x = data2obj1D(a)

    if diet == 'PR':
        cols = ['xkcd:kelly green', 'xkcd:light green']
    else:
        cols = ['xkcd:silver', 'w']
    
    ax, x, _, _ = jmfig.barscatter(x, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [cols[0], cols[1]],
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel=[],
                 scattersize = 100,
                 ax=ax)
#    ax.set_ylim([-.02, 0.15])
#    ax.set_yticks([0, 0.05, 0.1, 0.15])
    ax.set_ylabel('\u0394F')
       
def mainFig(rep_nr_cas, rep_nr_malt, rep_pr_cas, rep_pr_malt):
    
    gs = gridspec.GridSpec(2, 4, width_ratios=[1,1,1,0.5])
    f = plt.figure(figsize=(inch(520), inch(120)))
     #rep trace NR casein
    ax1 = f.add_subplot(gs[0,0])
#    ax1 = f.subplot
    repFig(ax1, rep_nr_cas, sub='cas')

    # rep trace NR maltodextrin
    ax2 = f.add_subplot(gs[0,1], sharey=ax1)
    repFig(ax2, rep_nr_malt, sub='malt', yscale=False)
    # average traces NR cas v malt
    ax3 = f.add_subplot(gs[0,2])
    averagetrace(ax3, 'NR')
    
    # rep trace NR casein
    ax4 = f.add_subplot(gs[1,0])    
    repFig(ax4, rep_pr_cas, sub='cas', color='xkcd:kelly green')
    # rep trace NR maltodextrin
    ax5 = f.add_subplot(gs[1,1], sharey=ax4)
    repFig(ax5, rep_pr_malt, sub='malt', color='xkcd:kelly green', yscale=False)
    # average traces NR cas v malt
    ax6 = f.add_subplot(gs[1,2])
    averagetrace(ax6, 'PR', color=['xkcd:kelly green', 'xkcd:light green'])
    
    #bar graphs
    ax7 = f.add_subplot(gs[0,3])
    keys = ['cas1_licks_peak', 'malt1_licks_peak']
    peakbargraph(ax7, 'NR', keys)
    plt.yticks([0,0.05, 0.1], ['0%', '5%', '10%'])
    
    ax8 = f.add_subplot(gs[1,3])
    peakbargraph(ax8, 'PR', keys)
    ax8.set_ylim([-0.03, 0.12])
    plt.yticks([0,0.05, 0.1], ['0%', '5%', '10%'])
    f.show()

# Data, choices for preference session 1 ['s10']
s = 's10'
rep_nr_cas = ('PPP1.7', 4)
rep_nr_malt = ('PPP1.7', 5)
rep_pr_cas = ('PPP1.4', 6)
rep_pr_malt = ('PPP1.4', 7)

event = 'snips_licks_forced'

mainFig(rep_nr_cas, rep_nr_malt, rep_pr_cas, rep_pr_malt)
#plt.savefig('R:/DA_and_Reward/es334/PPP1/figures/MMiN/pref1.eps')



