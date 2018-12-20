# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:11:25 2018

@author: James Rig
"""

from ppp_pub_figs_settings import *

def sacc_behav_fig(df):
    f, ax = plt.subplots(figsize=(7.2, 2.5), ncols=2)
    
    scattersize = 50
    
    x = [[df.xs('NR', level=1)['latx1'], df.xs('NR', level=1)['latx2'], df.xs('NR', level=1)['latx3']],
     [df.xs('PR', level=1)['latx1'], df.xs('PR', level=1)['latx2'], df.xs('PR', level=1)['latx3']]]
    
    jmfig.barscatter(x, paired=True, unequal=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [col['np_cas'], col['np_malt'], col['lp_cas'], col['lp_malt']],
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel=['NR', 'PR'],
                 barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                 scattersize = scattersize,
                 ylim=[-1,20],
                 ax=ax[0])

    x = [[df.xs('NR', level=1)['missed1'], df.xs('NR', level=1)['missed2'], df.xs('NR', level=1)['missed3']],
     [df.xs('PR', level=1)['missed1'], df.xs('PR', level=1)['missed2'], df.xs('PR', level=1)['missed3']]]

    jmfig.barscatter(x, paired=True, unequal=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [col['np_cas'], col['np_malt'], col['lp_cas'], col['lp_malt']],
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel=['NR', 'PR'],
                 barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                 scattersize = scattersize,
                 ylim=[-5,50],
                 ax=ax[1])

    return f

def cond_licks_fig(ax, df, diet):
    
    scattersize = 50
    
    if diet == 'NR':
        cols = [col['np_cas'], col['np_cas'], col['np_malt'], col['np_malt']]
        title = 'Non-restricted'
    else:
        cols = [col['lp_cas'], col['lp_cas'], col['lp_malt'], col['lp_malt']]
        title = 'Protein restricted'

    x = [[df.xs(diet, level=1)['cond1-cas1-licks'], df.xs(diet, level=1)['cond1-cas2-licks']],
     [df.xs(diet, level=1)['cond1-malt1-licks'], df.xs(diet, level=1)['cond1-malt2-licks']]]

    jmfig.barscatter(x, paired=True, unequal=True,
             barfacecoloroption = 'individual',
             barfacecolor = cols,
             scatteredgecolor = ['xkcd:charcoal'],
             scatterlinecolor = 'xkcd:charcoal',
             grouplabel=['Cas', 'Malt'],
             barlabels=['1', '2', '1', '2'],
             scattersize = scattersize,
             
#             ylim=[-5,50],
             ax=ax)
    ax.set_title(title)

def cond_photo_fig(ax, df, diet, keys, event='',
                 color=[almost_black, 'xkcd:bluish grey'],
                 errorcolors=['xkcd:silver', 'xkcd:silver'],
                 yerror=True):

    if diet == 'NR':
        color=[almost_black, 'xkcd:bluish grey']
        errorcolors=['xkcd:silver', 'xkcd:silver']
        title = 'Non-restricted'
    else:
        color=[green, light_green]
        errorcolors=['xkcd:silver', 'xkcd:silver']
        title = 'Protein restricted'
    
    df = df.xs(diet, level=1)
    
    # Plots casein and maltodextrin shaded erros
    jmfig.shadedError(ax, df[keys[0]], linecolor=color[0], errorcolor=errorcolors[0])
    jmfig.shadedError(ax, df[keys[1]], linecolor=color[1], errorcolor=errorcolors[1])
    
    
    
    #ax.legend(['Casein', 'Maltodextrin'], fancybox=True)    
    ax.axis('off')

# Marks location of event on graph with arrow    
    arrow_y = ax.get_ylim()[1]
    ax.plot([100, 150], [arrow_y, arrow_y], color='xkcd:silver', linewidth=3)
    ax.annotate(event, xy=(125, arrow_y), xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom')

# Adds y scale bar
    if yerror:
        y = [y for y in ax.get_yticks() if y>0][:2]
        l = y[1] - y[0]
        scale_label = '{0:.0f}% \u0394F'.format(l*100)
        ax.plot([50,50], [y[0], y[1]], c=almost_black)
        ax.text(40, y[0]+(l/2), scale_label, va='center', ha='right')

# Adds x scale bar   
    y = ax.get_ylim()[0]
    ax.plot([251,300], [y, y], c=almost_black, linewidth=2)
    ax.annotate('5 s', xy=(276,y), xycoords='data',
                xytext=(0,-5), textcoords='offset points',
                ha='center',va='top')
    
def cond_photobar_fig(ax, df, diet, keys):
    
    scattersize = 50
    
    if diet == 'NR':
        cols = [col['np_cas'], col['np_cas'], col['np_malt'], col['np_malt']]
        title = 'Non-restricted'
    else:
        cols = [col['lp_cas'], col['lp_cas'], col['lp_malt'], col['lp_malt']]
        title = 'Protein restricted'
    
    x = [[df.xs(diet, level=1)[keys[0][0]], df.xs(diet, level=1)[keys[0][1]]],
          [df.xs(diet, level=1)[keys[1][0]], df.xs(diet, level=1)[keys[1][1]]]]
    
    jmfig.barscatter(x, paired=True, unequal=True,
         barfacecoloroption = 'individual',
         barfacecolor = cols,
         scatteredgecolor = ['xkcd:charcoal'],
         scatterlinecolor = 'xkcd:charcoal',
         grouplabel=['Cas', 'Malt'],
         barlabels=['1', '2', '1', '2'],
         scattersize = scattersize,
         
#             ylim=[-5,50],
         ax=ax)

