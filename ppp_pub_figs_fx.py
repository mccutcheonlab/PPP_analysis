# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:16:41 2018

NEED TO RUN ppp1_grouped.py first to load data and certain functions into memory.
Trying to do this using import statement - but at the moment not importing modules.

@author: jaimeHP
"""
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import JM_custom_figs as jmfig
import JM_general_functions as jmf

import matplotlib as mpl
import matplotlib.pyplot as plt

import timeit
tic = timeit.default_timer()

#Colors
green = mpl.colors.to_rgb('xkcd:kelly green')
light_green = mpl.colors.to_rgb('xkcd:light green')
almost_black = mpl.colors.to_rgb('#262626')

## Colour scheme
col={}
col['np_cas'] = 'xkcd:silver'
col['np_malt'] = 'white'
col['lp_cas'] = 'xkcd:kelly green'
col['lp_malt'] = 'xkcd:light green'

def toc():
    tc = timeit.default_timer()
    print(tc-tic)

def inch(mm):
    result = mm*0.0393701
    return result

def singletrialFig(ax, blue, uv, licks=[], color=almost_black, xscale=True, plot_licks=True):
    
    # Plots data
    ax.plot(uv, c=color, alpha=0.3)    
    ax.plot(blue, c=color)
   
    #Makes lick scatters
    if plot_licks == True:
        xvals = [(x+10)*10 for x in licks]
        yvals = [ax.get_ylim()[1]]*len(licks)
        ax.plot(xvals,yvals,linestyle='None',marker='|',markersize=5)        
    
    # Adds x scale bar
    if xscale == True:
        y = ax.get_ylim()[0]
        ax.plot([251,300], [y, y], c=almost_black, linewidth=2)
        ax.annotate('5 s', xy=(276,y), xycoords='data',
                    xytext=(0,-5), textcoords='offset points',
                    ha='center',va='top')
    
    # Removes axes and spines
    jmfig.invisible_axes(ax)
    
    return ax

def averagetrace(ax, diet, keys, color=[almost_black, 'xkcd:bluish grey'],
                 errorcolors=['xkcd:silver', 'xkcd:silver']):
    dietmsk = df4.diet == diet
#    keys = ['cas1_licks_forced', 'malt1_licks_forced']
    shadedError(ax, df4[keys[0]][dietmsk], linecolor=color[0], errorcolor=errorcolors[0])
    ax = shadedError(ax, df4[keys[1]][dietmsk], linecolor=color[1], errorcolor=errorcolors[1])
    
    ax.legend(['Casein', 'Maltodextrin'], fancybox=True)    
    ax.axis('off')
    
    arrow_y = ax.get_ylim()[1]
    ax.plot([100], [arrow_y], 'v', color='xkcd:silver')
    ax.annotate('First lick', xy=(100, arrow_y), xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom')

    y = [y for y in ax.get_yticks() if y>0][:2]
    l = y[1] - y[0]
    scale_label = '{0:.0f}% \u0394F'.format(l*100)
    ax.plot([50,50], [y[0], y[1]], c=almost_black)
    ax.text(45, y[0]+(l/2), scale_label, va='center', ha='right')
   
    y = ax.get_ylim()[0]
    ax.plot([251,300], [y, y], c=almost_black, linewidth=2)
    ax.annotate('5 s', xy=(276,y), xycoords='data',
                xytext=(0,-5), textcoords='offset points',
                ha='center',va='top')

def averagetracesx4(fig, keys, color=[almost_black, 'xkcd:bluish grey'],
                 errorcolor=['xkcd:silver', 'xkcd:silver']):
    
    fig.subplots_adjust(wspace=0.01, hspace=0.2, top=0.95)
    dietmsk = df4.diet == 'NR'
    
    ax1 = fig.add_subplot(221)
    shadedError(ax1, df4[keys[0]][dietmsk], linecolor=color[0][0], errorcolor=errorcolors[0][0])
    
    ax2 = fig.add_subplot(222, sharey=ax1)
    shadedError(ax2, df4[keys[1]][dietmsk], linecolor=color[0][1], errorcolor=errorcolors[0][1])

    dietmsk = df4.diet == 'PR'
    
    ax3 = fig.add_subplot(223)
    shadedError(ax3, df4[keys[0]][dietmsk], linecolor=color[1][0], errorcolor=errorcolors[1][0])
    
    ax4 = fig.add_subplot(224, sharey=ax3)
    shadedError(ax4, df4[keys[1]][dietmsk], linecolor=color[1][1], errorcolor=errorcolors[1][1])

    NRcas_line = mlines.Line2D([], [], color=color[0][0], label='Casein')
    NRmalt_line = mlines.Line2D([], [], color=color[0][1], label='Maltodextrin')
    
    PRcas_line = mlines.Line2D([], [], color=color[1][0], label='Casein')
    PRmalt_line = mlines.Line2D([], [], color=color[1][1], label='Maltodextrin')
    
    for ax, title in zip([ax1, ax2], ['Casein', 'Maltodextrin']):
        ax.title.set_position([0.5, 1.1])
        ax.set_title(title)
    
    for ax, lines in zip([ax2, ax4], [[NRcas_line, NRmalt_line], [PRcas_line, PRmalt_line]]):
        ax.legend(handles=lines, fancybox=True)
        
    for ax in [ax1, ax2, ax3, ax4]:
        print(ax.legend())
     
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axis('off')
        
        arrow_y = ax.get_ylim()[1]
        ax.plot([100], [arrow_y], 'v', color='xkcd:silver')
        ax.annotate('First lick', xy=(100, arrow_y), xytext=(0,5), textcoords='offset points',
                    ha='center', va='bottom')
    
    for ax in [ax1, ax3]:
        y = [y for y in ax.get_yticks() if y>0][:2]
        l = y[1] - y[0]
        scale_label = '{0:.0f}% \u0394F'.format(l*100)
        ax.plot([50,50], [y[0], y[1]], c=almost_black)
        ax.text(45, y[0]+(l/2), scale_label, va='center', ha='right')
        
    for ax in [ax2, ax4]:
        y = ax.get_ylim()[0]
        ax.plot([251,300], [y, y], c=almost_black, linewidth=2)
        ax.annotate('5 s', xy=(276,y), xycoords='data',
                    xytext=(0,-5), textcoords='offset points',
                    ha='center',va='top')
        
  

def repFig(ax, data, sub, color=almost_black, yscale=True, legend=False):
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
    singletrialFig(ax, trial['blue'][n], trial['uv'][n],
                   licks=licks, color=color, plot_licks=False)
    
    if yscale == True:
        l = 0.05
        y1 = [y for y in ax.get_yticks() if y>0][0]
        y2 = y1 + l        
        scale_label = '{0:.0f}% \u0394F'.format(l*100)
        ax.plot([50,50], [y1, y2], c=almost_black)
        ax.text(45, y1 + (l/2), scale_label, va='center', ha='right')

    if legend == True:
        ax.annotate('470 nm', xy=(310,trial['blue'][n][299]), color=color, va='center')
        ax.annotate('405 nm', xy=(310,trial['uv'][n][299]), color=color, alpha=0.3, va='center')
    
    return ax

def removenoise(snipdata):
    # returns blue snips with noisey ones removed
    new_snips = [snip for (snip, noise) in zip(snipdata['blue'], snipdata['noise']) if not noise]
    return new_snips

def reptracesFig(f, gs, gsx, gsy, casdata, maltdata, color=almost_black, title=False):
    
    inner = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs[gsx,gsy],
                                             wspace=0.05, hspace=0.00,
                                             height_ratios=[1,8])    
    ax1 = f.add_subplot(inner[1,0])
    repFig(ax1, casdata, sub='cas', color=color)
    ax2 = f.add_subplot(inner[1,1], sharey=ax1)
    repFig(ax2, maltdata, sub='malt', color=color, yscale=False, legend=True)

    ax3 = f.add_subplot(inner[0,0], sharex=ax1)
    lickplot(ax3, casdata, sub='cas')
    ax4 = f.add_subplot(inner[0,1], sharey=ax3, sharex=ax2)
    lickplot(ax4, maltdata, sub='malt', ylabel=False)
    
    if title == True:
        ax3.set_title('Casein')
        ax4.set_title('Maltodextrin')

def lickplot(ax, data, sub='malt', ylabel=True, style='raster'):        
    # Removes axes and spines
    jmfig.invisible_axes(ax)

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
    licks_x = [(x+10)*10 for x in licks]
    if style == 'histo':
        hist, bins = np.histogram(licks_x, bins=30, range=(0,300))
        center = (bins[:-1] + bins[1:]) / 2
        width = 1 * (bins[1] - bins[0])   
        ax.bar(center, hist, align='center', width=width, color='xkcd:silver')
    
    if style == 'raster':
        yvals = [1]*len(licks)
        ax.plot(licks_x,yvals,linestyle='None',marker='|',markersize=5, color='xkcd:silver')
        
    else:
        print('Not a valid style for plotting licks')

    if ylabel == True:
        ax.annotate('Licks', xy=(95,1), va='center', ha='right')

def freechoicegraph(ax, df, diet, keys, bar_colors=['xkcd:silver', 'w'], sc_color='w'):

    dietmsk = df.diet == diet
    a = [df[keys[0]][dietmsk], df[keys[1]][dietmsk]]
    
    jmfig.barscatter(a, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = bar_colors,
                 scatteredgecolor = [almost_black],
                 scatterlinecolor = almost_black,
                 scatterfacecolor = [sc_color],
                 grouplabel=['Cas', 'Malt'],
                 scattersize = 80,
                 ax=ax)

    ax.set_ylabel('Free choices')
    ax.set_ylim([-2, 22])
#    plt.yticks([0,0.05, 0.1], ['0%', '5%', '10%'])

def averagetracesx2(f, gs, gsx, gsy, df, keys, diet,
                    color=[almost_black, 'xkcd:bluish grey'],
                    errorcolor=['xkcd:silver', 'xkcd:silver'],
                    title=False):
    
#    fig.subplots_adjust(wspace=0.01, hspace=0.2, top=0.95)
    dietmsk = df.diet == diet
    inner = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[gsx,gsy],
                                             wspace=0.15)
    
    ax1 = f.add_subplot(inner[0])
    jmfig.shadedError(ax1, df[keys[0]][dietmsk], linecolor=color[0], errorcolor=errorcolor[0])
    
    ax2 = f.add_subplot(inner[1], sharey=ax1)
    jmfig.shadedError(ax2, df[keys[1]][dietmsk], linecolor=color[1], errorcolor=errorcolor[1])

#    
    if title == True:
        for ax, title in zip([ax1, ax2], ['Casein', 'Maltodextrin']):
            ax.title.set_position([0.5, 1.2])
            ax.set_title(title)
#    cas_line = mlines.Line2D([], [], color=color[0], label='Casein')
#    malt_line = mlines.Line2D([], [], color=color[1], label='Maltodextrin')  
#    ax2.legend(handles=[cas_line, malt_line], fancybox=True)

    for ax in [ax1, ax2]:
        ax.axis('off')
        
        arrow_y = ax.get_ylim()[1]
        ax.plot([100], [arrow_y], 'v', color='xkcd:silver')
        ax.annotate('First lick', xy=(100, arrow_y), xytext=(0,5), textcoords='offset points',
                    ha='center', va='bottom')
    
    for ax in [ax1]:
        y = [y for y in ax.get_yticks() if y>0][:2]
        l = y[1] - y[0]
        scale_label = '{0:.0f}% \u0394F'.format(l*100)
        ax.plot([10,10], [y[0], y[1]], c=almost_black)
        ax.text(0, y[0]+(l/2), scale_label, va='center', ha='right')
        
    for ax in [ax2]:
        y = ax.get_ylim()[0]
        ax.plot([251,300], [y, y], c=almost_black, linewidth=2)
        ax.annotate('5 s', xy=(276,y), xycoords='data',
                    xytext=(0,-5), textcoords='offset points',
                    ha='center',va='top')

def peakbargraph(ax, df, diet, keys, bar_colors=['xkcd:silver', 'w'], sc_color='w'):
    dietmsk = df.diet == diet
    a = [df[keys[0]][dietmsk], df[keys[1]][dietmsk]]
    x = jmf.data2obj1D(a)
    
    ax, x, _, _ = jmfig.barscatter(x, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = bar_colors,
                 scatteredgecolor = [almost_black],
                 scatterlinecolor = almost_black,
                 scatterfacecolor = [sc_color],
                 grouplabel=['Cas', 'Malt'],
                 scattersize = 80,
                 ax=ax)

    ax.set_ylabel('\u0394F')
#    ax.set_ylim([-0.04, 0.14])
    plt.yticks([0,0.05, 0.1], ['0%', '5%', '10%'])

def mainPhotoFig(df_choice, df_photo, keys_choicebars, keys_traces, keys_photobars, dietswitch=False):
    
    gs = gridspec.GridSpec(2, 3, width_ratios=[1,4,1], wspace=0.5)
    f = plt.figure(figsize=(7,4))
    f.subplots_adjust(wspace=0.01, hspace=0.6, top=0.85, left=0.1)
    
    rowcolors = [[almost_black, 'xkcd:bluish grey'], [green, light_green]]
    rowcolors_bar = [['xkcd:silver', 'w'], [green, light_green]]
    
    if dietswitch == True:
        rowcolors.reverse()
        rowcolors_bar.reverse()

# Non-restricted figures, row 0
    ax1 = f.add_subplot(gs[0,0])
    freechoicegraph(ax1, df_choice, 'NR', keys_choicebars, bar_colors=rowcolors_bar[0], sc_color='w')
    
    # average traces NR cas v malt
    averagetracesx2(f, gs, 0, 1, df_photo, keys_traces, 'NR', color=rowcolors[0], title=True)

    ax3 = f.add_subplot(gs[0,2]) 
    peakbargraph(ax3, df_photo, 'NR', keys_photobars, bar_colors=rowcolors_bar[0], sc_color='w')
    
# Protein-restricted figures, row 1
    ax4 = f.add_subplot(gs[1,0])
    freechoicegraph(ax4, df_choice, 'PR', keys_choicebars, bar_colors=rowcolors_bar[1], sc_color='w')
    
    # average traces NR cas v malt
    averagetracesx2(f, gs, 1, 1, df_photo, keys_traces, 'PR', color=rowcolors[1])

    ax6 = f.add_subplot(gs[1,2]) 
    peakbargraph(ax6, df_photo, 'PR', keys_photobars, bar_colors=rowcolors_bar[1], sc_color='w')

    return f

# To make summary figure

def choicefig(df, keys, ax):
    dietmsk = df.diet == 'NR'
    
    a = [[df[keys[0]][dietmsk], df[keys[1]][dietmsk], df[keys[2]][dietmsk]],
          [df[keys[0]][~dietmsk], df[keys[1]][~dietmsk], df[keys[2]][~dietmsk]]]
    x = jmf.data2obj2D(a)
    
    cols = ['xkcd:silver', green]
    
    jmfig.barscatter(x, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [cols[0], cols[1], cols[1], cols[1], cols[0], cols[0]],
                 scatteredgecolor = [almost_black],
                 scatterlinecolor = almost_black,
                 scattersize = 100,
                 ax=ax)
    
    ax.set_xticks([])
    yval = ax.get_ylim()[0] - (ax.get_ylim()[1]-ax.get_ylim()[0])/20
    xlabels = ['NR \u2192 PR', 'PR \u2192 NR']
    for x,label in enumerate(xlabels):
        ax.text(x+1, yval, label, ha='center')

def peakresponsebargraph(ax, df, keys, ylabels=True, dietswitch=False, xlabels=[]):
    dietmsk = df.diet == 'NR'
    
    a = [[df[keys[0]][dietmsk], df[keys[1]][dietmsk]],
          [df[keys[0]][~dietmsk], df[keys[1]][~dietmsk]]]

    x = data2obj2D(a)
    if dietswitch == True:
        cols = [green, light_green, 'xkcd:silver', 'w']
    else:        
        cols = ['xkcd:silver', 'w', green, light_green]
    
    jmfig.barscatter(x, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [cols[0], cols[1], cols[2], cols[3]],
                 scatteredgecolor = [almost_black],
                 scatterlinecolor = almost_black,
                 scattersize = 100,
                 ax=ax)
    ax.set_xticks([])
    
    for x,label in enumerate(xlabels):
        ax.text(x+1, -0.0175, label, ha='center')
    
    ax.set_ylim([-.02, 0.135])
    yticks = [0, 0.05, 0.1]
    ax.set_yticks(yticks)
    
    if ylabels == True:
        yticklabels = ['{0:.0f}%'.format(x*100) for x in yticks]
        ax.set_yticklabels(yticklabels)
        ax.set_ylabel('\u0394F', rotation=0)
    else:
        ax.set_yticklabels([])

def makesummaryFig():
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,3], wspace=0.3)
    mpl.rcParams['figure.subplot.left'] = 0.10
    mpl.rcParams['figure.subplot.top'] = 0.90
    mpl.rcParams['axes.labelpad'] = 4
    f = plt.figure(figsize=(inch(300), inch(120)))
    
    adjust = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[0],
                                             wspace=0.05,
                                             height_ratios=[18,1])
    
    ax0 = f.add_subplot(adjust[0])
    choicefig(df1, ['pref1', 'pref2', 'pref3'], ax0)
    ax0.set_ylabel('Casein preference')
    plt.yticks([0, 0.5, 1.0])
    ax_ = f.add_subplot(adjust[1])
    jmfig.invisible_axes(ax_)
    
    inner = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[1],
                                             wspace=0.15)
    ax1 = f.add_subplot(inner[0])
    ax2 = f.add_subplot(inner[1])
    ax3 = f.add_subplot(inner[2])
    
    peakresponsebargraph(ax1, df4, ['cas1_licks_peak', 'malt1_licks_peak'],
                         xlabels=['NR', 'PR'])
    peakresponsebargraph(ax2, df4, ['cas2_licks_peak', 'malt2_licks_peak'],
                         xlabels=['NR \u2192 PR', 'PR \u2192 NR'],
                         ylabels=False, dietswitch=True)
    peakresponsebargraph(ax3, df4, ['cas3_licks_peak', 'malt3_licks_peak'],
                         xlabels=['NR \u2192 PR', 'PR \u2192 NR'],
                         ylabels=False, dietswitch=True)
    
    titles = ['Preference test 1', 'Preference test 2', 'Preference test 3']
    for ax, title in zip([ax1, ax2, ax3], titles):
        ax.set_title(title)
    
    return f

def makesummaryFig2(df_pref, df_photo):
    gs = gridspec.GridSpec(1, 2, wspace=0.5)
    mpl.rcParams['figure.subplot.left'] = 0.10
    mpl.rcParams['figure.subplot.top'] = 0.85
    mpl.rcParams['axes.labelpad'] = 4
    f = plt.figure(figsize=(5,2))
    
    ax0 = f.add_subplot(gs[0])
    choicefig(df_pref, ['pref1', 'pref2', 'pref3'], ax0)
    ax0.set_ylabel('Casein preference')
    ax0.set_yticks([0, 0.5, 1.0]) 
    ax0.set_yticklabels(['0', '0.5', '1'])
    ax0.set_title('Behaviour')
    ax1 = f.add_subplot(gs[1])
    choicefig(df_photo, ['pref1_peak_delta', 'pref2_peak_delta', 'pref3_peak_delta'], ax1)
    ax1.set_ylabel('\u0394F (Casein - Malt.)')
    
    ax1.set_ylim([-0.035, 0.09])
    ax1.set_yticks([-0.02, 0, 0.02, 0.04, 0.06, 0.08])
    ax1.set_yticklabels([-0.02, 0, 0.02, 0.04, 0.06, 0.08])
    ax1.set_title('Photometry')

    return f

def forcedandfreelicks(ax, df_forced, df_free, prefsession=1, dietswitch=False):

    forced_cas_key = 'forced' + str(prefsession) + '-cas'
    forced_malt_key = 'forced' + str(prefsession) + '-malt'
    free_cas_key = 'free' + str(prefsession) + '-cas'
    free_malt_key = 'free' + str(prefsession) + '-malt'
    
    dietmsk = df_forced.diet == 'NR'
    x = [[df_forced[forced_cas_key][dietmsk], df_forced[forced_malt_key][dietmsk]],
         [df_forced[forced_cas_key][~dietmsk], df_forced[forced_malt_key][~dietmsk]]]
    jmfig.barscatter(x, paired=True, unequal=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [col['np_cas'], col['np_malt'], col['lp_cas'], col['lp_malt']],
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel=['NR', 'PR'],
                 barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                 grouplabeloffset=0.1,
                 barlabeloffset=0.025,
                 scattersize = 80,
                 ax=ax[0])

# Fig for free choice licks
    dietmsk = df_free.diet == 'NR'
    x = [[df_free[free_cas_key][dietmsk], df_free[free_malt_key][dietmsk]],
         [df_free[free_cas_key][~dietmsk], df_free[free_malt_key][~dietmsk]]]
    jmfig.barscatter(x, paired=True, unequal=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [col['np_cas'], col['np_malt'], col['lp_cas'], col['lp_malt']],
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel=['NR', 'PR'],
                 barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                 grouplabeloffset=0.1,
                 barlabeloffset=0.025,
                 scattersize = 80,
                 ax=ax[1])

    ax[0].set_ylabel('Licks')
    ax[0].set_ylim([-50, 1050])
    ax[0].set_yticks([0, 500, 1000])
    ax[0].set_xticks([])



def behav_vs_photoFig(ax, xdata, ydata, diet):
    for x, y, d in zip(xdata, ydata, diet):
        if d == 'NR':
            color = 'k'
        else:
            color = 'g'
        ax.scatter(x, y, c=color)
