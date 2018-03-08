# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:16:41 2018

@author: jaimeHP
"""
import matplotlib.gridspec as gridspec
import JM_custom_figs as jmfig


import timeit
tic = timeit.default_timer()

#Colors
green = mpl.colors.to_rgb('xkcd:kelly green')
almost_black = mpl.colors.to_rgb('#262626')
light_green = mpl.colors.to_rgb('xkcd:light green')

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
        ax.plot([251,300], [y, y], c='k', linewidth=2)
        ax.annotate('5 s', xy=(276,y), xycoords='data',
                    xytext=(0,-5), textcoords='offset points',
                    ha='center',va='top')
    
    # Removes axes and spines
    jmfig.invisible_axes(ax)
    
    return ax

def averagetrace(ax, diet, keys, color=[almost_black, 'xkcd:bluish grey']):
    dietmsk = df4.diet == diet
#    keys = ['cas1_licks_forced', 'malt1_licks_forced']
    shadedError(ax, df4[keys[0]][dietmsk], linecolor=color[0])
    ax = shadedError(ax, df4[keys[1]][dietmsk], linecolor=color[1])
    
    ax.axis('off')

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
        y = [y for y in ax.get_yticks() if y>0][:2]
        l = y[1] - y[0]
        scale_label = '{0:.0f}% \u0394F'.format(l*100)
        ax.plot([50,50], [y[0], y[1]], c='k')
        ax.text(45, y[0]+(l/2), scale_label, verticalalignment='center', horizontalalignment='right')

    if legend == True:
        ax.annotate('470 nm', xy=(300,trial['blue'][n][299]), color=color, va='center')
        ax.annotate('405 nm', xy=(300,trial['uv'][n][299]), color=color, alpha=0.3, va='center')
    
    return ax

def peakbargraph(ax, diet, keys):
    dietmsk = df4.diet == diet
    a = [df4[keys[0]][dietmsk], df4[keys[1]][dietmsk]]
    x = data2obj1D(a)

    if diet == 'PR':
        cols = [green, light_green]
    else:
        cols = ['xkcd:silver', 'w']
    
    ax, x, _, _ = jmfig.barscatter(x, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [cols[0], cols[1]],
                 scatteredgecolor = [almost_black],
                 scatterlinecolor = almost_black,
                 grouplabel=[],
                 scattersize = 100,
                 ax=ax)

    ax.set_ylabel('\u0394F')
    ax.set_ylim([-0.04, 0.14])
    plt.yticks([0,0.05, 0.1], ['0%', '5%', '10%'])

def makeheatmap(ax, data, ylabel='Trials'):
    ntrials = np.shape(data)[0]
    xvals = np.linspace(-9.9,20,300)
    yvals = np.arange(1, ntrials+2)
    xx, yy = np.meshgrid(xvals, yvals)
    
    mesh = ax.pcolormesh(xx, yy, data, cmap='YlGnBu', shading = 'flat')
    ax.set_ylabel(ylabel)
    ax.set_yticks([1, ntrials])
    ax.set_xticks([])
    ax.invert_yaxis()
    
    return ax, mesh

def removenoise(snipdata):
    # returns blue snips with noisey ones removed
    new_snips = [snip for (snip, noise) in zip(snipdata['blue'], snipdata['noise']) if not noise]
    return new_snips

def heatmapFig(f, gs, gsx, gsy, session, rat, clims=[0,1]):
    x = rats[rat].sessions[s]
    data_cas = removenoise(x.cas['snips_licks_forced'])
    data_malt = removenoise(x.malt['snips_licks_forced'])

    inner = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs[gsx,gsy],
                                             width_ratios=[12,1],
                                             wspace=0.05)
    ax1 = f.add_subplot(inner[0,0])
    ax, mesh = makeheatmap(ax1, data_cas, ylabel='Casein')
    mesh.set_clim(clims)
    
    ax2 = f.add_subplot(inner[1,0], sharex=ax1)
    ax, mesh = makeheatmap(ax2, data_malt, ylabel='Malt')
    mesh.set_clim(clims)
   
    cbar_ax = f.add_subplot(inner[:,1])   
    cbar = f.colorbar(mesh, cax=cbar_ax, ticks=[clims[0], 0, clims[1]])
    cbar_labels = ['{0:.0f}%'.format(clims[0]*100),
                   '0% \u0394F',
                   '{0:.0f}%'.format(clims[1]*100)]
    cbar.ax.set_yticklabels(cbar_labels)

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

def mainFig(rep_nr_cas, rep_nr_malt, rep_pr_cas, rep_pr_malt):
    
    gs = gridspec.GridSpec(2, 4, width_ratios=[1.5,1,1,0.5], wspace=0.3)
    f = plt.figure(figsize=(inch(520), inch(120)))
    
    # Non-restricted figures, row 0
    reptracesFig(f, gs, 0, 0, rep_nr_cas, rep_nr_malt, title=True)
    heatmapFig(f, gs, 0, 1, 's10', 'PPP1.7', clims=clim_nr)
    # average traces NR cas v malt
    ax3 = f.add_subplot(gs[0,2])
    averagetrace(ax3, 'NR', keys_traces)
    
    ax7 = f.add_subplot(gs[0,3])
    
    peakbargraph(ax7, 'NR', keys_bars)
   
    # Protein-restricted figures, row 1
    reptracesFig(f, gs, 1, 0, rep_pr_cas, rep_pr_malt, color=green)    
    heatmapFig(f, gs, 1, 1, 's10', 'PPP1.3', clims=clim_pr)
    # average traces NR cas v malt
    ax6 = f.add_subplot(gs[1,2])
    averagetrace(ax6, 'PR', keys_traces, color=[green, light_green])

    ax8 = f.add_subplot(gs[1,3])
    peakbargraph(ax8, 'PR', keys_bars)
    
    return f

# Data, choices for preference session 1 ['s10']

s = 's10'
rep_nr_cas = ('PPP1.7', 16)
rep_nr_malt = ('PPP1.7', 19)
rep_pr_cas = ('PPP1.4', 6)
rep_pr_malt = ('PPP1.4', 4)

clim_nr = [-0.15,0.20]
clim_pr = [-0.11,0.17]

event = 'snips_licks_forced'
keys_traces = ['cas1_licks_forced', 'malt1_licks_forced']
keys_bars = ['cas1_licks_peak', 'malt1_licks_peak']

pref1Fig = mainFig(rep_nr_cas, rep_nr_malt, rep_pr_cas, rep_pr_malt)

# Data, choices for preference session 1 ['s11']
s = 's11'
rep_nr_cas = ('PPP1.7', 7)
rep_nr_malt = ('PPP1.7', 4) #19 OK
rep_pr_cas = ('PPP1.4', 20)
rep_pr_malt = ('PPP1.4', 15)

keys_traces = ['cas2_licks_forced', 'malt2_licks_forced']
keys_bars = ['cas2_licks_peak', 'malt2_licks_peak']

pref2Fig = mainFig(rep_nr_cas, rep_nr_malt, rep_pr_cas, rep_pr_malt)

# Data, choices for preference session 1 ['s16']
s = 's16'
rep_nr_cas = ('PPP1.7', 14)
rep_nr_malt = ('PPP1.7', 14)
rep_pr_cas = ('PPP1.4', 14)
rep_pr_malt = ('PPP1.4', 10)

keys_traces = ['cas3_licks_forced', 'malt3_licks_forced']
keys_bars = ['cas3_licks_peak', 'malt3_licks_peak']

pref3Fig = mainFig(rep_nr_cas, rep_nr_malt, rep_pr_cas, rep_pr_malt)

pref1Fig.savefig('R:/DA_and_Reward/es334/PPP1/figures/MMiN/pref1.pdf')
pref2Fig.savefig('R:/DA_and_Reward/es334/PPP1/figures/MMiN/pref2.pdf')
pref3Fig.savefig('R:/DA_and_Reward/es334/PPP1/figures/MMiN/pref3.pdf')


#pref1Fig.savefig('R:/DA_and_Reward/es334/PPP1/figures/MMiN/pref1.eps')
#pref2Fig.savefig('R:/DA_and_Reward/es334/PPP1/figures/MMiN/pref2.eps')
#pref3Fig.savefig('R:/DA_and_Reward/es334/PPP1/figures/MMiN/pref3.eps')

# To make summary figure

def choicefig(df, keys, ax):
    dietmsk = df.diet == 'NR'
    
    a = [[df[keys[0]][dietmsk], df[keys[1]][dietmsk], df[keys[2]][dietmsk]],
          [df[keys[0]][~dietmsk], df[keys[1]][~dietmsk], df[keys[2]][~dietmsk]]]
    x = data2obj2D(a)
    
    cols = ['xkcd:silver', green]
    
    jmfig.barscatter(x, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [cols[0], cols[1], cols[1], cols[1], cols[0], cols[0]],
                 scatteredgecolor = [almost_black],
                 scatterlinecolor = almost_black,
                 grouplabel=['NR \u2192 PR', 'PR \u2192 NR'],
                 scattersize = 100,
                 ax=ax)

def peakresponsebargraph(ax, df, keys, ylabels=True, dietswitch=False):
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
                 grouplabel=['NR \u2192 PR', 'PR \u2192 NR'],
                 scattersize = 100,
                 ax=ax)
#    ax.set_xticks([])
    
    if ylabels == True:
        ax.set_ylim([-.02, 0.15])
        yticks = [0, 0.05, 0.1, 0.15]
        ax.set_yticks(yticks)
        yticklabels = ['{0:.0f}%'.format(x*100) for x in yticks]
        ax.set_yticklabels(yticklabels)
        ax.set_ylabel('\u0394F')

def makesummaryFig():
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,3], wspace=0.3)
    f = plt.figure(figsize=(inch(520), inch(120)))
    
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
    ax2 = f.add_subplot(inner[1], sharey=ax1)
    ax3 = f.add_subplot(inner[2], sharey=ax1)
    
    peakresponsebargraph(ax1, df4, ['cas1_licks_peak', 'malt1_licks_peak'])
    peakresponsebargraph(ax2, df4, ['cas2_licks_peak', 'malt2_licks_peak'],
                         ylabels=False, dietswitch=True)
    peakresponsebargraph(ax3, df4, ['cas3_licks_peak', 'malt3_licks_peak'],
                         ylabels=False, dietswitch=True)
    
    titles = ['Preference test 1', 'Preference test 2', 'Preference test 3']
    for ax, title in zip([ax1, ax2, ax3], titles):
        ax.set_title(title)
        
    return f

summaryFig = makesummaryFig()
summaryFig.savefig('R:/DA_and_Reward/es334/PPP1/figures/MMiN/summary.pdf')
    
    
