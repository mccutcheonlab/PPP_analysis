# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:00:38 2018

@author: jaimeHP
"""
import JM_general_functions as jmf
import dill

def assemble_sessions(sessions,
                      rats_to_include=[]
                      rats_to_exclude=[],
                      sessions_to_include=[],
                      outputfile="",
                      savefile=False,
                      makefigs=False)

    if len(rats_to_include) > 0:
        print('Overriding values in rats_to_exclude because of entry in rats_to_include.')
        rats_to_exclude = list(rats)
        for rat in rats_to_include:
            rats_to_exclude.remove(rat)
    
    for session in sessions:
          
        x = sessions[session]
        
        if x.rat not in rats_to_exclude and x.session in sessions_to_include: 
    
            try:
                x.loadmatfile()
        
                print('\nAnalysing rat ' + x.rat + ' in session ' + x.session)
                
                # Load in data from .mat file (convert from Tank first using Matlab script)
                x.loadmatfile()
                # Work out time to samples
                x.setticks()
                x.time2samples()       
                # Find out which bottles have TTLs/Licks associated with them     
                x.check4events()
    
                x.setbottlecolors()
                try:
                    x.left['lickdata'] = jmf.lickCalc(x.left['licks'],
                                      offset = x.left['licks_off'],
                                      burstThreshold = 0.50)
                except IndexError:
                    x.left['lickdata'] = 'none'
                    
                try:
                    x.right['lickdata'] = jmf.lickCalc(x.right['licks'],
                              offset = x.right['licks_off'],
                              burstThreshold = 0.50)
                except IndexError:
                    x.right['lickdata'] = 'none'
                
                bins = 300
                
                x.randomevents = jmf.makerandomevents(120, max(x.tick)-120)
                x.bgTrials, x.pps = jmf.snipper(x.data, x.randomevents,
                                                t2sMap = x.t2sMap, fs = x.fs, bins=bins)
                
                for side in [x.left, x.right]:   
                    if side['exist'] == True:
                        side['snips_sipper'] = jmf.mastersnipper(x, side['sipper'])
                        side['snips_licks'] = jmf.mastersnipper(x, side['lickdata']['rStart'])
                        try:
                            side['snips_licks_forced'] = jmf.mastersnipper(x, [licks for licks in side['lickdata']['rStart'] if licks < x.both['sipper'][0]])
                        except KeyError:
                            pass
                        
                        side['lats'] = jmf.latencyCalc(side['lickdata']['licks'], side['sipper'], cueoff=side['sipper_off'], lag=0)
                
                x.side2subs()
                 
                if makefigs == True:
                    pdf_pages = PdfPages(x.outputfolder + session + '.pdf')
                    makeBehavFigs(x)
                    makePhotoFigs(x)
                    pdf_pages.close()
                    plt.close('all')
            except:
                print('Could not extract data from ' + x.sessionID) 
            
    try:
        pdf_pages.close()
        plt.close('all')
    except:
        print('Everything already closed')
    
    if savefile == True:
        pickle_out = open(savefile)
        dill.dump([sessions, rats], pickle_out)
        pickle_out.close()


assemble_sessions(sessions,
                  rats_to_include = [],
                  rats_to_exclude = ['PPP1-8', 'PPP3-1', 'PPP3-6', 'PPP3-7', 'PPP3-2', 'PPP3-8'],
                  sessions_to_include = ['s6', 's7', 's8', 's9'],
                  outputfile='R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_rats.pickle',
                  savefile=True,
                  makefigs=True)


