# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:14:26 2020

@author: admin
"""
# May need to run this first in ipython %matplotlib qt5

import dill

from fx4assembly import *

picklefolder = 'C:\\Github\\PPP_analysis\\data\\'

ppp1_sessions = metafile2sessions('D:\\DA_and_Reward\\es334\PPP1\\PPP1.xlsx',
                                  'D:\\DA_and_Reward\\es334\PPP1\\PPP1_metafile',
                                  'D:\\DA_and_Reward\\es334\\PPP1\\tdtfiles\\',
                                  'D:\\DA_and_Reward\\es334\\PPP1\\output\\',
                                  sheetname='metafile')

ppp3_sessions = metafile2sessions('D:\\DA_and_Reward\\gc214\\PPP3\\PPP3.xlsx',
                                  'D:\\DA_and_Reward\\gc214\\PPP3\\PPP3_metafile',
                                  'D:\\DA_and_Reward\\gc214\\PPP3\\tdtfiles\\',
                                  'D:\\DA_and_Reward\\gc214\\PPP3\\output\\',
                                  sheetname='PPP3_metafile')

ppp4_sessions = metafile2sessions('D:\\DA_and_Reward\\gc214\\PPP4\\PPP4.xlsx',
                                  'D:\\DA_and_Reward\\gc214\\PPP4\\PPP4_metafile',
                                  'D:\\DA_and_Reward\\gc214\\PPP4\\tdtfiles\\',
                                  'D:\\DA_and_Reward\\gc214\\PPP4\\output\\',
                                  sheetname='PPP4_metafile')

ppp_sessions = {**ppp1_sessions, **ppp3_sessions, **ppp4_sessions}

savefile=True
makefigs=False

assemble_pref = True
assemble_single = False
assemble_exclusion = False
assemble_cond1 = False # No photometry recordings for PPP4 during conditioning so cannot analyse this way using TDT timestamps
assemble_cond1_metafiledata = False

# assemble_pref = False
# assemble_single = True
# assemble_exclusion = False
# assemble_cond1_metafiledata = False

if assemble_pref:
    sessions = assemble_sessions(ppp_sessions,
                  rats_to_include = [],
                  rats_to_exclude = ['PPP1-8',
                                     'PPP3-1', 'PPP3-6', 'PPP3-7',
                                     'PPP4-2', 'PPP4-3', 'PPP4-5', 'PPP4-7', 'PPP4-8'],
                  sessions_to_include = ['s10', 's11', 's16'],
                  outputfile='C:\\Github\\PPP_analysis\\data\\ppp_pref.pickle',
                  savefile=savefile,
                  makefigs=makefigs)
    
if assemble_cond1:
    sessions = assemble_sessions(ppp_sessions,
                  rats_to_include = [],
                  rats_to_exclude = ['PPP1-8',
                                     'PPP3-1', 'PPP3-6', 'PPP3-7',
                                     'PPP4-2', 'PPP4-3', 'PPP4-5', 'PPP4-7', 'PPP4-8'],
                  sessions_to_include = ['s6', 's7', 's8', 's9'],
                  outputfile='C:\\Github\\PPP_analysis\\data\\ppp_cond.pickle',
                  savefile=savefile,
                  makefigs=makefigs)

if assemble_cond1_metafiledata:
    sessions = savejustmetafiledata(ppp_sessions,
                                    rats_to_exclude = ['PPP1-8',
                                     'PPP3-1', 'PPP3-6', 'PPP3-7',
                                     'PPP4-2', 'PPP4-3', 'PPP4-5', 'PPP4-7', 'PPP4-8'],
                                    sessions_to_include = ['s6', 's7', 's8', 's9'],
                                    outputfile='C:\\Github\\PPP_analysis\\data\\ppp_cond.pickle')
    
# Code to run for single rat
if assemble_single:
    sessions_to_add = assemble_sessions(ppp_sessions,
                  rats_to_include = ['PPP4-6'],
                  rats_to_exclude = ['PPP1-8', 'PPP3-1', 'PPP3-6', 'PPP3-7', 'PPP3-2', 'PPP3-8'],
                  sessions_to_include = ['s16'],
                  outputfile=picklefolder + 'ppp_test.pickle',
                  savefile=savefile,
                  makefigs=makefigs)
    
### This code plots som,ething from single session to test processing
    for session in sessions_to_add:
        s = sessions_to_add[session]
        f, ax = plt.subplots(ncols=4)
        for axis, event in zip(ax, [s.cas['snips_sipper']['filt_avg_z'],
                                    s.malt['snips_sipper']['filt_avg_z'],
                                    s.cas['snips_licks_forced']['filt_avg_z'],
                                    s.malt['snips_licks_forced']['filt_avg_z']]):
            axis.plot(event)
        print(s.peakdiff)

#PPP3-6 excluded completely as no photometry data collected due to 'spinning' rat

if assemble_exclusion:
    sessions = assemble_sessions(ppp_sessions,
                  rats_to_include = ['PPP1-8',
                                     'PPP3-1', 'PPP3-7',
                                     'PPP4-2', 'PPP4-3', 'PPP4-5', 'PPP4-7', 'PPP4-8'],
                  sessions_to_include = ['s10', 's11', 's16'],
                  outputfile='C:\\Github\\PPP_analysis\\data\\ppp_pref_excl.pickle',
                  savefile=savefile,
                  makefigs=makefigs)

#test_sessions = metafile2sessions("..\\data\\test.xlsx",
#                                  "..\\data\\test",
#                                  "..\\data\\",
#                                  "..\\output\\")
#
#s = test_sessions['PPP1-7_s10']
#
#process_rat(s)

#s=sessions_to_add['PPP4-5_s16']



