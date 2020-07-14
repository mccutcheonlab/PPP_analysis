# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:46:40 2020

This is a simple patch script that fixes problems in the saved session file
relating to diet data.

It basically "copies and pastes the diet (e.g. group) values from s10 into
s11 and s16 and resaves the file.

Should become unecessary when metafile is fixed and assembly script is re-run."

@author: admin
"""
import dill


try:
    type(sessions)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    try:
        pickle_in = open('C:\\Github\\PPP_analysis\\data\\ppp_pref.pickle', 'rb')
    except FileNotFoundError:
        print('Cannot access pickled file')
    sessions, rats = dill.load(pickle_in)

NR_rats, PR_rats = [], []

for key in sessions.keys():
    s = sessions[key]
    if s.session == 's10':
        if s.diet == 'NR':
            NR_rats.append(s.rat)
        elif s.diet == 'PR':
            PR_rats.append(s.rat)
        else:
            print(s.rat, " cannot be assigned")
        

print("PPP1-5, s11 ", sessions["PPP1-5_s11"].diet)

for key in sessions.keys():
    s = sessions[key]
    if s.session == 's11' or s.session == 's16':
        if s.rat in NR_rats:
            s.diet = "NR"
        elif s.rat in PR_rats:
            s.diet = "PR"
        else:
            print(s.rat, " not in either list")
            
print("PPP1-5, s16 ", sessions["PPP1-5_s16"].diet)

outputfile='C:\\Github\\PPP_analysis\\data\\ppp_pref.pickle'
savefile=True

if savefile:
    pickle_out = open(outputfile, 'wb')
    dill.dump([sessions, rats], pickle_out)
    pickle_out.close()