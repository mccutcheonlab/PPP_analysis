# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:39:12 2018

@author: jaimeHP
"""

from subprocess import Popen, PIPE, run

Rscriptpath = 'C:\\Program Files\\R\\R-3.5.1\\bin\\Rscript'
Rprogpath = 'C:\\Users\\James Rig\\Documents\\GitHub\\PPP_analysis\\bw_fi_stats.R'

#subprocess.call("C:\\Users\\James Rig\\Documents\\GitHub\\PPP_analysis\\test.Rexec")

#pipe = subprocess.Popen([Rscriptpath, "--vanilla", Rprogpath], stdout=PIPE)
#
#text = pipe.communicate()[0]


result = run([Rscriptpath, "--vanilla", Rprogpath], stdout=PIPE, stderr=PIPE, universal_newlines=True)

print(result.returncode, result.stderr, result.stdout)