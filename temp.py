# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:39:12 2018

@author: jaimeHP
"""


"""
Code for running stats using R

This requires R to be installed and an Rscript written. At the moment I am using
the R package, EZ, which makes running mixed, between-within ANOVAs simple, and
tests for sphericity etc as appropriate.

EZ can be installed using the command install.packages('ez') in R. The package
seems to work best in R3.4.4 or later.

An R script is written to run the analysis and print the results. This script
is then called by Rscript.exe via the subprocess module in Python.

"""

from subprocess import PIPE, run

Rscriptpath = 'C:\\Program Files\\R\\R-3.5.1\\bin\\Rscript'
Rprogpath = 'C:\\Users\\James Rig\\Documents\\GitHub\\PPP_analysis\\bw_fi_stats.R'


result = run([Rscriptpath, "--vanilla", Rprogpath], stdout=PIPE, stderr=PIPE, universal_newlines=True)

print(result.returncode, result.stderr, result.stdout)