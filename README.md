# PPP
# Created by Jaime on 8 Nov 2017

# This repo will contain analysis code for PPP experiment (Protein Preference Photometry).

# Steps to perform analysis are as follows:
# TDT files (in tanks) need to be extracted in Matlab using TDTbin2mat. This is accomplished by using PPP1_indexer.m which reads metadata from the metafile and saves extracted streams as .mat structured files.
# The main Python script (v. 3.6), PPP1_analysis, will then use the same metafile to read in .mat files into an object-orientated structure with each Session nested within each Rat.
