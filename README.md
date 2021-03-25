# PPP (Protein Preference Photometry) Analysis
Created by Jaime McCutcheon on 8 Nov 2017
Edited 2017-2021
Uses Python 3.x

This repository contains contain analysis code for an experiment in which neural activity in ventral tegmental area (VTA) was measured using fiber photometry in rats that were either protein-restricted or on control diet. Full details of the experiment can be found at https://www.biorxiv.org/content/10.1101/542340v3.

Original data files were collected on Tucker Davis Technologies equipment and are arranged into "tanks" that contain photometry signals, TTLs corresponding to behavioral events, and .avi files with videos of each session. Each tank contains data from two separate behavioral chambers (boxes). Metafiles exist for each cohort of rats (3 cohorts in total named PPP1, PPP3 and PPP4) which contain information pertaining to each tank including: rat, date, box, diet group, identifiers for photometry signals and TTLs. These metafiles are used by the scripts to extract required data from the raw data tanks. All data files are available at doi: 10.25392/leicester.data.7636268.

## Steps to perform analysis are as follows:
[optional] Install environment from _environment.yml_. This file gives details of all versions of packages used for analysis. Instructions on creating an environment from this file are found <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file" target="_blank">here</a>.

### Extract data and assemble dataframes for subsequent analysis
[optional steps to extract data from raw data files]
_main.py_ - Uses metafiles to extract data from appropriate tanks and saves as .pickle file(s). This .pickle file contains a dictionary (sessions) in which each key is a subdictionary conating data from a single rat on a single day (e.g. "PPP1-1_s10" contains data from rat PPP1-1 on the 10th day (s10)). This script uses functions within _fx4assembly.py_ to accomplish this.

_make_dfs.py_ - Loads in .pickle file and places data for plotting into pandas dataframes that can be indexed and selected via diet group etc. Saves several dataframes including: df_photo (all photometry data), df_behav (all behavioral data). All subsequent analysis and plotting can be conducted using these dataframes, which are much smaller than the raw and processed data files.

Most analysis can be conducted using the dataframes which are available as .pickle files and can be directly downloaded using the following directions:

1. After cloning repository, navigate to directory containing notebooks (e.g. `cd PPP_analysis\notebooks`)
2. Open a Python session (`python`)
3. Run the following code
```
import sys
sys.path.append('..\\helperfx')
from download_dfs import *
download_dfs()with Python.
```
This function `download_dfs` can also be run in a standard IDE (e.g. Spyder) or in Jupyter but you must ensure that the `data` directory that is created is in the main directory (`PPP_analysis`) or it will not be found by the subsequent steps.

### Plotting figures
All figures are plotted in the corresponding notebooks located in the `notebooks` folder. These figures depend on the following files found in the `helperfx`directory:

_settings4figs.py_ - This script is run at the start of other scripts and loads in plotting conventions (e.g. font sizes, colors etc) as well as the required data from dataframes.
_fx4figs-py_ - This script contains the functions used to make figures and is called by the notebooks.

### Statistics
Most statistics are reported in the notebook, _stats_notebook.ipynb_. The _stats_ folder contains R scripts that are required to conduct ANOVAs. This requires R to be installed and paths to _Rscript.exe_ to be entered. In the absence of this, .csv files can be used in another statistical package, e.g. SPSS. Estimation stats are conducted within the main plotting scripts using the dabest package and added to an Excel file in the _stats_ folder.  

### Extras
Scripts for video tracking of rats using DeepLabCut are provided in the folder _video_.


