# PPP (Protein Preference Photometry) Analysis
Created by Jaime McCutcheon on 8 Nov 2017
Edited 2017-2021
Uses Python 3.x

This repository contains contain analysis code for an experiment in which neural activity in ventral tegmental area (VTA) was measured using fiber photometry in rats that were either protein-restricted or on control diet. Full details of the experiment can be found at https://www.biorxiv.org/content/10.1101/542340v3.

Original data files were collected on Tucker Davis Tehcnologies equipment and are arranged into "tanks" that contain photometry signals, TTLs corresponding to behavioral events, and .avi files with videos of each session. Each tank contains data from two separate behavioral chambers (boxes). Metafiles exist for each cohort of rats (3 in total) which contain information pertaining to each tank including: rat, date, box, diet group, identifiers for photometry signals and TTLs. These metafiles are used by the scripts to extract required data from the raw data tanks. All data files are available at doi: 10.25392/leicester.data.7636268.

## Steps to perform analysis are as follows:
[optional] Install environment from <i>environment.yml</i>

### Extract data and assemble dataframes for subsequent analysis
<i>main.py</i> - Uses metafiles to extract data from appropriate tanks and saves as .pickle file. This .pickle file contains a dictionary (sessions) in which each key is a subdictionary conating data from a single rat on a single day (e.g. "PPP1-1_s10" contains data from rat PPP1-1 on the 10th day (s10)). This script uses functions within _fx4assembly.py_ and _fx4behavior.py_ to accomplish this.

<i>ppp_averages_pref.py</i> - Loads in .pickle file and places data for plotting into pandas dataframes that can be indexed and selected via diet group etc. Saves several dataframes including: df_photo (all photometry data), df_behav (all behavioral data). All subsequent analysis and plotting can be conducted using these dataframes, which are much smaller than the raw and processed data files.

### Plotting figures
_ppp_pub_figs_settings.py_ - This script is run at the start of other scripts and loads in plotting conventions (e.g. font sizes, colors etc) as well as the required data from dataframes.

### Statistics
The <i>stats</i> folder contains Python and R scripts that conduct ANOVAs. This requires R to be installed and paths to _Rscript.exe_ to be entered. In the absence of this, .csv files can be used in another statistical package, e.g. SPSS. Estimation stats are conducted within the main plotting scripts using the dabest package.  

### Extras
<i>ppp_bw and fi.py</i> - contains analysis of food and body weight data.


