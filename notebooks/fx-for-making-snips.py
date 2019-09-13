# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:51:54 2019

@author: jmc010
"""

def mastersnipper(x, events,
                  bins=300,
                  preTrial=10,
                  trialLength=30,
                  threshold=10,
                  peak_between_time=[0, 1],
                  output_as_dict=True,
                  latency_events=[],
                  latency_direction='pre',
                  verbose=True):
    
    if len(events) < 1:
        print('Cannot find any events. All outputs will be empty.')
        blueTrials, uvTrials, noiseindex, diffTrials, peak, latency = ([] for i in range(5))
    else:
        if verbose: print('{} events to analyze.'.format(len(events)))
        
        blueTrials,_ = snipper(x.data, events,
                                   t2sMap=x.t2sMap,
                                   fs=x.fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength)
        uvTrials,_ = snipper(x.dataUV, events,
                                   t2sMap=x.t2sMap,
                                   fs=x.fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength)
        filtTrials,_ = snipper(x.data_filt, events,
                                   t2sMap=x.t2sMap,
                                   fs=x.fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength,
                                   adjustBaseline=False)
        
        filtTrials_z = zscore(filtTrials)
        filtTrials_z_adjBL = zscore(filtTrials, baseline_points=50)
        
        bgMAD = findnoise(x.data_filt, x.randomevents,
                              t2sMap=x.t2sMap, fs=x.fs, bins=bins,
                              method='sum')        
        sigSum = [np.sum(abs(i)) for i in filtTrials]
        sigSD = [np.std(i) for i in filtTrials]
        noiseindex = [i > bgMAD*threshold for i in sigSum]
        
                
        # do i need to remove noise trials first before averages
        filt_avg = np.mean(removenoise(filtTrials, noiseindex), axis=0)
        if verbose: print('{} noise trials removed'.format(sum(noiseindex)))
        filt_avg_z = zscore(filt_avg)

    
        bin2s = bins/trialLength
        peakbins = [int((preTrial+peak_between_time[0])*bin2s),
                    int((preTrial+peak_between_time[1])*bin2s)]
        peak = [np.mean(trial[peakbins[0]:peakbins[1]]) for trial in filtTrials_z]
        
        latency = []
        try:
            for event in events:
                if latency_direction == 'pre':
                    latency.append(np.abs([lat-event for lat in latency_events if lat-event<0]).min())
                elif latency_direction == 'post':
                    latency.append(np.abs([lat-event for lat in latency_events if lat-event>0]).min())
                else:
                    latency.append(np.abs([lat-event for lat in latency_events]).min())
#            latency = [x if (x<30) else None for x in latency]
            latency = np.asarray(latency)
            latency[latency>30] = np.nan
        except ValueError:
            print('No latency events found')

    if output_as_dict == True:
        output = {}
        output['blue'] = blueTrials
        output['uv'] = uvTrials
        output['filt'] = filtTrials
        output['filt_z'] = filtTrials_z
        output['filt_z_adjBL'] = filtTrials_z_adjBL
        output['filt_avg'] = filt_avg
        output['filt_avg_z'] = filt_avg_z
        output['noise'] = noiseindex
        output['peak'] = peak
        output['latency'] = latency
        return output
    else:
        return blueTrials, blueTrials_raw, blueTrials_z, blueTrials_z_adjBL, uvTrials, uvTrials_raw, uvTrials_z, noiseindex, diffTrials, peak, latency

def zscore(snips, baseline_points=100):
    
    BL_range = range(baseline_points)
    z_snips = []
    try:
        for i in snips:
            mean = np.mean(i[BL_range])
            sd = np.std(i[BL_range])
            z_snips.append([(x-mean)/sd for x in i])
    except IndexError:
        mean = np.mean(snips[BL_range])
        sd = np.std(snips[BL_range])
        z_snips = [(x-mean)/sd for x in snips]

    return z_snips

"""
This function will check for traces that are outliers or contain a large amount
of noise, relative to other trials (or relative to the whole data file.
"""


def findnoise(data, background, t2sMap = [], fs = 1, bins=0, method='sd'):
    
    bgSnips, _ = snipper(data, background, t2sMap=t2sMap, fs=fs, bins=bins)
    
    if method == 'sum':
        bgSum = [np.sum(abs(i)) for i in bgSnips]
        bgMAD = med_abs_dev(bgSum)
    elif method == 'sd':
        bgSD = [np.std(i) for i in bgSnips]
        bgMAD = med_abs_dev(bgSD)
   
    return(bgMAD)

def removenoise(snipsIn, noiseindex):
    snipsOut = np.array([x for (x,v) in zip(snipsIn, noiseindex) if not v])   
    return snipsOut

def findphotodiff(blue, UV, noise):
    blueNoNoise = removenoise(blue, noise)
    UVNoNoise = removenoise(UV, noise)
    diffSig = blueNoNoise-UVNoNoise
    return diffSig

def makerandomevents(minTime, maxTime, spacing = 77, n=100):
    events = []
    total = maxTime-minTime
    start = 0
    for i in np.arange(0,n):
        if start > total:
            start = start - total
        events.append(start)
        start = start + spacing
    events = [i+minTime for i in events]
    return events

def med_abs_dev(data, b=1.4826):
    median = np.median(data)
    devs = [abs(i-median) for i in data]
    mad = np.median(devs)*b
                   
    return mad