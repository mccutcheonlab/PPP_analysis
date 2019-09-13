# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:54:33 2019

@author: jmc010
"""


"""
This version of mastersnipper was written once Vaibhav fft baseline correction 
had been implemented and focuses only on 'filtered' signal (i.e. ignores blue 
and UV signals)
"""

def mastersnipper_filt(x, events,
                  bins=300,
                  preTrial=10,
                  trialLength=30,
                  threshold=10,
                  peak_between_time=[0, 1],
                  output_as_dict=True,
                  latency_events=[],
                  latency_direction='pre',
                  verbose=False):
    
    if len(events) < 1:
        print('Cannot find any events. All outputs will be empty.')
        blueTrials, uvTrials, noiseindex, diffTrials, peak, latency = ([] for i in range(5))
    else:
        if verbose: print('{} events to analyze.'.format(len(events)))
        
        filtTrials,_ = snipper(x.data_filt, events,
                                   t2sMap=x.t2sMap,
                                   fs=x.fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength)
        
        filtTrials_z = zscore(filtTrials)
        filtTrials_z_adjBL = zscore(filtTrials, baseline_points=50)
        
        bgMAD = findnoise(x.data_filt, x.randomevents,
                              t2sMap=x.t2sMap, fs=x.fs, bins=bins,
                              method='sum')        
        sigSum = [np.sum(abs(i)) for i in filtTrials]
        sigSD = [np.std(i) for i in filtTrials]
        noiseindex = [i > bgMAD*threshold for i in sigSum]
        
                
        # do i need to remove noise trials first before averages
        filt_avg = np.mean(filtTrials, axis=0)
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
        output['filt'] = filtTrials
        output['filt_z'] = filtTrials_z
        output['filt_z_adjBL'] = filtTrials_z_adjBL
        output['filt_avg'] = filt_avg
        output['noise'] = noiseindex
        output['peak'] = peak
        output['latency'] = latency
        return output
    else:
        return blueTrials, blueTrials_raw, blueTrials_z, blueTrials_z_adjBL, uvTrials, uvTrials_raw, uvTrials_z, noiseindex, diffTrials, peak, latency
