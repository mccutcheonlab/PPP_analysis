# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 05:54:09 2017

@author: jaimeHP
"""

import moviepy.editor as mv
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pyplot as plt

x = rats['PPP1.7'].sessions['s10']


def makevideoclip(videofile, event, data, pre=10, length=30, savefile='output.mp4'):
    vidclip = mv.VideoFileClip(videofile).subclip(event-pre,event-pre+length)
    
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    animation = mv.VideoClip(make_frame(ax), duration=length)

    combinedclip = mv.clips_array([[vidclip, animation]])
    combinedclip.write_videofile(savefile, fps=10)

    return combinedclip

def make_frame(t):
    axislimits, rasterlimits = setfiglims(data)
    
    ax.clear()        
    ax.plot(dataUV[:t*(len(dataUV)/duration)], lw=3, color='grey')
    ax.plot(data[:t*(len(data)/duration)], lw=3, color='white')
    
    ax.vlines([val for val in lickdata if val < t*(len(data)/duration)], rasterlimits[0], rasterlimits[1], color='white', lw=1)
    
    ax.set_xlim(0, len(data))
    ax.set_ylim(axislimits[0], axislimits[1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.text(0, rasterlimits[0], 'licks', fontsize=14, color='w')
    
    return mplfig_to_npimage(fig)

def setfiglims(data):
    datarange = np.max(data) - np.min(data)
    axislimits = [np.min(data), np.max(data)+datarange*0.15]
    rasterlimits = [np.max(data)+datarange*0.05, np.max(data)+datarange*0.15]
      
    return axislimits, rasterlimits

events = x.left['sipper']
videofile = 'R:\\DA_and_Reward\\es334\\PPP1\\Tanks\\Eelke-171027-111329\\PPP1-171017-081744_Eelke-171027-111329_Cam2.avi'


for trial in range(0, len(events)):
    print(trial)
    
    event = x.trialsL[trial]
    data = x.trialsLSnips[trial][:]
    dataUV = x.trialsLSnipsUV[trial]
    
    # Extracts lick data
    lickdatabytrial = jmf.nearestevents(x.trialsL, np.concatenate((x.lickDataL['licks'], x.lickDataR['licks']), axis=0))
    lickdata = lickdatabytrial[trial]
    lickdata = (lickdata+10)*10 # scales to match bin number
    
    savefile = 'R:\\DA_and_Reward\\es334\\PPP1\\video\\combined-' + str(trial) + '.mp4'
    
    #makevideoclip(videofile, event, data, savefile=savefile)
    
    clip = mv.VideoFileClip(videofile).subclip(event-10,event+20)
    print(clip.size)
    c2 = clip.crop(x1=220,y1=0, x2=640, y2=320)
    print(c2.size)
    #c2.save_frame('R:\\DA_and_Reward\\es334\\PPP1\\video\\frame1.png', t=15)
    
    c3 = c2.on_color(size=(420,480), pos='bottom')
    #c3.save_frame('R:\\DA_and_Reward\\es334\\PPP1\\video\\frame2.png', t=15)
    
    duration = 30
    #
    fig, ax = plt.subplots(figsize=(4.2,1.6))
    fig.patch.set_facecolor('k')
    ax.set_facecolor('k')
    
    animation = mv.VideoClip(make_frame, duration=duration)
    #animation.save_frame('R:\\DA_and_Reward\\es334\\PPP1\\video\\frame.png', t=30)
    
    a2 = animation.resize((420,160))
    
    #original_clip = mv.clips_array([[clip, animation]])
    #original_clip.write_videofile('R:\\DA_and_Reward\\es334\\PPP1\\video\\combined.mp4', fps=10)
    
    final_clip =  mv.CompositeVideoClip([c3,
                                a2.set_pos(('center', 'top'))])
    #final_clip.save_frame('R:\\DA_and_Reward\\es334\\PPP1\\video\\frame2.png', t=15)
    final_clip.write_videofile(savefile, fps=10)
    
    #final_clip.ipython_display(width=280)
    #final_clip.save_frame('R:\\DA_and_Reward\\es334\\PPP1\\video\\frame.png', t=10)