from scipy.io import wavfile
from bokeh.plotting import figure, show
from scipy.signal import convolve
from bokeh.layouts import column
import numpy as np
'''
This script takes the convolution between the squeak and the full song
'''

# Load audio
sample_freq, x_squeak = wavfile.read('wavs/squeak.wav')
sample_freq, x_walking = wavfile.read('wavs/walking.wav')
x_squeak, x_walking = np.array(x_squeak), np.array(x_walking)
downsample_factor = 2
sample_freq = sample_freq // downsample_factor
x_walking = x_walking[np.arange(len(x_walking)//downsample_factor)*downsample_factor]
x_squeak  = x_squeak[np.arange(len(x_squeak)//downsample_factor)*downsample_factor]

x_walking = x_walking[0:len(x_walking)//8]
n_samples = len(x_walking)
T = n_samples / sample_freq  # total time sampled

# Convolution
# x_conv = convolve(x_walking, x_walking[5*len(x_walking)//1000:6*len(x_walking)//1000], method='direct', mode='same')
# x_conv = convolve(x_walking, x_walking[1000:1010], method='fft', mode='same')
x_conv = []
filter = x_walking[0:200]
for i in range(0,len(x_walking)-len(filter)-1):
    cnv = np.sum(filter*x_walking[i:i+len(filter)])
    x_conv.append(cnv)



# print(x_walking.shape)
# print(x_conv.shape)
# exit()

# Plotting
t = np.arange(0,T,1/sample_freq)
fig_signal = figure(width=1000, x_range=[0,T])
fig_signal.line(t,x_walking)

fig_conv = figure(width=1000) #, x_range=[0,T])
fig_conv.line(range(len(x_conv)),x_conv)

show(column(fig_signal, fig_conv))
# show(fig_signal)