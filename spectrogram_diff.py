from scipy.io import wavfile
from scipy.fftpack import rfft
from scipy import signal
from bokeh.plotting import figure, show
from scipy.signal import spectrogram, istft
from bokeh.layouts import column
import numpy as np
'''
This script learns a filter from the squeak and applies it to the squeak
'''

# Load audio
sample_freq, x_w = wavfile.read('wavs/walking.wav')
sample_freq, x_s = wavfile.read('wavs/squeak.wav')
x_w, x_s = np.array(x_w), np.array(x_s)


def downsample_signal(x, factor):
    return x[np.array(range(len(x)//factor))*factor]

def plot_spectrograms(S_w, S_s):
    S_w, S_s = np.log(S_w+1), np.log(S_s+1)
    fig_w = figure(x_range=[0,T_w], y_range=[0,sample_freq//2], width=1000)
    fig_w.image(image=[S_w], x=0, y=0, dw=T_w, dh=sample_freq//2, palette="Spectral11", level="image")
    fig_s = figure(x_range=[0,T_s], y_range=[0,sample_freq//2], width=1000)
    # fig_s.image(image=[S_s], x=0, y=0, dw=T_s, dh=sample_freq//2, palette="Spectral11", level="image")
    fig_s.image(image=[S_s], x=0, y=0, dw=T_s, dh=sample_freq//2, palette="Spectral11", level="image")
    show(column(fig_w, fig_s))

def line_plot(x,y):
    fig = figure(width=1000)
    fig.line(x, y)
    show(fig)


# Downsample to make the program run faster
downsample_factor = 2
x_w = x_w[0:len(x_w)]
x_w = downsample_signal(x_w, downsample_factor)
x_s = downsample_signal(x_s, downsample_factor)
n_w, n_s = len(x_w), len(x_s)
sample_freq = sample_freq // downsample_factor
T_w, T_s = n_w / sample_freq, n_s / sample_freq  # total time sampled
# wavfile.write('wavs/walking_.wav', rate=sample_freq, data=x_w)


# Make a spectrogram for each signal
print('Making spectrograms...')
nperseg = 128
noverlap = nperseg - 1
# noverlap = nperseg*8//10
f_w, t_w, S_w = spectrogram(x_w, sample_freq, nperseg=nperseg, noverlap=noverlap)
f_s, t_s, S_s = spectrogram(x_s, sample_freq, nperseg=nperseg, noverlap=noverlap)
# plot_spectrograms(S_w, S_s)
# print(S_w.shape[-1])
# print(len(x_w))
# exit()


# Make the squeak filter
filter = np.sum(S_s, axis=1)
filter = filter / np.sum(filter)


# Convolve the filter with the song
print('Taking convolution...')
conv = []
for t in range(len(t_w)):
    val = np.sum(filter * S_w[:,t] / (np.sum(S_w[:,t]+1)))  # normalize the song at each time-step
    conv.append(val)
conv = np.array(conv)**2
# line_plot(range(len(conv)), conv)
len_diff = len(x_w) - len(conv)
x_w = x_w[:-len_diff]


# Zero values that convolve above a threshold or have neighbors above threshold
print('Zeroing matches...')
thresh = 0.00012
width = 300
conv_masked = conv[:]
for i in range(len(conv)-width):
    if np.any(conv[i:i+width] > thresh):
        x_w[i] = 0


# Smooth the convolution result by taking the mean of each point's neighbors
# print('Smoothing convolution...')
# smoothing_width = 100
# a = smoothing_width//2
# t_stop = len(conv) - a
# conv_smooth = [0]*a  # padding
# for t in range(a, t_stop):
#     conv_smooth.append(np.mean(conv[t-a:t+a]))
# conv_smooth += [0]*(len(x_w)-len(conv_smooth))  # padding
# conv_smooth = np.array(conv_smooth)
# conv_smooth = np.tanh( np.pi * np.array(conv_smooth) / np.max(conv_smooth))
# line_plot(t_w, conv_smooth)
# exit()


# Dampen song when the convolution is high
print('Saving wav file...')
# x_w[conv_smooth > 0.00025] = 0
# x_w = x_w * (1. - conv_smooth)
# x_w = x_w.astype(np.int16)
wavfile.write('wavs/walking_inverted.wav', rate=sample_freq, data=x_w)


# Invert spectrogram
# t, x_w_inverted = istft(np.log(S_w+1), fs=sample_freq, nperseg=nperseg, noverlap=noverlap)
# t, x_w_inverted = istft(S_w, fs=sample_freq, nperseg=nperseg, noverlap=noverlap)
# fig_w = figure(width=1000)
# fig_w_inverted = figure(width=1000)
# fig_w_inverted.line(t, x_w_inverted)
# fig_w.line(np.arange(0,T_w,1/sample_freq), x_w)
# show(column(fig_w, fig_w_inverted))
# wavfile.write('wavs/walking_inverted.wav', rate=sample_freq, data=x_w_inverted)
