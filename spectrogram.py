from scipy.io import wavfile
from scipy.fftpack import rfft
from scipy import signal
from bokeh.plotting import figure, show
from scipy.signal import spectrogram
from bokeh.layouts import column
import numpy as np
'''
This script learns a filter from the squeak and applies it to the squeak
'''

# Load audio
wav_file = 'wavs/walking.wav'
# wav_file = 'wavs/squeak.wav'
sample_freq, x = wavfile.read(wav_file)
# x = x[0:len(x)//4]
n_samples = len(x)
T = n_samples / sample_freq  # total time sampled

# Get spectrogram
nperseg = 256
f_, t_, Sxx = spectrogram(x, sample_freq, nperseg=nperseg, noverlap=nperseg//2)

t = np.arange(0,T,1/sample_freq)
fig_signal = figure(width=1000, x_range=[0,T])
fig_signal.line(t,x)

fig_sgram = figure(x_range=[0,T], y_range=[0,sample_freq//2], width=1000)
fig_sgram.image(image=[np.log(Sxx+1)], x=0, y=0, dw=T, dh=sample_freq//2, palette="Spectral11", level="image")

show(column(fig_signal, fig_sgram))


