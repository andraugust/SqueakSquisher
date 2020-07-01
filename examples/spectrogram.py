from scipy.signal import spectrogram
import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import column
from scipy.io import wavfile


sample_freq = 8000
n_samples = 20000
T = n_samples / sample_freq  # total time sampled
t = np.arange(0,T,1/sample_freq)
mod = 20 * np.sin(2*np.pi*2*t)  # time varying frequency
x = np.sin(2*np.pi*1000*t + 2*np.pi*mod)
# wavfile.write('../wavs/spectrogram_signal.wav', rate=sample_freq, data=x)

nperseg = 300
f_, t_, Sxx = spectrogram(x, sample_freq, nperseg=nperseg, noverlap=nperseg-1)

fig_signal = figure(width=1000, x_range=[0,T])
fig_signal.line(t,x)

fig_sgram = figure(x_range=[0,T], y_range=[0,sample_freq//2], width=1000)
fig_sgram.image(image=[Sxx], x=0, y=0, dw=T, dh=sample_freq//2, palette="Spectral11", level="image")

show(column(fig_signal, fig_sgram))