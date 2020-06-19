from scipy.fftpack import rfft, fft
from bokeh.plotting import figure, show
from bokeh.layouts import column
import numpy as np
'''
Do a fft on a signal and plot with frequency axis scaled appropriately
'''

# Make signal
sample_freq = 10
n_samples = 1000
T = n_samples / sample_freq  # total time sampled
t = np.arange(0,T,1/sample_freq)
x = np.sin(2*np.pi* 0.5*t) + np.sin(2*np.pi* 0.2*t)

# Fourier transform squeak
amplitudes = rfft(x)

# Frequencies (horizontal axis)
f = np.linspace(0, sample_freq/2, len(amplitudes))

# Plot
fig_x = figure(width=1200)
fig_f = figure(width=1200)
fig_x.line(t, x)
fig_f.line(f, amplitudes)
show(column(fig_x, fig_f))