from scipy.io import wavfile
from scipy.fftpack import rfft, ifft, fft, irfft
from scipy import signal
from bokeh.plotting import figure, show
from bokeh.layouts import column
import numpy as np
'''
This script learns a filter from the squeak and applies it to the squeak
'''

# Load audio
# sample_rate, x_walk = wavfile.read('walking_mono.wav')
sample_rate, x_squeak = wavfile.read('squeak_mono.wav')

# n_walk = len(x_walk)
n_squeak = len(x_squeak)

# Fourier transform squeak
f_squeak = rfft(x_squeak)
# wavfile.write('squeak_mono_after.wav', rate=sample_rate, data=x_squeak_after.astype(np.int16))

# Spectrogram
sgram_f, sgram_t, sgram_vals = signal.spectrogram(x_squeak)  # vals are n_frequenxies x n_times
print('Frequency shape', sgram_f.shape)
print('Times shape', sgram_t.shape)
print('Spectrogram shape', sgram_vals.shape)
exit()





# Frequencies
frequencies = np.linspace(0,sample_rate/2,len(amplitudes)//2)

p_ft = figure(width=1200)
p_ft.line(frequencies, amplitudes**2)
p_sig = figure(width=1000)
show(p_ft)
# show(column(p_ft, p_sig))