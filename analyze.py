import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from ekg_utils import *

if __name__== "__main__":
    sig = HeartSignal()
    path = "ScopeData/ALL0005/F0005CH1.CSV"
    V,t = sig.read_file(path)

    # Preprocess data
    min_thresh = -5.
    corrected_val = 0.
    V = np.array([i if i>min_thresh else corrected_val for i in V])

    # Get peak info for main r wave. Saved as class data member.
    distance = 10
    height_thresh_ratio = 0.6
    # peak_info is [[idx],[y val]]
    peak_info = sig.find_hr_peaks(_height=max(V)*height_thresh_ratio,
                                  _distance=distance)

    bpm = sig.calc_bpm()

    # Find P waves

    # Find S waves

    # Find T waves

    # Compute fourier transform of signal
    sec_per_div = 10 * sig.t_step / len(V) # 10 subdivisions on oscilloscope for given t_step

    sig_fft = -np.fft.rfft(V)
    sig_fft_freq = np.fft.rfftfreq(len(t), sec_per_div)

    print(t[1]-t[0])
    print(sec_per_div)
    print(bpm/60)

    plt.figure(1)
    plt.plot(t, V)
    plt.scatter([t[i] for i in peak_info[0]], peak_info[1], color="red")
    plt.title("BPM = {}".format(bpm))
    plt.figure(2)
    plt.plot(sig_fft_freq, sig_fft)
    plt.show()
