import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from ekg_utils import *

if __name__== "__main__":
    sig = HeartSignal()
    path = "ScopeData/ALL0011I/F0011CH1.CSV"
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

    aux_peak_info = sig.find_aux_peaks(_distance=distance, d_scale=3)

    # Find S waves

    # Compute fourier transform of signal
    sec_per_div = 10 * sig.t_step / len(V) # 10 subdivisions on oscilloscope for given t_step

    sig_fft = -np.fft.rfft(V)
    sig_fft_freq = np.fft.rfftfreq(len(t), sec_per_div)

    # Find P waves 
    avg_pr_dist = np.mean(peak_info[0][1:] - aux_peak_info[2])*sec_per_div

    # Find T waves
    avg_rs_dist = np.mean(aux_peak_info[0] - peak_info[0][:-1])*sec_per_div


    
    #print(t[1]-t[0])
    #print(sec_per_div)
    #print(bpm/60)

    plt.figure(1)
    plt.plot(t, V)
    plt.scatter([t[i] for i in peak_info[0]], peak_info[1], color="red")
    plt.scatter([t[i] for i in aux_peak_info[0]], aux_peak_info[1], color="green")
    plt.scatter([t[i] for i in aux_peak_info[2]], aux_peak_info[3], color="purple")
    plt.title("BPM = {:.4f},\n PR Segment = {:.4f},\n RT Segment = {:.4f}".format(bpm, avg_pr_dist, avg_rs_dist))
    plt.xlabel("time, (s)")
    plt.ylabel("Voltage, (V)")
    #plt.savefig("figures/analysis6.png")
    plt.figure(2)
    plt.plot(sig_fft_freq, sig_fft)
    plt.xlabel("Frequency, (Hz)")
    #plt.savefig("figures/analysis6_fft.png")
    plt.show()
