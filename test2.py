import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from ekg_utils import *

if __name__ == "__main__":
    sig = HeartSignal()
    V,t = sig.read_file("ScopeData/ALL0005/F0005CH1.CSV")  
    min_thresh = -1.
    V = np.array([i if i>min_thresh else min_thresh for i in V])
    
    peak_info = sig.find_hr_peaks(_height=max(V)*0.6, _distance=10)

    bpm = sig.calc_bpm()
    #t -= 2.5

    plt.plot(t, V, label="Signal")
    plt.scatter([t[i] for i in peak_info[0]], peak_info[1], color="red", label="Peak")
    plt.title("BPM = {}".format(bpm))
    plt.ylabel("Voltage, V")
    plt.xlabel("Time, s")
    plt.legend()
    plt.savefig("figures/Jena.png")
    plt.show()
    
    
