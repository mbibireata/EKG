import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from ekg_utils import *

if __name__ == "__main__":
    t = pd.read_csv("ScopeData/sample1.CSV").t
    V = pd.read_csv("ScopeData/sample1.CSV").V
    x_scale = 10 ** -3 #sec
    y_scale = 1/0.5 #V
    V = np.array(V)
    t = np.array(t)
    V *= y_scale
    t *= x_scale
    _height = max(V)/2.
    peaks = find_peaks(V, height=_height)
    peak_idx = peaks[0]
    y_vals = np.array([V[i] for i in peak_idx])

    plt.plot(V)
    plt.scatter(peak_idx, y_vals, color="red")
    plt.show()

