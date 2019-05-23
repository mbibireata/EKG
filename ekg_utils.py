import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math
import pandas as pd 

class HeartSignal:
    def __init__(self, signal = np.array([]), sample_rate = 100):
        self.signal = signal
        self.sample_rate = sample_rate
        self.signal_arr = np.array(signal)
        self.t = np.arange(len(self.signal))
        self.t_step = 1.
        self.peak_idx = []
        self.peak_y_vals = []

    def __repr__(self):
        return str(self.signal)

    def read_file(self, path, f_type="scope"):
        if f_type == "scope": 
            df = pd.read_csv(path, header=None, usecols=[1,3,4])
            t = np.array(df[3])
            V = np.array(df[4])
            self.signal = V 
            self.t = t
            self.signal_arr = np.array(V)
            self.t_step = float(df[1].iloc[11])
            
            return(V, t)
        else: 
            pass

    def plot(self):
        plt.plot(self.signal)
        plt.show()

    def set_t_step(self, t_step):
        self.t_step = t_step

    '''
    ### TODO: Generalize
    def calc_bpm(self):
        #Calculate moving average with 0.75s in both directions, append to dataset
        hrw = 0.75
        mov_avg = self.signal.rolling(int(hrw*self.sample_rate)).mean()
        avg_hr = (np.mean(self.signal))
        mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
        mov_avg = [x*1.2 for x in mov_avg]
        rolling_mean = mov_avg

        #Mark regions of interest 
        window = []
        peaklist = []
        listpos = 0
        for datapoint in self.signal:
            r_mean = rolling_mean[listpos]
            if (datapoint < r_mean) and (len(window) < 1):
                listpos += 1
            elif (datapoint > r_mean):
                window.append(datapoint)
                listpos += 1
            else: 
                maximum = max(window)
                beatposition = listpos - len(window) + (window.index(max(window)))
                peaklist.append(beatposition)
                window = []
                listpos += 1

        ybeat = [self.signal[x] for x in peaklist] #y value of all peaks 

        hb_dist_list = []
        for idx in range(len(peaklist) - 1):
            hb_dist = (peaklist[idx+1] - peaklist[idx]) / self.sample_rate
            hb_dist_list.append(hb_dist)
        
        hb_dist_list = np.array(hb_dist_list)
        heart_rate = np.mean(hb_dist_list)*60 #60 comes from scaling to BPM from BPS

        plt.plot(self.signal, alpha=0.5, color='blue')
        plt.scatter(peaklist, ybeat, color='red', label="BPM: %.3f"%heart_rate)
        plt.legend()
        plt.show()

        return(heart_rate)
    '''

    def calc_bpm(self): 
        hb_dist_list = []
        for idx in range(len(self.peak_idx) - 1):
            hb_dist = (self.t[self.peak_idx[idx+1]] - self.t[self.peak_idx[idx]]) #/ self.t_step        
            hb_dist_list.append(hb_dist)

        hb_dist_list = np.array(hb_dist_list)
        heart_rate = 1/np.mean(hb_dist_list)*60 #60 comes from scaling to BPM from BPS
        return(heart_rate)




    # Works very well on good sample using scipy. 
    def find_hr_peaks(self, use_scipy=True, _height=None, _distance=None):
        
        if use_scipy: 
            from scipy.signal import find_peaks
            peaks = find_peaks(self.signal_arr, height=_height, distance=_distance)
            peak_idx = peaks[0]
            y_vals = np.array([self.signal_arr[i] for i in peak_idx])
            self.peak_idx = peak_idx
            self.peak_y_vals = y_vals
            return([peak_idx, y_vals])
        else:
            pass

    def compute_signal_fft(self, _type="real"):
        if _type == "real": 
            self.sig_fft = np.fft.rfft(self.signal)
            self.sig_fft_freq = np.fft.rfftfreq(len(self.sig_fft), d=1./self.sample_rate)

        elif _type == "":
            self.sig_fft = np.fft.fft(self.signal)
            self.sig_fft_freq = np.fft.fftfreq(len(self.sig_fft), d=1./self.sample_rate)

        elif _type == "imaginary":
            self.sig_fft = np.fft.ifft(self.signal)
            self.sig_fft_freq = np.fft.ifftfreq(len(self.sig_fft), d=1./self.sample_rate)

        else:
            return False
            
             

if __name__ == "__main__":
    import heartpy as hp
    import pandas as pd
    sig = pd.read_csv("data.csv").hart #Sample data recorded at 100 Hz 
    #sig = pd.read_csv("heartrate_analysis_python/data3.csv").hart.iloc[50000:55000]
    #sig =  np.array(sig).flatten()
    hr = HeartSignal(sig)
    #hr.calc_bpm()
    peak_data=hr.find_hr_peaks(_height=600)
    peak_idx = peak_data[0]
    y_vals = peak_data[1]

    plt.plot(hr.signal_arr)
    plt.scatter(peak_idx, y_vals, color="red")
    plt.show()
