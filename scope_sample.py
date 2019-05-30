import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gpiozero import MCP3008
from scipy.signal import find_peaks
from ekg_utils import *
from time import sleep

mcp = MCP3008(channel=0, differential=True)

class Scope(object):
    def __init__(self, ax, maxt=5., dt=0.02):
        self.ax = ax
        self.dt = dt
        self.maxt = maxt
        self.ymin = -0.1
        self.ymax = 1.1
        #self.title = self.ax.text(7.5, 0.85, "", bbox={'facecolor' : 'w', 'alpha':0.5, 'pad':5}, 
        #                          transform=self.ax.transAxes, ha="center")

        self.tdata = [0]
        self.ydata = [0]
        self.line = Line2D(self.tdata, self.ydata)
        self.ax.add_line(self.line)
        self.ax.set_ylim(self.ymin, self.ymax) 
        self.ax.set_xlim(0, self.maxt)

        self.t_interval = 1.
        
        #Information regarding heartbeat data
        self.sig = HeartSignal()

        self.last_sample_data = []
        self.last_sample_t = []
        
        self.bpm = 0.
        self.out_str = "bpm: {}".format(self.bpm)

    def analyze_rhythm(self, _height=None, _distance=None):
        # Initialize heart signal class instance with last sample's data
        self.sig = HeartSignal(signal = self.last_sample_data)
        self.sig.sig_arr = np.array(self.sig.signal)
        self.sig.t = self.last_sample_t
        self.sig.t_step = self.t_interval

        # Find peaks
        peak_info = self.sig.find_hr_peaks(_height=max(self.sig.sig_arr)*0.6, _distance=10)
        if len(peak_info[0]) == 0:
            self.bpm = 0
        else:
            self.bpm = self.sig.calc_bpm()
        #self.bpm = self.sig.calc_bpm()
        self.out_str="bpm = {}".format(self.bpm)
        self.ax.set_title(self.out_str)

    def find_hr_peaks(self, _height=None, _distance=None):
        peaks = find_peaks(self.ydata, height=_height, distance=_distance)
        peak_idx = peaks[0]
        y_vals = np.array(y[i] for i in peak_idx)
        return([peak_idx, y_vals])

    def calc_bpm(self):
        pass

    #TODO?: logic to not reset image but just to shift all data back smoothly
    def update(self, y):
        lastt = self.tdata[-1]
        if lastt > self.tdata[0] + self.maxt:  # reset the arrays
	    #Save last samples for analysis
            self.last_sample_data = self.ydata
            self.last_sample_t = self.tdata
            self.analyze_rhythm()
	    #reset scope image 
            self.tdata = [self.tdata[-1]]
            self.ydata = [self.ydata[-1]]
            self.ax.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
            self.ax.figure.canvas.draw()

        t = self.tdata[-1] + self.dt
        self.tdata.append(t)
        self.ydata.append(y)
        self.line.set_data(self.tdata, self.ydata)
        return self.line,

    def update2(self, y):
        lastt = self.tdata[-1]
        if lastt > self.tdata[0] + self.maxt:
            self.last_sample_data = self.ydata
            self.last_sample_t = self.tdata

            self.tdata = [self.tdata[-1]]
            self.ydata = [self.ydata[-1]]
            self.ax.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
            self.ax.figure.canvas.draw()
	
        t = self.tdata[-1] + self.dt
        self.tdata.append(t)
        self.ydata.append(y)
        self.line.set_data(self.tdata, self.ydata)
        return self.line,


def emitter():
    'return output from MCP'
    while True:
        yield mcp.value
        #yield np.random.random_sample()


fig, ax = plt.subplots()
scope = Scope(ax)

# pass a generator in "emitter" to produce data for the update func
ani = animation.FuncAnimation(fig, scope.update, emitter, interval=scope.t_interval,
                              blit=True)
plt.title(scope.out_str)
plt.show()
