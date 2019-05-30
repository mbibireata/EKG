import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gpiozero import MCP3008

mcp = MCP3008(channel=0, differential=True)

class Scope(object):
    def __init__(self, ax, maxt=10., dt=0.02):
        self.ax = ax
        self.dt = dt
        self.maxt = maxt
        self.ymin = -0.1
        self.ymax = 1.1


        self.tdata = [0]
        self.ydata = [0]
        self.line = Line2D(self.tdata, self.ydata)
        self.ax.add_line(self.line)
        self.ax.set_ylim(self.ymin, self.ymax) 
        self.ax.set_xlim(0, self.maxt)
        self.last_sample_data = []
        self.last_sample_t = []
        self.bpm = 0.
        self.out_str = "Hello World!"

    #TODO?: logic to not reset image but just to shift all data back smoothly
    def update(self, y):
        lastt = self.tdata[-1]
        if lastt > self.tdata[0] + self.maxt:  # reset the arrays
	    #Save last samples for analysis
            self.last_sample_data = self.ydata
            self.last_sample_t = self.tdata
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
ani = animation.FuncAnimation(fig, scope.update, emitter, interval=5,
                              blit=True)
plt.title(scope.out_str)
plt.show()
