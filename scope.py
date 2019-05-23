from gpiozero import MCP3008
from time import sleep
import matplotlib.pyplot as plt 
from matplotlib import animation
import numpy as np 
import time 

#mcp = MCP3008(channel=0)

# Set up scope parameters 
x_min = 0
x_max = 2
y_min = -2
y_max = 2

# Set up figure, axis, and plot element we want to animate 
fig = plt.figure() 
ax = plt.axes(xlim=(x_min, x_max), ylim=(y_min, y_max))
line, = ax.plot([], [], lw=2)

# Initialization function plotting background of each frame 
def init():
    line.set_data([], [])
    return line,

# Animation function called sequentially
def animate(i):
    x = np.linspace(0, 2, 1000)
    #y = mcp.value
    y = x-i
    line.set_data(x, y)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)

plt.show()
