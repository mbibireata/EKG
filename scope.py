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
    x = x_max - i
    #y = mcp.value
    y = 1
    line.set_data(x, y)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init, interval=20, blit=True)

plt.show()
