from gpiozero import MCP3008
from time import sleep 
import matplotlib.pyplot as plt 
import numpy as np 
import time 

mcp = MCP3008(channel=0, differential=True)
delay = 0.005 #Seconds
n_samples = 1000
vals = []

for i in range(n_samples):
    vals.append(mcp.value)
    sleep(delay)

plt.plot(vals)
plt.ylim(-0.1,1.1)
plt.show()
