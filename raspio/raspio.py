from gpiozero import MCP3008
from time import sleep 
import matplotlib.pyplot as plt 
import numpy as np 
import time 

mcp = MCP3008(channel=0)

while True:
    print(mcp.value)
