import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import time
import numpy as np
import random


dat=[0,1]
fig = plt.figure()
ax = fig.add_subplot(111)
Ln, = ax.plot(dat)
ax.set_xlim([0,20])
plt.ion()
plt.show()    
for i in range (18):
    dat.append(random.uniform(0,1))
    Ln.set_ydata(dat)
    Ln.set_xdata(range(len(dat)))
    plt.pause(1)