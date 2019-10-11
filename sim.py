import numpy as np
import time
import os
import sys
np.random.seed(int(time.time()))

def step():
    # generate number between -1 and 1 for x,y,z directions
    [x,y,z] = np.random.uniform(low=-1, high = 1, size=3)

    # return only points that lie within a circle of radius 1
    r = np.sqrt(x**2 + y**2 + z**2)
    if r <= 1:
        return [x,y,z]
    else:
        return 0

def walk3(Np, Nt):
    x = np.zeros(Np)
    y = np.zeros(Np)
    z = np.zeros(Np)
    for i in range(Np):
        x1 = 0
        y1 = 0
        z1 = 0
        j = 0
        while j < Nt:
            r1 = step()
            if r1 != 0:
                x1 += r1[0]
                y1 += r1[1]
                z1 += r1[2]
                j += 1
                r1 = 0 # Not sure if this is necessary
        x[i] = x1
        y[i] = y1
        z[i] = z1
    for i in range(len(x)):
        print(x[i], y[i], z[i])

walk3(int(sys.argv[1]), int(sys.argv[2]))
