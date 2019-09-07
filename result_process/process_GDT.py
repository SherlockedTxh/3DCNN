# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.io
import numpy as np

f = open("/home/a503tongxueheng/jupyter_project/T0864.txt")
infile = list(f)
GDT_dict = []
for i in range(2,len(infile)):
    line = infile[i].split()
    # print(line[3])
    if len(line)>0:
        GDT_dict.append([line[1],line[3]])
print(GDT_dict)
f.close()

