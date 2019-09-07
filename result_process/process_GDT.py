# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.io
import numpy as np

x = []
y1 = []
y2 = []

f = open("/home/a503tongxueheng/jupyter_project/T0864.txt")
softmaxfile = open("/home/a503tongxueheng/SoftmaxResults/result.txt")
infile = list(f)
infile_softmax = list(softmaxfile)
Softmax_dict = []

for line in infile_softmax:
    line_data = line.split()
    Softmax_dict.append(line_data[0][:-1],line_data[1])

for i in range(2,len(infile)):
    line = infile[i].split()
    # print(line[3])
    if len(line)>0:
        x.append(line[1])
        y1.append(float(line[3]))
        for data in Softmax_dict:
            if data[0] == line[1]:
                y2.append(100*float(data[1]))
f.close()

plt.plot(x,y1,label='GDT_Score')
plt.plot(x,y2,label='3DCNN_Score')

plt.tick_params(labelsize=1)
plt.legend(loc="upper right")  #set legend location
plt.ylabel('GDT Score/3DCNN Score')   # set ystick label
plt.xlabel('T0864')  # set xstck label
pylab.xticks(rotation=90) 

plt.savefig('snapshots.eps',dpi = 1000,bbox_inches='tight')
plt.show()