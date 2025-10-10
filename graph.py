import matplotlib.pyplot as plt
import csv
import numpy as np


data1 = np.genfromtxt('timings_add.csv', delimiter='		', skip_header = 1)
data2 = np.genfromtxt('timings_add-inorder.csv', delimiter='		', skip_header = 1)

x1 = data1[:,0]
y1 = data1[:,1]
z1 = data1[:,2]

x2 = data2[:,0]
y2= data2[:,1]
z2 = data2[:,2]

plt.plot(x1,y1)
plt.plot(x2,y2)
plt.show()
plt.plot(x1,z1)
plt.plot(x2,z2)
plt.show()