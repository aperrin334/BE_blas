import matplotlib.pyplot as plt
import csv
import numpy as np

data1 = np.genfromtxt('block_256.csv', delimiter='		', skip_header = 1)
data2 = np.genfromtxt('block_512.csv', delimiter='		', skip_header = 1)
data3 = np.genfromtxt('block_1024.csv', delimiter='		', skip_header = 1)
data4 = np.genfromtxt('block_2048.csv', delimiter='		', skip_header = 1)


x1 = data1[:,0]
y1 = data1[:,1]
z1 = data1[:,2]

x2 = data2[:,0]
y2 = data2[:,1]
z2 = data2[:,2]

x3 = data3[:,0]
y3 = data3[:,1]
z3 = data3[:,2]

x4 = data4[:,0]
y4 = data4[:,1]
z4 = data4[:,2]





#Figure USE_ADD
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Execution time
ax1.plot(x1,y1, label='256 block')
ax1.plot(x2,y2,label='512 block')
ax1.plot(x3,y3, label='1024 block')
#ax1.plot(x4,y4,label='2048 block')
ax1.set_title("Evolution du temps moyen d'execution")
ax1.set_xlabel("Nombre de block")
ax1.set_ylabel("time (s)")

# Calculation speed
ax2.plot(x1,z1, label='256 block')
ax2.plot(x2,z2,label='512 block')
ax2.plot(x3,z3, label='1024 block')
ax2.plot(x4,z4,label='2048 block')
ax2.set_title("Evolution de la rapidité de calcul (Gflop/S)")
ax2.set_xlabel("Nombre de block")
ax2.set_ylabel("Gflop/S")
plt.legend()
plt.tight_layout()
plt.show()