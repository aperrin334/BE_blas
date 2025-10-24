import matplotlib.pyplot as plt
import csv
import numpy as np

data1 = np.genfromtxt('timings_BLAS3_1.csv', delimiter='		', skip_header = 1)
data2 = np.genfromtxt('timings_BLAS3_2.csv', delimiter='		', skip_header = 1)
data3 = np.genfromtxt('timings_BLAS3_3.csv', delimiter='		', skip_header = 1)
data4 = np.genfromtxt('timings_BLAS3_4.csv', delimiter='		', skip_header = 1)
data5 = np.genfromtxt('timings_BLAS3_5.csv', delimiter='		', skip_header = 1)
data6 = np.genfromtxt('timings_BLAS3_6.csv', delimiter='		', skip_header = 1)
data7 = np.genfromtxt('timings_BLAS3_7.csv', delimiter='		', skip_header = 1)
data8 = np.genfromtxt('timings_BLAS3_8.csv', delimiter='		', skip_header = 1)
data9 = np.genfromtxt('timings_BLAS3_9.csv', delimiter='		', skip_header = 1)
data10 = np.genfromtxt('timings_BLAS3_10.csv', delimiter='		', skip_header = 1)
data11 = np.genfromtxt('timings_BLAS3_11.csv', delimiter='		', skip_header = 1)
data12 = np.genfromtxt('timings_BLAS3_12.csv', delimiter='		', skip_header = 1)

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

x5 = data5[:,0]
y5 = data5[:,1]
z5 = data5[:,2]

x6 = data6[:,0]
y6 = data6[:,1]
z6 = data6[:,2]

x7 = data7[:,0]
y7 = data7[:,1]
z7 = data7[:,2]

x8 = data8[:,0]
y8 = data8[:,1]
z8 = data8[:,2]

x9 = data9[:,0]
y9 = data9[:,1]
z9 = data9[:,2]

x10 = data10[:,0]
y10 = data10[:,1]
z10 = data10[:,2]

x11 = data11[:,0]
y11 = data11[:,1]
z11 = data11[:,2]

x12 = data12[:,0]
y12 = data12[:,1]
z12 = data12[:,2]




#Figure USE_ADD
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Execution time
ax1.plot(x1,y1, label='1 THREAD')
ax1.plot(x2,y2,label='2 THREAD')
ax1.plot(x3,y3, label='3 THREAD')
ax1.plot(x4,y4,label='4 THREAD')
ax1.plot(x5,y5, label='5 THREAD')
ax1.plot(x6,y6,label='6 THREAD')
ax1.plot(x7,y7, label='7 THREAD')
ax1.plot(x8,y8,label='8 THREAD')
ax1.plot(x9,y9, label='9 THREAD')
ax1.plot(x10,y10,label='10 THREAD')
ax1.plot(x11,y11, label='11 THREAD')
ax1.plot(x12,y12,label='12 THREAD')
ax1.set_title("Evolution du temps moyen d'execution")
ax1.set_xlabel("Dimension")
ax1.set_ylabel("time (s)")

# Calculation speed
ax2.plot(x1,z1, label='1 THREAD')
ax2.plot(x2,z2,label='2 THREAD')
ax2.plot(x3,z3, label='3 THREAD')
ax2.plot(x4,z4,label='4 THREAD')
ax2.plot(x5,z5, label='5 THREAD')
ax2.plot(x6,z6,label='6 THREAD')
ax2.plot(x7,z7, label='7 THREAD')
ax2.plot(x8,z8,label='8 THREAD')
ax2.plot(x9,z9, label='9 THREAD')
ax2.plot(x10,z10,label='10 THREAD')
ax2.plot(x11,z11, label='11 THREAD')
ax2.plot(x12,z12,label='12 THREAD')
ax2.set_title("Evolution de la rapidité de calcul (Gflop/S)")
ax2.set_xlabel("Dimension")
ax2.set_ylabel("Gflop/S")
plt.legend()
plt.tight_layout()
plt.show()