import matplotlib.pyplot as plt
import csv
import numpy as np

data1 = np.genfromtxt('timings_add.csv', delimiter='		', skip_header = 1)
data2 = np.genfromtxt('timings_add-inorder.csv', delimiter='		', skip_header = 1)
datablas1 = np.genfromtxt('timings_BLAS1.csv', delimiter='		', skip_header = 1)
datablas2 = np.genfromtxt('timings_BLAS2.csv', delimiter='		', skip_header = 1)
datablas3 = np.genfromtxt('timings_BLAS3.csv', delimiter='		', skip_header = 1)
datablas3ob = np.genfromtxt('timings_BLAS3_openblas.csv', delimiter='		', skip_header = 1)

x1 = data1[:,0]
y1 = data1[:,1]
z1 = data1[:,2]

x2 = data2[:,0]
y2= data2[:,1]
z2 = data2[:,2]

xblas1 = datablas1[:,0]
yblas1= datablas1[:,1]
zblas1 = datablas1[:,2]

xblas2 = datablas2[:,0]
yblas2= datablas2[:,1]
zblas2 = datablas2[:,2]

xblas3 = datablas3[:,0]
yblas3 = datablas3[:,1]
zblas3 = datablas3[:,2]

xblas3ob = datablas3ob[:,0]
yblas3ob = datablas3ob[:,1]
zblas3ob = datablas3ob[:,2]

#Figure USE_ADD
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Execution time
ax1.plot(x1,y1, label='inhous_add')
ax1.plot(x2, y2,label='inhouse_add_reorder')
ax1.set_title("Evolution du temps moyen d'execution USE_ADD")
ax1.set_xlabel("Dimension")
ax1.set_ylabel("time (s)")

# Calculation speed
ax2.plot(x1, z1, label='inhous_add')
ax2.plot(x2, z2, label='inhous_add_reorder')
ax2.set_title("Evolution de la rapidité de calcul (Gflop/S) USE_ADD")
ax2.set_xlabel("Dimension")
ax2.set_ylabel("Gflops/s")
plt.tight_layout()
plt.show()

#Figure BLAS1,2,3 sans optimisation
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

ax3.plot(xblas1,yblas1, color='gray', label='BLAS1')
ax3.plot(xblas2,yblas2,color='g', label='BLAS2')
ax3.plot(xblas3,yblas3,color='r', label='BLAS3')
ax3.set_title("Evolution du temps moyen d'execution")
ax3.set_xlabel("Dimension")
ax3.set_ylabel("time (s)")

ax4.plot(xblas1,zblas1,color='gray', label='BLAS1')
ax4.plot(xblas2,zblas2,color='g', label='BLAS2')
ax4.plot(xblas3,zblas3,color='r', label='BLAS3')
ax4.set_title("Evolution de la rapidité de calcul (Gflop/S)")
ax4.set_xlabel("Dimension")
ax4.set_ylabel("Gflops/s")
plt.tight_layout()
plt.show()


