import matplotlib.pyplot as plt
import csv
import numpy as np

data1 = np.genfromtxt('timings_add.csv', delimiter='		', skip_header = 1)
data2 = np.genfromtxt('timings_add-inorder.csv', delimiter='		', skip_header = 1)
datablas1 = np.genfromtxt('timings_BLAS1.csv', delimiter='		', skip_header = 1)
datablas2 = np.genfromtxt('timings_BLAS2.csv', delimiter='		', skip_header = 1)
datablas3 = np.genfromtxt('timings_BLAS3.csv', delimiter='		', skip_header = 1)
datablas1ob = np.genfromtxt('timings_BLAS1_openblas.csv', delimiter='		', skip_header = 1)
datablas2ob = np.genfromtxt('timings_BLAS2_openblas.csv', delimiter='		', skip_header = 1)
datablas3ob = np.genfromtxt('timings_BLAS3_openblas.csv', delimiter='		', skip_header = 1)
datamuld2 = np.genfromtxt('timings_para_d2.csv', delimiter='		', skip_header = 1)
datamuld4 = np.genfromtxt('timings_para_d4.csv', delimiter='		', skip_header = 1)
datamuld6 = np.genfromtxt('timings_para_d6.csv', delimiter='		', skip_header = 1)
datamuld8 = np.genfromtxt('timings_para_d8.csv', delimiter='		', skip_header = 1)
datamuld10 = np.genfromtxt('timings_para_d10.csv', delimiter='		', skip_header = 1)
datamuld12 = np.genfromtxt('timings_para_d12.csv', delimiter='		', skip_header = 1)
datamuls2 = np.genfromtxt('timings_para_s2.csv', delimiter='		', skip_header = 1)
datamuls4 = np.genfromtxt('timings_para_s4.csv', delimiter='		', skip_header = 1)
datamuls6 = np.genfromtxt('timings_para_s6.csv', delimiter='		', skip_header = 1)
datamuls8 = np.genfromtxt('timings_para_s8.csv', delimiter='		', skip_header = 1)
datamuls10 = np.genfromtxt('timings_para_s10.csv', delimiter='		', skip_header = 1)
datamuls12 = np.genfromtxt('timings_para_s12.csv', delimiter='		', skip_header = 1)
datablas32112 = np.genfromtxt('timings_BLAS3_2112.csv', delimiter='		', skip_header = 1)
datamulfunroll = np.genfromtxt('timings_mul_funroll.csv', delimiter='		', skip_header = 1)
datamulBfunroll = np.genfromtxt('timings_mul_blocking_funroll_32.csv', delimiter='		', skip_header = 1)



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

xblas1ob = datablas1ob[:,0]
yblas1ob = datablas1ob[:,1]
zblas1ob = datablas1ob[:,2]

xblas2ob = datablas2ob[:,0]
yblas2ob = datablas2ob[:,1]
zblas2ob = datablas2ob[:,2]

xblas3ob = datablas3ob[:,0]
yblas3ob = datablas3ob[:,1]
zblas3ob = datablas3ob[:,2]

xd2 = datamuld2[:,0]
yd2 = datamuld2[:,1]
zd2 = datamuld2[:,2]

xd4 = datamuld4[:,0]
yd4 = datamuld4[:,1]
zd4 = datamuld4[:,2]

xd6 = datamuld6[:,0]
yd6 = datamuld6[:,1]
zd6 = datamuld6[:,2]

xd8 = datamuld8[:,0]
yd8 = datamuld8[:,1]
zd8 = datamuld8[:,2]

xd10 = datamuld10[:,0]
yd10 = datamuld10[:,1]
zd10 = datamuld10[:,2]

xd12 = datamuld12[:,0]
yd12 = datamuld12[:,1]
zd12 = datamuld12[:,2]

xs2 = datamuls2[:,0]
ys2 = datamuls2[:,1]
zs2 = datamuls2[:,2]

xs4 = datamuls4[:,0]
ys4 = datamuls4[:,1]
zs4 = datamuls4[:,2]

xs6 = datamuls6[:,0]
ys6 = datamuls6[:,1]
zs6 = datamuls6[:,2]

xs8 = datamuls8[:,0]
ys8 = datamuls8[:,1]
zs8 = datamuls8[:,2]

xs10 = datamuls10[:,0]
ys10 = datamuls10[:,1]
zs10 = datamuls10[:,2]

xs12 = datamuls12[:,0]
ys12 = datamuls12[:,1]
zs12 = datamuls12[:,2]

xblas32112 = datablas32112[:,0]
yblas32112 = datablas32112[:,1]
zblas32112 = datablas32112[:,2]

xmulfunroll= datamulfunroll[:,0]
ymulfunroll = datamulfunroll[:,1]
zmulfunroll = datamulfunroll[:,2]

xmulfunrollB= datamulBfunroll[:,0]
ymulfunrollB = datamulBfunroll[:,1]
zmulfunrollB = datamulBfunroll[:,2]


#Figure USE_ADD
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Execution time
ax1.plot(x1,y1, label='inhouse_add')
ax1.plot(x2, y2,label='inhouse_add_reorder')
ax1.set_title("Evolution du temps moyen d'execution USE_ADD")
ax1.set_xlabel("Dimension")
ax1.set_ylabel("time (s)")
plt.legend()

# Calculation speed
ax2.plot(x1, z1, label='inhouse_add')
ax2.plot(x2, z2, label='inhouse_add_reorder')
ax2.set_title("Evolution de la rapidité de calcul (Gflop/S) USE_ADD")
ax2.set_xlabel("Dimension")
ax2.set_ylabel("Gflops/s")
plt.legend()
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
plt.legend()
plt.tight_layout()
plt.show()

#Figure BLAS1,2,3 avec optimisation openblas
fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))

ax5.plot(xblas1ob,yblas1ob, color='gray', label='BLAS1')
ax5.plot(xblas2ob,yblas2ob,color='g', label='BLAS2')
ax5.plot(xblas3ob,yblas3ob,color='r', label='BLAS3')
ax5.set_title("Evolution du temps moyen d'execution")
ax5.set_xlabel("Dimension")
ax5.set_ylabel("time (s)")

ax6.plot(xblas1ob,zblas1ob,color='gray', label='BLAS1')
ax6.plot(xblas2ob,zblas2ob,color='g', label='BLAS2')
ax6.plot(xblas3ob,zblas3ob,color='r', label='BLAS3')
ax6.set_title("Evolution de la rapidité de calcul (Gflop/S)")
ax6.set_xlabel("Dimension")
ax6.set_ylabel("Gflops/s")
plt.legend()
plt.tight_layout()
plt.show()

#Figure MUL avec optimisation parallelisation dynamique
fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(12, 5))

ax7.plot(xd2,yd2)
ax7.plot(xd4,yd4)
ax7.plot(xd6,yd6)
ax7.plot(xd8,yd8)
ax7.plot(xd10,yd10)
ax7.plot(xd12,yd12)
ax7.set_title("Evolution du temps moyen d'execution")
ax7.set_xlabel("Dimension")
ax7.set_ylabel("time (s)")

ax8.plot(xd2,zd2)
ax8.plot(xd4,zd4)
ax8.plot(xd6,zd6)
ax8.plot(xd8,zd8)
ax8.plot(xd10,zd10)
ax8.plot(xd12,zd12)
ax8.set_title("Evolution de la rapidité de calcul (Gflop/S)")
ax8.set_xlabel("Dimension")
ax8.set_ylabel("Gflops/s")
plt.tight_layout()
plt.show()

#Figure MUL avec optimisation parallelisation static
fig2, (ax9, ax10) = plt.subplots(1, 2, figsize=(12, 5))

ax9.plot(xs2,ys2)
ax9.plot(xs4,ys4)
ax9.plot(xs6,ys6)
ax9.plot(xs8,ys8)
ax9.plot(xs10,ys10)
ax9.plot(xs12,ys12)
ax9.set_title("Evolution du temps moyen d'execution")
ax9.set_xlabel("Dimension")
ax9.set_ylabel("time (s)")

ax10.plot(xs2,zs2)
ax10.plot(xs4,zs4)
ax10.plot(xs6,zs6)
ax10.plot(xs8,zs8)
ax10.plot(xs10,zs10)
ax10.plot(xs12,zs12)
ax10.set_title("Evolution de la rapidité de calcul (Gflop/S)")
ax10.set_xlabel("Dimension")
ax10.set_ylabel("Gflops/s")
plt.tight_layout()
plt.show()

#Figure BLAS3, MUL_inhouse, MUL_blocking
fig2, (ax11, ax12) = plt.subplots(1, 2, figsize=(12, 5))

ax11.plot(xblas32112,yblas32112)
ax11.plot(xmulfunroll,ymulfunroll)
ax11.plot(xmulfunrollB,ymulfunrollB)
ax11.set_title("Evolution du temps moyen d'execution")
ax11.set_xlabel("Dimension")
ax11.set_ylabel("time (s)")

ax12.plot(xblas32112,zblas32112, label ='BLAS3')
ax12.plot(xmulfunroll,zmulfunroll, label='mul_parallel')
ax12.plot(xmulfunrollB,zmulfunrollB, label='mul_blocking taille 32')
ax12.set_title("Evolution de la rapidité de calcul (Gflop/S)")
ax12.set_xlabel("Dimension")
ax12.set_ylabel("Gflops/s")
plt.legend()
plt.tight_layout()
plt.show()