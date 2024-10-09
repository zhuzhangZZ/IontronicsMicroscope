import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,6.1*np.pi)

y = np.cos(x)
z = np.array([np.cos(x)+np.sin(a*x) for a in np.linspace(0,1)])

plt.plot(x,z)
plt.plot(x,y, linewidth=4)
plt.show()

#%%
ncyc = 3  # change to full_cycles for complete video
cyc = 2*np.pi

this_cycle = np.zeros(len(z))
off_cycle = np.zeros(len(z))

sync_average = np.zeros(len(z))
sync_w = y[int(0*cyc):int(0*cyc+cyc)]
async_average = np.zeros(len(z))
# qp = np.int(period/4)
# async_w = cell_potential[cyc2-qp:cyc2+period-qp+1]
for i in range(ncyc):
    for j in range(len(z)):
        cr = np.cov(z[j][int(i*cyc):int(i*cyc+cyc)], sync_w)
        this_cycle[j] = cr[0,1] / cr[1,1] 
    sync_average = sync_average + this_cycle/ncyc


plt.plot(x,z)
plt.plot(x,y, linewidth=4)
plt.show()

plt.plot(x,z[:][int(np.pi)])
plt.plot(x,sync_average)
plt.plot(x,y, linewidth=4)
plt.show()

#%%

thing = np.argmax(sync_average)

for thing in range(len(sync_average)):
    plt.plot(x,z[thing][:], label='%lf'%(sync_average[thing]))
    plt.plot(x,y, linewidth=4)
    plt.legend()
    plt.show()
    print("_")
    plt.pause(0.5)

#%%
    


a = [10,9,0]
b = [1,1,0]
cr = np.cov(a,b)
print(cr)
print(cr[0,1] / cr[1,1] )
print(cr[0,1] / cr[1,1] / np.mean(a))




    