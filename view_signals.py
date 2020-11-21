####### libraries
import numpy as np
import matplotlib.pyplot as plt

####### graphix stuff
font = {'weight' : 'bold',
        'size'   : 30}
plt.rc('font', **font)


######## read data
### signal #1
t1=np.load('signal_repo/time1.npy')
R1=np.load('signal_repo/R1.npy')
t1=t1-t1[0]

### signal #2
t2=np.load('signal_repo/time2.npy')
R2=np.load('signal_repo/R2.npy')
t2=t2-t2[0]



#### plot data
h1=plt.figure(figsize=(60,25))
ax1=h1.add_subplot(111)

ax1.plot(t1,R1,linewidth=3)
ax1.plot(t2,R2,linewidth=3)



### save figure
h1.savefig('all_signals.png')