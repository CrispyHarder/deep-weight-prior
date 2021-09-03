import numpy as np 
import os 
import matplotlib.pyplot as plt 

path_to_convs = os.path.join('data','fconv')
convs5_p = os.path.join(path_to_convs,'conv5x5','test.npy')
convs7_p = os.path.join(path_to_convs,'conv7x7','test.npy')

conv5 = np.load(convs5_p)
conv7 = np.load(convs7_p)

mean5_norms = np.array([np.linalg.norm(conv5[i]) for i in range(len(conv5))])
print(np.min(mean5_norms),np.max(mean5_norms))

mean7_norms = np.array([np.linalg.norm(conv7[i]) for i in range(len(conv7))])
print(np.min(mean7_norms),np.max(mean7_norms))

plt.hist(mean7_norms, np.arange(0,3,0.05))
plt.show()
