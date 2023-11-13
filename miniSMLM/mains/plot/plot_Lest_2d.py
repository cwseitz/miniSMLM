import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

data = pd.read_csv('AMPKa/combined_data.csv')
#data_control = pd.read_csv('Control/combined_data.csv')
r = data['r']
data = data.drop(columns='r')
data = data.values
avg = np.mean(data,axis=1)
std = np.std(data,axis=1)

fig,ax=plt.subplots()

ax.plot(r,avg,color='black',label='AMPKa')
ax.fill_between(r, avg-std,avg+std,color='black',alpha=0.2)

ax.set_xlabel('r (nm)',size=12)
ax.set_ylabel('L(r)-r',size=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()
plt.show()

