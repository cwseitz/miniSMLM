import json
import pandas as pd

with open('run_track_2d.json', 'r') as f:
    config = json.load(f)

prefixes_live = [
]


all_traj = []
for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    traj = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots-tracked.csv')
    all_traj.append(traj)
im_h2b_live = pd.concat(all_traj,axis=1)

prefixes_fixed = [
]

all_traj = []
for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    traj = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots-tracked.csv')
    all_traj.append(traj)
im_h2b_fixed = pd.concat(all_traj,axis=1)


fig, ax = plt.subplots()
avg_msd_h2b = np.mean(im_h2b.values,axis=1)
avg_msd_h2b = np.insert(avg_msd_h2b,0,0)
ax.plot(avg_msd_h2b,color='red',label='H2B')
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel='lag time $t$')
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()
