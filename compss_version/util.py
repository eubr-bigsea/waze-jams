import numpy as np
np.set_printoptions(threshold=np.nan)
filename = '/home/lucasmsp/workspace/BigSea/waze-jams/compss_version/sample_train.txt'

ytab = np.loadtxt(filename, delimiter=' ')

acc = np.sum(ytab, axis=0)

print acc
print len(acc)

print max(acc)

max_grid = np.where(acc == acc.max())
print max_grid
print acc[max_grid]
