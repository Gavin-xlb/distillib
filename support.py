import numpy as np
import random


loadData = np.load('/data/xulingbing/projects/distillib/data/lits/tumor/slices/lits_tumor_slices_test.npy')
l = len(loadData)
print(len(loadData))

# 随机划分数据集(按病例划分)√
# kits
l_support = random.sample(range(0, l), 64)

support_sample = []
for data in l_support:
    support_sample.append(loadData[data])

print('train_sample', support_sample)
print(len(support_sample))
np.save('/data/xulingbing/projects/distillib/data/lits/tumor/slices/lits_tumor_slices_support.npy', support_sample)
