import numpy as np
import random
# path1 = '/data/xulingbing/projects/distillib/data/kits/organ/test/kits_organ_slices.npy'
# path2 = '/data/xulingbing/projects/distillib/data/kits/organ/train/kits_organ_slices.npy'
# a = np.load(path1)
# b = np.load(path2)
# print(a)
# print(b)
# c = []
# for i in b:
#     c.append(i)
# for j in a:
#     c.append(j)
# # c = np.append(a, b)
# # print(c)
# np.save('/data/xulingbing/projects/distillib/data/kits/organ/slices/kits_organ_slices.npy', c)

loadData = np.load('/data/xulingbing/projects/distillib/data/lits/organ/slices/lits_organ_slices_train.npy')
print(len(loadData))

# loadData = np.load('/data/xulingbing/projects/distillib/data/kits/tumor/slices/kits_tumor_slices_train.npy')
# print(len(loadData))

# 随机划分数据集(按病例划分)√
# kits
# l_train = random.sample(range(0, 210), 168)
# l_test = list(set(list(range(210))) - set(l_train))
# for i in range(len(l_train)):
#     lenth = len(str(l_train[i]))
#     s = '0' * (5 - lenth) + str(l_train[i])
#     l_train[i] = s

# for i in range(len(l_test)):
#     lenth = len(str(l_test[i]))
#     s = '0' * (5 - lenth) + str(l_test[i])
#     l_test[i] = s

# lits
l_train = random.sample(range(0, 131), 105)
l_test = list(set(list(range(131))) - set(l_train))

# print('l_train', l_train)
# print('l_test', l_test)
# print(set(l_train)&set(l_test))

train_sample = []
test_sample = []
for data in loadData:
    case = data.split('_')[0]
    if int(case) in l_train:
        train_sample.append(data)
    else:
        test_sample.append(data)
# print('train_sample', train_sample)
# print('test_sample', test_sample)
np.save('/data/xulingbing/projects/distillib/data/lits/organ/slices/lits_organ_slices_train.npy', train_sample)
np.save('/data/xulingbing/projects/distillib/data/lits/organ/slices/lits_organ_slices_test.npy', test_sample)


# 另一种划分方式(按切片划分)x
# rng = np.random.default_rng(42)
# loadData = np.load('/data/xulingbing/projects/distillib/data/kits/tumor/slices/kits_tumor_slices.npy')
# N = len(loadData)
# p = rng.permutation(N)
# i = int(np.floor(0.7 * N))
# print(N)
# print(i)
# print(p)
# train_sample = [loadData[i] for i in p[:i]]
# test_sample = [loadData[i] for i in p[i:]]
# print(len(train_sample))
# print(len(test_sample))
# np.save('/data/xulingbing/projects/distillib/data/kits/tumor/slices/kits_tumor_slices_train.npy', train_sample)
# np.save('/data/xulingbing/projects/distillib/data/kits/tumor/slices/kits_tumor_slices_test.npy', test_sample)

# loadData2 = np.load('/data/xulingbing/projects/distillib/data/kits/tumor/slices/kits_tumor_slices_train.npy')
# print(len(loadData2))
# s2 = set()  # 获取所有的训练case
# for i in loadData2:
#     s2.add(i[:5])
# print(sorted(list(s2)))
#
# train_sample = []
# test_sample = []
# for data in loadData:
#     case = data.split('_')[0]
#     if case in s2:
#         train_sample.append(data)
#     else:
#         test_sample.append(data)
# print('train_sample', len(train_sample))
# print('test_sample', len(test_sample))
# np.save('/data/xulingbing/projects/distillib/data/kits/organ/slices/kits_organ_slices_train.npy', train_sample)
# np.save('/data/xulingbing/projects/distillib/data/kits/organ/slices/kits_organ_slices_test.npy', test_sample)