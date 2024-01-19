
import torch
import numpy as np 
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from copy import deepcopy

# 初始化数据
X = torch.randn(100, 2)

# 生成数据集
# X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
# X = torch.from_numpy(X).float()
def kmeans_pp(X, K, max_iters=100):
    # if input is numpy array, convert to torch tensor
    if type(X) == np.ndarray:
        X = torch.from_numpy(X).float()
    # 从数据集中随机选择一个中心点
    centers = X[torch.randint(X.size(0), (1,))]
    # 选择其他的中心点
    for i in range(K - 1):
        # 计算每个点与已有中心点的最小距离
        distances = torch.cdist(X, centers).min(dim=1).values + 1e-6
        # 通过距离的概率分布选择下一个中心点
        probs = distances ** 2 / torch.sum(distances ** 2)
        idx = torch.multinomial(probs, num_samples=1)
        centers = torch.cat([centers, X[idx]], dim=0)
    # 迭代更新簇分配和中心点
    for i in range(max_iters):
        # 计算每个点到每个中心点的距离
        distances = torch.cdist(X, centers)
        # 分配簇
        cluster_assignment = torch.argmin(distances, dim=1)
        # 计算新的中心点
        new_centers2 = torch.stack([torch.mean(X[cluster_assignment == k], dim=0)
                                   for k in range(K)], dim=0)
        # 上面这行代码有bug，当某个簇没有点分配到时，会报错
        # 为了避免这个问题，我们可以使用下面的代码
        new_centers = []
        for k in range(K):
            if torch.sum(cluster_assignment == k) > 0:
                new_centers.append(torch.mean(X[cluster_assignment == k], dim=0))
            else:
                new_centers.append(centers[k])
        new_centers = torch.stack(new_centers, dim=0)
        # assert torch.equal(new_centers, new_centers2)
        # 判断中心点是否变化不大，若不大则退出迭代
        # is nan?
        assert not torch.isnan(new_centers).any()

        if torch.all(torch.eq(new_centers, centers)):
            break
        centers = deepcopy(new_centers)
    return centers, cluster_assignment

# # 运行 K-Means++ 算法
# K = 20
# centers, cluster_assignment = kmeans_pp(X, K)
#
# # 可视化聚类结果
# # colors = ['r', 'g', 'b']
# # for k in range(K):
# #     plt.scatter(X[cluster_assignment == k, 0], X[cluster_assignment == k, 1], c=colors[k])
# # plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, c='black')
# # # save figure
# # plt.savefig('mz_project/kmeans_pp.png')
#
# # k>3
# colors = np.random.rand(K, 3)
# for k in range(K):
#     plt.scatter(X[cluster_assignment == k, 0], X[cluster_assignment == k, 1], c=colors[k])
# plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, c='black')
# # save figure
# plt.savefig('mz_project/kmeans_pp_new.png')