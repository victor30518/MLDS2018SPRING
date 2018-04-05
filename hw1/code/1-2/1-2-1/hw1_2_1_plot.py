import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.decomposition import PCA

plt.switch_backend('agg')
# 畫全部模型的圖
weights_matrix = np.load("./weights_matrix.npy")
# 轉置成正確方向
weights_matrix = weights_matrix.T

# 輸入有多少成份我們想要留住分解
pca = PCA(n_components=2,whiten=True)

# 將資料轉成兩個主成份
pca.fit(weights_matrix)
weights_pca = pca.transform(weights_matrix)

plt.figure(figsize=(12,8))

point_scale = 75
alpha = 0.1

plt.scatter(weights_pca[0:250,0], weights_pca[0:250,1], s=point_scale, c="b", alpha=alpha)
plt.scatter(weights_pca[250:500,0], weights_pca[250:500,1], s=point_scale, c="g", alpha=alpha)
plt.scatter(weights_pca[500:750,0], weights_pca[500:750,1], s=point_scale, c="r", alpha=alpha)
plt.scatter(weights_pca[750:1000,0], weights_pca[750:1000,1], s=point_scale, c="c", alpha=alpha)
plt.scatter(weights_pca[1000:1250,0], weights_pca[1000:1250,1], s=point_scale, c="m", alpha=alpha)
plt.scatter(weights_pca[1250:1500,0], weights_pca[1250:1500,1], s=point_scale, c="y", alpha=alpha)
plt.scatter(weights_pca[1500:1750,0], weights_pca[1500:1750,1], s=point_scale, c="k", alpha=alpha)
plt.scatter(weights_pca[1750:2000,0], weights_pca[1750:2000,1], s=point_scale, c='tab:brown', alpha=alpha)

plt.savefig("all_model_pca.png")



# 畫第一層的圖
weights_matrix = np.load("./weights_matrix.npy")
# 轉置成正確方向
weights_matrix = weights_matrix.T
one_layer_weights_matrix = weights_matrix[:,:10]

# 輸入有多少成份我們想要留住分解
pca = PCA(n_components=2,whiten=True)

# 將資料轉成兩個主成份
pca.fit(one_layer_weights_matrix)
one_layer_weights_pca = pca.transform(one_layer_weights_matrix)

plt.figure(figsize=(12,8))

point_scale = 75
alpha = 0.1

plt.scatter(one_layer_weights_pca[0:250,0], one_layer_weights_pca[0:250,1], s=point_scale, c="b", alpha=alpha)
plt.scatter(one_layer_weights_pca[250:500,0], one_layer_weights_pca[250:500,1], s=point_scale, c="g", alpha=alpha)
plt.scatter(one_layer_weights_pca[500:750,0], one_layer_weights_pca[500:750,1], s=point_scale, c="r", alpha=alpha)
plt.scatter(one_layer_weights_pca[750:1000,0], one_layer_weights_pca[750:1000,1], s=point_scale, c="c", alpha=alpha)
plt.scatter(one_layer_weights_pca[1000:1250,0], one_layer_weights_pca[1000:1250,1], s=point_scale, c="m", alpha=alpha)
plt.scatter(one_layer_weights_pca[1250:1500,0], one_layer_weights_pca[1250:1500,1], s=point_scale, c="y", alpha=alpha)
plt.scatter(one_layer_weights_pca[1500:1750,0], one_layer_weights_pca[1500:1750,1], s=point_scale, c="k", alpha=alpha)
plt.scatter(one_layer_weights_pca[1750:2000,0], one_layer_weights_pca[1750:2000,1], s=point_scale, c='tab:brown', alpha=alpha)

plt.savefig("one_layer_pca.png")