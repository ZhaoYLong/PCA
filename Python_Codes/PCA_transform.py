import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.mlab import PCA as mlabPCA   #matplotlib库内置的PCA()类

np.random.seed(2**32 -1)  # random seed for consistency


###计算均值和散步矩阵

mu_vec1 = np.array([0, 0, 0])   #第一类均值
cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])   #第一类协方差矩阵
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T    #根据均值、协方差矩阵生成20个向量组成3*20的矩阵
assert class1_sample.shape == (3, 20), "The matrix has not the dimensions 3x20"  #赋值矩阵大小

mu_vec2 = np.array([1, 1, 1])
cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
assert (3, 20) == class2_sample.shape, "The matrix has not the dimensions 3x20"


all_samples = np.concatenate((class1_sample, class2_sample), axis=1)  #合并两个矩阵
assert all_samples.shape == (3,40), "The matrix has not the dimensions 3x40"

###PCA TRANSFORM
mlab_pca = mlabPCA(all_samples.T)

print('PC axes in terms of the measurement axes scaled by the standard deviations:\n', mlab_pca.Wt)

plt.plot(mlab_pca.Y[0:20,0],mlab_pca.Y[0:20,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(mlab_pca.Y[20:40,0], mlab_pca.Y[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()
plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

plt.show()