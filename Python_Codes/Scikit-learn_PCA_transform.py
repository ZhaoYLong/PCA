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

#####使用scikit-learn中的machine learning库～～PCA()class
#####Using the PCA() class from the sklearn.decomposition library to confirm our results

from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=2)
sklearn_transf = sklearn_pca.fit_transform(all_samples.T)

plt.plot(sklearn_transf[0:20,0],sklearn_transf[0:20,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(sklearn_transf[20:40,0], sklearn_transf[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()
plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

plt.show()

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##用scikit-learn pca()与step by step approach的比较
sklearn_transf = sklearn_transf * (-1)

# sklearn.decomposition.PCA
plt.plot(sklearn_transf[0:20,0],sklearn_transf[0:20,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(sklearn_transf[20:40,0], sklearn_transf[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()
plt.title('Transformed samples via sklearn.decomposition.PCA')
plt.show()

# step by step PCA
##代码分类使用文件不太连贯！下次将进行改进！

transformed = matrix_w.T.dot(all_samples)
plt.plot(transformed[0,0:20], transformed[1,0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(transformed[0,20:40], transformed[1,20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples step by step approach')
plt.show()