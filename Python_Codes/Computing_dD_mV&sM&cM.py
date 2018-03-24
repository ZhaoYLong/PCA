import numpy as np

np.random.seed(2**32 -1)  # random seed for consistency


###计算均值和散布矩阵

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

##计算均值    Computing the d-dimensional mean vector
mean_x = np.mean(all_samples[0,:])
mean_y = np.mean(all_samples[1,:])
mean_z = np.mean(all_samples[2,:])

mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

print('Mean Vector:\n', mean_vector)

###计算散布矩阵   Computing the Scatter Matrix
scatter_matrix = np.zeros((3,3))
for i in range(all_samples.shape[1]):
    scatter_matrix += (all_samples[:,i].reshape(3,1) - mean_vector).dot((all_samples[:,i].reshape(3,1) - mean_vector).T)
print('Scatter Matrix:\n', scatter_matrix)

###就算协方差矩阵   Computing the Covariance Matrix
cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])
print('Covariance Matrix:\n', cov_mat)

