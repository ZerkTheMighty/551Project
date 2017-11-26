from pymf import pca
import numpy as np

print("TEST START")
data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
pca_mdl = pca.PCA(data, num_bases=2)
pca_mdl.factorize()
print(pca_mdl.W)
print(pca_mdl.H)
print("TEST END")
