from pymf import pca, cmeans, aa
import numpy as np

print("TEST START")
data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
pca_mdl = pca.PCA(data, num_bases=2)
pca_mdl.factorize()
print("PCA Results...")
print(pca_mdl.W)
print(pca_mdl.H)

data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
cmeans_mdl = cmeans.Cmeans(data, num_bases=2, niter=10)
cmeans_mdl.factorize()
print("Cmeans Results...")
print(cmeans_mdl.W)
print(cmeans_mdl.H)

data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
aa_mdl = aa.AA(data, num_bases=2)
aa_mdl.factorize(niter=5)

data = np.array([[1.5], [1.2]])
W = np.array([[1.0, 0.0], [0.0, 1.0]])
aa_mdl = aa.AA(data, num_bases=2)
aa_mdl.W = W
aa_mdl.factorize(niter=5, compute_w=False)
print("AA results...")
print(aa_mdl.W)
print(aa_mdl.H)
print("TEST END")
