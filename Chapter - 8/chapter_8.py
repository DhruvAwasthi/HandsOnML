#!/usr/bin/env python
# coding: utf-8

# In[1]:


# With thousands or millions of features for each training instance, not only training becomes extremely slow but 
# it also makes it much harder to find a good solution. This is called Curse of Dimensionality. 
# High dimensional datasets are at risk of being sparse, leading to overfitting many times. So we need to reduce
# the dimension of training instances. The process of reducing high-dimensional data into a lower-dimensional data
# is called Dimensionality Reduction. It is useful in many cases:
# 1. To compress the data so it takes up less computer memory/disk space.
# 2. To reduce the dimensions of input data so as to speed up a learning algorithm.
# 3. TO visualize high-dimensional data.
# There are many techniques to Dimensionaly Reduction e.g., projection, etc.


# In[2]:


# PROJECTION:
# In most real-world problems, training instances are not spread out uniformly across all dimensions. Many features
# are almose constant, while others are highly correlated. As a result, all training instances actually lie within
# a much lower-dimensional subspace of the high-dimensional space. For e.g, while reducing 3D to 2D, we project 
# every training instance of 3D perpendicularly onto a subspace (i.e., plane) which results into a 2D dataset.
# However, projection is not always the best approach to dimensionality reduction as in many cases the subspace may
# twist and turn, such as in the famous Swiss roll toy dataset. Simply projecting onto a plane would squash 
# different layers of the Swiss roll together.


# In[3]:


# The Swiss roll is a example of a 2D manifold. A 2D manifold is a shape that can be bent and twisted in a higher 
# dimensional space. More generally, a d-dimensional manifold is a part of an n-dimensional space that locally 
# represents a d-dimensional hyperplane. In the case of swiss roll, d = 2 and n = 3.
# Many dimensionality reduction algorithms work by modeling the manifold on which the training instances lie; this 
# is called Manifold Learning. It relies on the manifold assumption, also called the manifold hypothesis, which 
# holds that most real-worlds high-dimensional datasets lie close to a much lower-dimensional manifold.


# In[4]:


# Principal Component Analysis (PCA) is the most popular dimensionality reduction algorithm. 
# The unit vector that defines the ith axis is called the ith principal component (PC), which are represented as
# c1, c2, c3, ... so on.
# BUILDING 3D DATASET
import numpy as np
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)


# In[5]:


# The following code uses NumPy's svd() function to obtain all the principal componets of the training set, then 
# extracts the first two PCs.
X_centered = X - X.mean(axis=0)
U, s, V = np.linalg.svd(X_centered)
c1 = V.T[:, 0]
c2 = V.T[:, 1]


# In[6]:


# Once you have identified all the principal components, you can reduce the dimensionality of the dataset down to 
# d dimensions by projecting it onto the hyperplane defined by the first d principal components. 
# The following code projects the training set onto the plane defined by the first two principal components:

W2 = V.T[:, :2]
X2D = X_centered.dot(W2)


# In[7]:


# Using Scikit-Learn's PCA class (which implements PCA using SVD decomposition) we can do this as:

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)


# In[8]:


# You can access the principal components using the components_ variable (note it contains the PCs as horizontal 
# vectors)
pca.components_


# In[9]:


# Explained variance ratio indicates the proportion of the dataset's variance that lies alon gthe axis of each 
# principal component. We can access this via explained_variance_ratio_ variable
pca.explained_variance_ratio_

# This tells you that 84.2% of the dataset's variance lies along the first axis, and 14.6% lies along the second 
# axis. This leaves less than 1.2% for the third axis, so it is reasonable to assume that it probably carries 
# little information.


# In[10]:


# Let's generate the X_train on MNIST dataset.
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.int64)

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[11]:


# Instead of arbitrarily choosing the number of dimensions to reduce down to, it is generally preferable to choose
# the dimensions that add up to a sufficiently large portion of the variance. 
# Following code computes PCA without reducing dimensionlity, then computes the minimum number of dimensions 
# required to preserve 95% of the training set's variance.

pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) +1

# You can then set n_components=d and run PCA again.


# In[12]:


# However, there is a much better option: instead of specifying the number of principal components you want to 
# preserve, you can set n_components to be a float between 0.0 and 1.0, indicating the ration of variance you wish
# to preserve:

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)


# In[13]:


X_reduced.shape
# Now after dimensionality reduction, the training set takes uo much less space, and each instance have just over 
# 150 features, instead of the original 784 features.


# In[14]:


# It is also possible to decompress the reduced dataset back to 784 dimensions by applying the inverse 
# transformation of the PCA projection. But it won't give back the original data, since the projection lost a bit 
# of information (within the 5% variance that was dropped), but it will likely be quite close to the original data.
# The mean squared distance between the original data and the reconstructed data (compressed and then decompressed)
# is called the reconstruction error.

pca = PCA(n_components = 154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)


# In[15]:


# One problem with the preceding implementation of PCA is that it requires the whole training set to fit in memory
# in order for the SVD algorithm. To overcome this, we can split the training set into mini-batches and feed an 
# Incremental PCA (IPCA) algorithm one mini-batch at a time. This is useful for large training sets, and also to 
# apply PCA online (i.e., on the fly, as new instances arrive).
# The following code splits the MNIST dataset into 100 mini-batches and feed them to Scikit-Learn's IncrementalPCA
# class to reduce the dimensionality of the MNIST dataset down to 154 dimensions. Now we must call partial_fir() 
# method instead of fit() method:

from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)
    
X_reduced = inc_pca.transform(X_train)


# In[16]:


# Alternatively, you can use NumPy's memmap class, which allows you to manipulate a large array stored in a binary
# file on disk as if it were entirely in memory; the class loads only the data it needs in memory, when it need it.
# Since the IncrementalPCA class uses only a small part of the array at any given time, the memory usage remains 
# under control
#
# X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
#
# batche_size = m // n_batches
# inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
# inc_pca.fit(X_mm)


# In[17]:


# Scikit-Learn offers another option to perform PCA, called Randomized PCA. This is a stochastic algorithm that 
# quickly finds as approximation of the first d principal components. Its computational complexity is:
# O(mXn^2) + O(d^3), instead of O(mXn^2). + O(n^3)

rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_train)


# In[18]:


# Kernel trick maps instances into a very high-dimensional space (called the featuere space), enabling nonlinear
# classification and regression with Support Vector Machines. The same trick can be applied to PCA, making it 
# possible to perform complex nonlinear projections for dimensionality reducitons. This is called Kernel PCA (kPCA)
# It is often good at preserving clusters of instances after projection, or sometimes even unrolling datasets that
# lie close to a twisted manifold.
#
# from sklearn.decomposition import KernelPCA
#
# rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
# X_reduced = rbf_pca.fit_transform(X)


# In[19]:


# As kPCA is an insupervised learning algorithm, there is no obvious performance measure to help you select the 
# best kernel and hyperparameter values. However, dimensionality reduction is often a preparation step for a 
# supervised learning task, so you can simply use grid search to select the kernel and hyperparameters that lead to
# the best performance on that task.

# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline

# clf = Pipeline([
#     ("kpca", KernelPCA(n_components=2)),
#     ("log_reg", LogisticRegression())
# ])

# param_grid = [{
#    "kpca__gamma": np.linspace(0.03, 0.05, 10),
#    "kpca__kernel": ["rbf", "sigmoid"]
# }]

# grid_search = GridSearchCV(clf, param_grid, cv=3)
# grid_search.fit(X, y)

# The best kernel and hyperparameters are then available throught the best_params_  variable:
# print(grid_search.best_params_)


# In[20]:


# Another approach, this time entirely unsupervised, is to select the kernel and hyperparameters that yield the 
# lowest reconstruction error. However, reconstruction is not as easy as with linear PCA.  
# For e.g., let's imagine the original Swiss roll 3D dataset and the resulting 2D dataset after kPCA is applied 
# using an RBF kernel. Thanks to the kernel trick, this is mathematically equivalent to mapping the training set
# to an infinite-dimensional feature space using the feature map, then projecting the transformed training set down
# to 2D using linear PCA. Notice that if we could invert the linear PCA step for a given instance in the reduced 
# space, the reconstructed point would lie in feature space, not in the original space. Since the feature space is
# infinite-dimensional, we cannot compute the reconstructed point, and therefore we cannot compute the true 
# reconstruction error.
# Fortunately it is possible to a point in the original space that would map close to the reconstructed point.
# This is called reconstruction pre-image. Once you have this pre-image, you can measure its squared distance to
# the original instance. You can then select the kernel and hyperparamters that minimize this reconstruction pre-
# image error.

# Now to perform this reconstruction, one solution is to train a supervised regression model, with the projected
# instances as the training set and the original instances as the targets. Scikit-Learn will do this automatically
# if you set fit_inverse_transform=True, as shown in following code:

# rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
# X_reduced = rbf_pca.fit_transform(X)
# X_preimage = rbf_pca.inverse_transform(X_reduced)

# You can then compute the reconstruction pre-image error:
# from sklearn.metrics import mean_squared_error
# mean_squared_error(X, X_preimage)
# evaluates to 32.786308795766132

# Now you can use grid search with cross-validation to find the kernel and hyperparameters that minimize this 
# pre-image reconstruction error.


# In[21]:


# Local Linear Embedding (LLE) is another very powerful nonlinear dimensionality reduction (NLDR) technique. It is 
# a Manifold Learning technique that does not rely on projections like the previous algorithms. 
# In a nutshell, LLE works by first measuring how each training instance linearly relates to its closest neighbors
# (c.n.), and then looking for a low-dimensional representation of the training set where these local relationships
# are best preserved. This makes it particularly good at unrolling manifolds, especially when there is not too much
# noise.

# The following code uses Scikit-Learn's LocallyLinearEmbedding class to unroll the Swiss roll.

# from sklearn.manifold import LocallyLinearEmbedding
#
# lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
# X_reduced = lle.fit_transform(X)


# In[22]:


# Let's load the MNIST dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.int64)


# In[23]:


# Take first 60,000 instances for training, and the remaining 10,000 for testing
X_train = mnist['data'][:60000]
y_train = mnist['target'][:60000]

X_test = mnist['data'][60000:]
y_test = mnist['target'][60000:]


# In[24]:


# Train a random forest classifier on the dataset
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)


# In[25]:


import time

t0 = time.time()
rnd_clf.fit(X_train, y_train)
t1 = time.time()


# In[26]:


print("Training took {:.2f}s".format(t1 - t0))


# In[27]:


from sklearn.metrics import accuracy_score

y_pred = rnd_clf.predict(X_test)
accuracy_score(y_test, y_pred)


# In[28]:


# Let's use PCA to reduce the dataset's dimensionality, with an explained variance ratio of 95%
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)


# In[29]:


# Now let's train a new radom forest classifier on the reduced dataset.

rnd_clf2 = RandomForestClassifier(n_estimators=10, random_state=42)
t0 = time.time()
rnd_clf2.fit(X_train_reduced, y_train)
t1 = time.time()
print("Training took {:.2f}s".format(t1 - t0))

# Oh no! Training is actually more than twice slower now! How can that be? Well, as we saw in this chapter, 
# dimensionality reduction does not always lead to faster training time: it depends on the dataset, the model and 
# the training algorithm.


# In[30]:


# Let's evaluate the classifier on the test set

X_test_reduced = pca.transform(X_test)

y_pred = rnd_clf2.predict(X_test_reduced)
accuracy_score(y_test, y_pred)

# It is common for performance to drop slightly when reducing dimensionality, because we do lose some useful 
# signal in the process.


# In[31]:


# Now let's see if softmax regression helps
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
t0 = time.time()
log_clf.fit(X_train, y_train)
t1 = time.time()


# In[32]:


print("Training took {:.2f}s".format(t1 - t0))


# In[33]:


# Evaluating scores
y_pred = log_clf.predict(X_test)
accuracy_score(y_test, y_pred)


# In[34]:


# Okay, so softmax regression takes much longer to train on this dataset than the random forest classifier, plus 
# it performs worse on the test set. But that's not what we are interested in right now, we want to see how much
# PCA can help softmax regression. Let's train the softmax regression model using the reduced dataset:

log_clf2 = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
t0 = time.time()
log_clf2.fit(X_train_reduced, y_train)
t1 = time.time()


# In[35]:


print("Training took {:.2f}s".format(t1 - t0))

# Nice! Reducing dimensionality led to a 3× speedup


# In[36]:


# Now let's evaluate the accuracy score:

y_pred = log_clf2.predict(X_test_reduced)
accuracy_score(y_test, y_pred)

