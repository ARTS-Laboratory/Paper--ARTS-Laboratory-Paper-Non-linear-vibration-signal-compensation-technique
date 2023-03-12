import numpy as np
"""

"""
X = np.loadtxt("./model_predictions/0-10 Hz Test.csv", delimiter=',')

package = X[:,0:1]
ref = X[:,1:2]
model = X[:,2:]

t = np.array([i/400 for i in range(package.size)]).reshape(-1, 1)

Y = np.append(t, ref, -1)
Y = np.append(Y, package, -1)
Y = np.append(Y, model, -1)

np.savetxt("c.csv", Y, delimiter=',')