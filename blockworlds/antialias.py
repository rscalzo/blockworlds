#!/usr/bin/env python

"""
RS 2020/07/01:  Antialiasing in Blockworld Models

I'm curious about fast emulators for anti-aliasing, so I'll try to learn some
kind of anti-aliasing model here and see how I go.
"""

import numpy as np
import matplotlib.pyplot as plt
from discretize import TensorMesh
from sklearn import gaussian_process as GP
from blockworlds import profile_timer

Neval = 20
hx = hy = hz = [(1.0 / Neval, Neval), ]
mesh_eval = TensorMesh([hx, hy, hz], "CCC")
# Generate an experimental design, one that catches edges and corners
Nplan = 4
hx = hy = hz = [(1.0 / Nplan, Nplan), ]
mesh_plan = TensorMesh([hx, hy, hz], "CCC")

def partial_volume(mesh, r0, n):
    """
    Calculate the partial volume inside a unit cube bounded by the plane
        np.dot(r-r0, n) < 0
    :param r0: np.array of shape (3,) with components (x0, y0, z0)
    :param n: np.array of shape (3,) with components (nx, ny, nz)
    :return: fractional volume (between 0 and 1)
    """
    mu_prod = np.dot(mesh.gridCC - r0, n.T)
    b1 = np.mean(mu_prod <  0, axis=0)
    b2 = np.mean(mu_prod <= 0, axis=0)
    return 0.5*(b1+b2)

def generate_test_data():
    """
    Generate a few specific test data instances for an anti-aliasing model
    :return pars: np.array of shape (N, 6), each row (rx, ry, rz, nx, ny, nz)
    :return results: np.array of shape (N, ), each row a partial volume
    """
    # Parameters are (rx, ry, rz, nx, ny, nz)
    pars = np.array([[0.0,  0.0,  0.0,  0.0000,  0.0000,  1.0000],
                     [0.0,  0.0,  0.0,  0.5773,  0.5773,  0.5773],
                     [0.0,  0.0,  0.2,  0.0000,  0.0000,  1.0000],
                     [0.0,  0.0, -0.2,  0.0000,  0.0000,  1.0000],
                     [0.0,  0.0, -0.2,  0.0000,  0.0000, -1.0000],
                     [0.5,  0.5,  0.5, -0.7071,  0.7071,  0.0000]])
    pV = np.array([0.5, 0.5, 0.7, 0.3, 0.7, 0.5])
    return pars, pV

def generate_unit_vectors(N, uniform_omega=True):
    """
    Generate random directions on the 2-sphere
    :param N: number of training instances to return
    :param uniform_omega: distribute (nx, ny, nz) uniformly in solid angle?
    :return: np.array of shape (N, 3)
    """
    if uniform_omega:
        # This way of doing it will distribute them uniformly in solid angle,
        # which is the statistically unbiased way of doing things
        # dA = cos(theta)*dtheta*dphi, theta = 0 at the equator
        # z = sin(theta) -> dA = -dz
        nz = np.random.uniform(size=(N,)) - 0.5
        phi = 2*np.pi*np.random.uniform(size=(N,))
        nx, ny = np.sin(phi), np.cos(phi)
        n = np.vstack([[nx],[ny],[nz]]).T
    else:
        # This way of doing it will concentrate them at the cube's corners,
        # which might be useful for some training methods
        n = np.random.uniform(size=(N,3)) - 0.5
        n /= np.sqrt(np.sum(n**2, axis=1))[:,np.newaxis]
    return n

def generate_random_data(N, uniform_omega=True):
    """
    Generate random data for testing or training an anti-aliasing model
    :param N: number of training instances to return
    :param uniform_omega: distribute (nx, ny, nz) uniformly in solid angle?
    :return pars: np.array of shape (N, 6), each row (rx, ry, rz, nx, ny, nz)
    :return results: np.array of shape (N, ), each row a partial volume
    """
    # Generate r0 uniformly throughout the block
    r0 = np.random.uniform(size=(N,3)) - 0.5
    # Generate random unit vectors n
    n = generate_unit_vectors(N, uniform_omega=uniform_omega)
    # Generate partial volumes
    pars = np.hstack([r0, n])
    pV = np.array([partial_volume(mesh_eval, r0[i], n[i]) for i in range(N)])
    return pars, pV

class GaussianProcessAntialiasing:

    def __init__(self):
        k1 = GP.kernels.Matern(length_scale=[1.0,1.0], nu=0.5)
        k2 = GP.kernels.WhiteKernel(noise_level=1e-5)
        self.gp = GP.GaussianProcessRegressor(kernel=k1 + k2)

    def _preprocess(self, rawpars):
        """
        Preprocess the raw features into a version useful for GP prediction
        :param rawpars: np.array of shape (N, 6), cf. generate_random_data(N)
        :return: np.array of shape (N, 2), suitable for fitting
        """
        r0, n = rawpars[:,:3], rawpars[:,3:]
        z = np.max(np.abs(n), axis=1)
        # p1 = dot product of r0 into n, the primary predictor
        p1 = np.sum(r0*n, axis=1)
        # p2 = tangent of angle with nearest face of cube
        # this differentiates between face-on, edge-on, and corner-on cases
        p2 = np.sqrt(1-z**2)/z
        return np.vstack([[p1],[p2]]).T

    def fit(self, pars, pV):
        """
        Wraps sklearn.gaussian_process.GaussianProcessRegressor.fit()
        :param pars: np.array of shape (N, 6), cf. generate_random_data(N)
        :param pV: np.array of shape (N, ) containing partial volumes
        :return: nothing (yet)
        """
        X = self._preprocess(pars)
        Y = pV.reshape(-1,1)
        print("X.shape, Y.shape =", X.shape, Y.shape)
        print("Fitting GP...")
        self.gp.fit(X, Y)
        print("Learned kernel: {}".format(self.gp.kernel_))
        print("Log-marginal-likelihood: {:.3f}"
              .format(self.gp.log_marginal_likelihood(self.gp.kernel_.theta)))

    def predict(self, rawpars):
        """
        Wraps sklearn.gaussian_process.GaussianProcessRegresssor.predict()
        :param rawpars: np.array of shape (N, 6), cf. generate_random_data
        :return: np.array of shape (N, )
        """
        X = self._preprocess(rawpars)
        Y = self.gp.predict(X)
        Y[np.abs(X[:,0]) > 0.866] = 0.0
        return Y.ravel()

def compare_antialiasing():
    """
    Demo different functional forms for antialiasing
    :return: nothing (yet)
    """

    def parpV1(x):          # piecewise linear interpolation
        r = 1.1*x + 0.5
        r[r < 0.0] = 0.0
        r[r > 1.0] = 1.0
        return r

    def parpV2(x):          # softmax interpolation
        d = -5*x
        r = 1.0 * (x < 0)
        idx = (np.abs(d) < 100)
        r[idx] = 1.0 / (1.0 + np.exp(d[idx]))
        return r

    # Generate some data and go
    Xtrain, Ytrain = generate_random_data(1000, uniform_omega=True)
    gp = GaussianProcessAntialiasing()
    profile_timer(gp.fit, Xtrain, Ytrain)

    # Show some typical curves
    pars, pV = generate_random_data(100)
    r0, n = pars[:,:3], pars[:,3:]
    fig = plt.figure(figsize=(8,4))
    plt.subplot(121)
    resids1, resids2, resids3 = [ ], [ ], [ ]
    for ni in n:
        x = np.dot(r0, ni)
        X = np.array([np.concatenate([r0i, ni]) for r0i in r0])
        y = np.array([partial_volume(mesh_eval, r0i, ni) for r0i in r0])
        resids1.extend(y - parpV1(x))
        resids2.extend(y - parpV2(x))
        resids3.extend(y - gp.predict(X))
        idx = np.argsort(x)
        plt.plot(x[idx], y[idx], c='gray', lw=0.5)
    x = np.linspace(-1.0, 1.0, 41)
    xgp = np.vstack([[x], [np.ones(len(x))]]).T
    print("xgp.shape =", xgp.shape)
    plt.plot(x, parpV1(x), c='r', lw=2, ls='--', label="piecewise")
    plt.plot(x, parpV2(x), c='b', lw=2, ls='--', label="softmax")
    plt.plot(x, gp.gp.predict(xgp), c='g', lw=2, ls='--', label="GP")
    plt.xlabel("Coverage Parameter $(\mathbf{r_0 \cdot n})/h$")
    plt.ylabel("Cumulative Partial Volume / $h^3$")
    plt.legend()
    plt.subplot(122)
    plt.hist(resids1, bins=40, range=(-0.1, 0.1), alpha=0.5, label='piecewise')
    plt.hist(resids2, bins=40, range=(-0.1, 0.1), alpha=0.5, label='softmax')
    plt.hist(resids3, bins=40, range=(-0.1, 0.1), alpha=0.5, label='GP')
    print("resids(piecewise) mean, std = {:.3g}, {:.3g}"
          .format(np.mean(resids1), np.std(resids1)))
    print("resids(softmax)   mean, std = {:.3g}, {:.3g}"
          .format(np.mean(resids2), np.std(resids2)))
    print("resids(GP)     mean, std = {:.3g}, {:.3g}"
          .format(np.mean(resids3), np.std(resids3)))
    plt.xlabel("Residuals in Partial Volume / $h^3$")
    plt.legend()
    plt.subplots_adjust(bottom=0.15)
    plt.show()


if __name__ == "__main__":
    compare_antialiasing()
