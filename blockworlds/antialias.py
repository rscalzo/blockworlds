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

    def __init__(self, N_features=3, nu=1.5):
        """
        :param N_features: number of features for prediction (1, 2, or 3)
        :param nu: degrees of freedom in Matern kernel for GP
        """
        if N_features not in (1, 2, 3):
            raise IndexError("GaussianProcessAntialiasing.__init__:"
                             "  N_features must be either 1, 2, or 3")
        self.N_features = N_features
        length_scale = np.ones(N_features)
        k1 = GP.kernels.Matern(length_scale=length_scale, nu=nu)
        k2 = GP.kernels.WhiteKernel(noise_level=1e-5)
        self.gp = GP.GaussianProcessRegressor(kernel=k1 + k2)

    def _preprocess(self, rawpars):
        """
        Preprocess the raw features into a version useful for GP prediction
        :param rawpars: np.array of shape (N, 6), cf. generate_random_data(N)
        :return: np.array of shape (N, 2), suitable for fitting
        """
        r0, n = rawpars[:,:3], rawpars[:,3:]
        p = np.zeros(shape=(len(r0), 3))
        # feature 1 = dot product of r0 into n, the primary predictor
        p[:,0] = np.sum(r0*n, axis=1)
        # feature 2 = tangent of angle with nearest face of cube
        # this differentiates between face-on, edge-on, and corner-on cases
        if self.N_features >= 2:
            z = np.max(np.abs(n), axis=1)
            p[:,1] = np.sqrt(1-z**2)/z
        # feature 3 = sine of azimuthal angle for further direction detail
        if self.N_features >= 3:
            x = np.min(np.abs(n), axis=1)
            p[:,2] = x/np.sqrt(1-z**2)
        return p[:,:self.N_features]

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
        Y[Y < 0] = 0.0
        Y[Y > 1] = 1.0
        return Y.ravel()

    def predict_1d(self, d):
        """
        Wraps predictions based only on the first GP feature, marginalizing
        out the other two features if they have been fitted.
        :param d: the first feature np.dot(r0, d), np.array of shape (N, )
        :return: np.array of shape (N, )
        """
        X = np.repeat(d, self.N_features).reshape(-1, self.N_features)
        if self.N_features >= 2:
            X[:,1] = 0.44
        if self.N_features >= 3:
            X[:,2] = 0.64
        Y = self.gp.predict(X)
        Y[Y < 0] = 0.0
        Y[Y > 1] = 1.0
        return Y.ravel()

def compare_antialiasing(N_features_gp=3, vertical=False, histlogy=False):
    """
    Demo different functional forms for antialiasing
    :param N_features_gp: number of GP features to use (1, 2, or 3)
    :param vertical: stack plots vertically (True) or side by side (False)?
    :param histlogy: use logarithmic y-axis for histogram? (default False)
    :return: nothing (yet)
    """

    def parpV1(x):          # piecewise linear interpolation
        r = 1.0*x + 0.5
        r[r < 0.0] = 0.0
        r[r > 1.0] = 1.0
        return r

    def parpV2(x):          # error function (cdf of a Gaussian)
        r = 1.0 * (x < 0)
        idx = (np.abs(r) < 100)
        r[idx] = 0.5*(1 + np.tanh(2.2*x[idx] + 3.2*(x[idx])**3))
        return r

    def parpV3(x, gp):      # GP interpolation (w/one feature, for display)
        # Grab the underlying GP and evaluate it using a single feature
        # This is only for plots; residuals calculated using all features
        r = gp.predict_1d(x)
        return r

    # Generate some data and go
    Xtrain, Ytrain = generate_random_data(1000, uniform_omega=True)
    gp = GaussianProcessAntialiasing(N_features=N_features_gp)
    profile_timer(gp.fit, Xtrain, Ytrain)

    # Show some typical curves
    pars, pV = generate_random_data(200)
    r0, n = pars[:,:3], pars[:,3:]
    if vertical:
        figsize=(4.8, 6)
    else:
        figsize=(10, 6)
    fig = plt.figure(figsize=figsize)
    plt.subplot(211)
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
    plt.plot(x, parpV1(x), c='r', lw=2, ls='--', label="piecewise")
    plt.plot(x, parpV2(x), c='b', lw=2, ls='--', label="linear model")
    plt.plot(x, parpV3(x,gp), c='g', lw=2, ls='--',
             label="GP ($N_\mathrm{{pars}} = {}$)".format(N_features_gp))
    plt.xlabel("Coverage Parameter $(\mathbf{r_0 \cdot n})/h$")
    plt.ylabel("Cumulative Partial Volume / $h^3$")
    plt.legend()

    # Histograms
    Nhistbins = 50
    histrange = (-0.1, 0.1)
    plt.subplot(212)
    plt.hist(resids1, bins=Nhistbins, range=histrange, log=histlogy,
             color='r', alpha=0.5, label='piecewise')
    plt.hist(resids2, bins=Nhistbins, range=histrange, log=histlogy,
             color='b', alpha=0.5, label='linear model')
    plt.hist(resids3, bins=Nhistbins, range=histrange, log=histlogy,
             color='g', alpha=0.5,
             label="GP ($N_\mathrm{{pars}} = {}$)".format(N_features_gp))
    print("resids(piecewise)    mean, std, mad, max "
          "= {:.3g}, {:.3g}, {:.3g}, {:.3g}"
          .format(np.mean(resids1), np.std(resids1),
                  np.mean(np.abs(resids1)), np.max(np.abs(resids1))))
    print("resids(linear model) mean, std, mad, max "
          "= {:.3g}, {:.3g}, {:.3g}, {:.3g}"
          .format(np.mean(resids2), np.std(resids2),
                  np.mean(np.abs(resids2)), np.max(np.abs(resids2))))
    print("resids(GP)           mean, std, mad, max "
          "= {:.3g}, {:.3g}, {:.3g}, {:.3g}"
          .format(np.mean(resids3), np.std(resids3),
                  np.mean(np.abs(resids3)), np.max(np.abs(resids3))))
    plt.xlabel("Residuals in Partial Volume / $h^3$")
    plt.legend()
    if vertical:
        # for vertical format
        plt.subplots_adjust(bottom=0.08, top=0.92,
                            left=0.12, right=0.88, hspace=0.35)
        # save as figure for paper
        plt.savefig("compare_antialiasing.eps")
    else:
        # for horizontal format
        plt.subplots_adjust(bottom=0.15, top=0.88,
                            left=0.08, right=0.92, wspace=0.25)
    plt.show()


def fit_antialiasing():
    """
    Look for alternative models that can deliver GP-like accuracy, quickly
    :return: nothing (yet)
    """

    # Generate some random data
    np.random.seed(42)
    Xtrain, Ytrain = generate_random_data(1000, uniform_omega=True)
    # Transform Xtrain to the set of features useful in GP regression
    gp = GaussianProcessAntialiasing(N_features=3)
    Xtrain = gp._preprocess(Xtrain)

    # Recursive experimental design to avoid double-counting cross-terms
    def indices(N, order, terms=[]):
        termslist = [ ]
        # Termination case
        if len(terms) == order:
            return terms
        # Starting case
        elif len(terms) == 0:
            ilo = 0
        # Recursion
        else:
            ilo = terms[-1]
        for i in range(ilo, N):
            termslist.append(indices(N, order, terms=(terms + [i])))
        termslist = np.concatenate(termslist).reshape(-1, order)
        return termslist

    # Unique cross-terms
    def generate_cross_terms(X, order=1):
        Xlist = np.ones(shape=(len(X), 1))
        for k in range(1, order+1):
            for idx in indices(X.shape[1], k):
                cols = np.array([X[:,i] for i in idx])
                newcol = np.prod(cols, axis=0).reshape(-1, 1)
                Xlist = np.hstack([Xlist, newcol])
        return np.array(Xlist)

    # Hand-picked features up to order 3 based on symmetries of the cube
    def selected_features(X):
        rdotn, tanth, tanph = X.T
        # Odd terms in rdot; even terms in tanth and tanph
        Xp = np.hstack([rdotn.reshape(-1,1),
                        (rdotn**3).reshape(-1,1)])
        return Xp

    Xtrain = selected_features(Xtrain)
    # Xtrain = generate_cross_terms(Xtrain, order=3)
    print("Xtrain.shape =", Xtrain.shape)

    # Define a predictor and/or an objective function
    def linmodel(X, *p):
        Xp = np.dot(X, p)
        Ypred = np.zeros(Xp.shape)
        idx_ok = (Xp < 100.0)
        Ypred[idx_ok] = 0.5*(1 + np.tanh(Xp[idx_ok]))
        # experiment:  can a polynomial do just as well?
        # Ypred[idx_ok] = 0.5*(1 + Xp[idx_ok])
        return Ypred

    def logpost(p, *args):
        # print("args =", args)
        X, Y, Lreg = args
        Ypred = linmodel(X, *p)
        resids = Ypred - Y
        # maximizing log probability = minimizing -log probability
        return np.sum((Ypred-Y)**2) + Lreg*np.sum(np.abs(p))

    # Let's get out our curve-fitting apparatus
    from scipy.optimize import curve_fit
    p0 = np.zeros(Xtrain.shape[1])
    popt, pcov = profile_timer(curve_fit, linmodel, Xtrain, Ytrain, p0)
    resids = (linmodel(Xtrain, *popt)-Ytrain)
    print("linear model fit:  popt =", popt)
    print("resids:  mean = {}, std = {}".format(resids.mean(), resids.std()))

    # Let's add the lasso penalty now
    # Find the regularization strength by cross-validation
    from scipy.optimize import minimize
    popts = [ ]
    objfevals = [ ]
    Lregvals = 10**np.arange(-3.0, 5.5, 0.5)
    for Lreg in Lregvals:
        results = profile_timer(minimize, logpost, p0, (Xtrain, Ytrain, Lreg),
                                method='Nelder-Mead', options={'maxiter': 10000})
        popt = np.array(results.x)
        objfunc = logpost(popt, Xtrain, Ytrain, Lreg)
        print("Lreg = {}:  objfunc = {}".format(Lreg, objfunc))
        popts.append(popt)
        objfevals.append(objfunc)
    # Select threshold value of parameters favoring shrinkage
    objfevals = np.array(objfevals)
    idx = np.arange(len(Lregvals))
    iopt = len(idx[objfevals < 2.0*np.min(objfevals)])
    print("selected Lreg = {}, popt = {}".format(Lregvals[iopt], popts[iopt]))
    resids = (linmodel(Xtrain, *popts[iopt]) - Ytrain)
    print("resids:  mean = {}, std = {}".format(resids.mean(), resids.std()))


if __name__ == "__main__":
    compare_antialiasing(N_features_gp=3, vertical=True)
    # fit_antialiasing()
