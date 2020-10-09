#!/usr/bin/env python

"""
RS 2020/10/09:  MCMC Demo on Kinematic Models

This separates the MCMC dependencies out from the main code.  It's basically
the same infrastructure I've been using in the notebooks, put into production.
"""

import numpy as np
import matplotlib.pyplot as plt
from blockworlds import profile_timer, DiscreteGravity
from blockworlds import baseline_tensor_mesh, survey_gridded_locations
from implicit import gen_two_fault_model_demo
from riemann import Sampler, Model
from riemann.proposals.randomwalk import AdaptiveMetropolisRandomWalk as AMRW
import pickle


# Reproducible random numbers
np.random.seed(413)

# True values of parameters for various demo models
model_pars = np.array([
    # Original graben model
    [3.0, 350.0, 2.5, 190.0, 2.0,           # layer densities and thicknesses
     -400.0, 0.0, +20.0, 0.0, -220.0,       # 1st fault x0, y0, theta, phi, s
     +400.0, 0.0, -20.0, 0.0, +220.0],      # 2nd fault x0, y0, theta, phi, s
    # Mark Lindsay's implicit model 1
    [3.0, 350.0, 2.5, 190.0, 2.0,
     -400.0, 0.0, +45.0, 0.0, -220.0,
     +400.0, 0.0, -45.0, 0.0, +220.0],
    # Mark Lindsay's implicit model 2
    [3.0, 350.0, 2.5, 190.0, 2.0,
     -450.0, 0.0, 45.0, 0.0, -220.0,
      +50.0, 0.0, 20.0, 0.0, +220.0],
    # Mark Lindsay's implicit model 3
    [3.0, 350.0, 2.5, 190.0, 2.0,
     -50.0, 0.0, -20.0, 0.0, -220.0,
     +50.0, 0.0, 20.0, 0.0, +220.0],
    # Mark Lindsay's implicit model 4
    [3.0, 350.0, 2.5, 190.0, 2.0,
     -250.0, 0.0, 0.0, 0.0, -220.0,
     +250.0, 0.0, 0.0, 0.0, +220.0],
    # Mark Lindsay's implicit model 5
    [3.0, 350.0, 2.5, 190.0, 2.0,
     -450.0, 0.0, 30.0, 0.0, 220.0,
     +50.0, 0.0, 10.0, 0.0, -220.0],
    # Mark Lindsay's implicit model 6
    [3.0, 350.0, 2.5, 190.0, 2.0,
     -400.0, 0.0, 20.0, 0.0, 220.0,
     -300.0, 0.0, 40.0, 0.0, -220.0],
    # Mark Lindsay's implicit model 7
    [3.0, 350.0, 2.5, 190.0, 2.0,
     -400.0, 0.0, 20.0, 0.0, 140.0,
     -300.0, 0.0, 40.0, 0.0, 80.0],
])

# Characteristic initial-guess step sizes for MCMC
stepsizes = np.array([0.1, 100, 0.01, 100, 0.01,
                      1.0, 1.0, 1.0, 1.0, 100,
                      1.0, 1.0, 1.0, 1.0, 100])


# ============================================================================
#             riemann classes specifying statistical model for MCMC
# ============================================================================


class GeoModel(Model):

    def __init__(self, history, fwdmodel, dsynth, sigdata):
        # Set baseline attributes
        self.history = history
        self.fwdmodel = fwdmodel
        self.dsynth = dsynth
        self.sigdata = sigdata
        # Set forward model to represent history
        self.fwdmodel.gfunc = history.rockprops
        # Turn full anti-aliasing on; match h to grid resolution
        mesh = self.fwdmodel.mesh
        self.h = np.exp(np.mean(np.log([mesh.hx, mesh.hy, mesh.hz])))

    def log_likelihood(self, theta):
        # Load this parameter vector into the history and calculate gravity
        self.history.deserialize(theta)
        dpred = self.fwdmodel.calc_gravity(self.h)
        resids = dpred - self.dsynth
        resids = resids - resids.mean()
        # Independent Gaussian likelihood with variance sigdata**2
        return -0.5 * np.sum(resids ** 2 / self.sigdata ** 2 +
                             np.log(2 * np.pi * self.sigdata ** 2))

    def log_prior(self, theta):
        # Load this parameter vector into the history and calculate prior density
        self.history.deserialize(theta)
        return self.history.logprior()


def initialize_geomodel(pars):
    """
    Initialize a GeoModel instance and get it ready for sampling
    :param pars: parameter vector (see top of file)
    :return: GeoModel instance ready for sampling with riemann
    """
    # Initialize a mesh for forward gravity calculation
    z0, L, NL = 0.0, 1000.0, 15
    h = L / NL
    print("z0, L, nL, h =", z0, L, NL, h)
    mesh = baseline_tensor_mesh(NL, h, centering='CCN')
    survey = survey_gridded_locations(L, L, 20, 20, z0)

    # Initialize a GeoHistory based on the parameters passed in
    history = gen_two_fault_model_demo(pars)

    # Initialize a DiscreteGravity forward model instance
    fwdmodel = DiscreteGravity(mesh, survey, history.event_list[0])
    fwdmodel.gfunc = history.event_list[0].rockprops
    fwdmodel.edgemask = profile_timer(fwdmodel.calc_gravity, h)

    # Make some synthetic data based on this history and mesh
    data0 = fwdmodel.calc_gravity(h)
    sigdata = 0.05 * np.std(data0)
    epsilon = sigdata * np.random.normal(size=data0.shape)
    dsynth = data0 + epsilon

    # Construct and return a GeoModel
    return GeoModel(history, fwdmodel, dsynth, sigdata)

def run_mcmc(model, Nsamp=100000, Nburn=20000, Nthin=100):
    """
    Runs the MCMC using riemann.
    :param model: GeoModel instance
    :return: chain
    """
    print("run_mcmc: running chain...")
    model.history.set_to_prior_draw()
    histpars = model.history.serialize()
    proposal = AMRW(0.1 * np.diag(stepsizes), 100, marginalize=False)
    sampler = Sampler(model, proposal, np.array(histpars))
    profile_timer(sampler.run, Nsamp)
    chain = np.array(sampler._chain_thetas)
    accept_frac = np.mean(chain[1:] - chain[:-1] != 0)
    print("run_mcmc: chain finished; acceptance fraction =", accept_frac)
    return chain[Nburn:Nsamp:Nthin]

def gelman_rubin(data, verbose=False):
    """
    Apply Gelman-Rubin convergence diagnostic to a collection of chains.
    :param data: np.array of shape (Nchains, Nsamples, Npars)
    """
    Nchains, Nsamples, Npars = data.shape
    B_on_n = data.mean(axis=1).var(axis=0)      # variance of in-chain means
    W = data.var(axis=1).mean(axis=0)           # mean of in-chain variances

    # simple version, as in Obsidian
    sig2 = (Nsamples/(Nsamples-1))*W + B_on_n
    Vhat = sig2 + B_on_n/Nchains
    Rhat = Vhat/W

    # advanced version that accounts for degrees of freedom
    # see Gelman & Rubin, Statistical Science 7:4, 457-472 (1992)
    m, n = np.float(Nchains), np.float(Nsamples)
    si2 = data.var(axis=1)
    xi_bar = data.mean(axis=1)
    xi2_bar = data.mean(axis=1)**2
    var_si2 = data.var(axis=1).var(axis=0)
    allmean = data.mean(axis=1).mean(axis=0)
    cov_term1 = np.array([np.cov(si2[:,i], xi2_bar[:,i])[0,1]
                          for i in range(Npars)])
    cov_term2 = np.array([-2*allmean[i]*(np.cov(si2[:,i], xi_bar[:,i])[0,1])
                          for i in range(Npars)])
    var_Vhat = ( ((n-1)/n)**2 * 1.0/m * var_si2
             +   ((m+1)/m)**2 * 2.0/(m-1) * B_on_n**2
             +   2.0*(m+1)*(n-1)/(m*n**2)
                    * n/m * (cov_term1 + cov_term2))
    df = 2*Vhat**2 / var_Vhat
    if verbose:
        print("gelman_rubin(): var_Vhat = {}".format(var_Vhat))
        print("gelman_rubin(): df = {}".format(df))
    Rhat *= df/(df-2)

    return Rhat


# ============================================================================
#                    Helper functions for figure generation
# ============================================================================


def run_grid(model, p1_vals, p2_vals, p1_idx, p2_idx):
    origpars = model.history.serialize()
    grid_vals = [ ]
    for p1i in p1_vals:
        for p2i in p2_vals:
            theta = np.array(origpars)
            theta[p1_idx], theta[p2_idx] = p1i, p2i
            grid_vals.append([p1i, p2i, model.log_posterior(theta)])
    model.history.deserialize(origpars)
    grid_vals = np.array(grid_vals).T.reshape(3, len(p1_vals), len(p2_vals))
    return grid_vals

def show_contours(xg, yg, Lg, p1_vals, p2_vals, p1_0, p2_0):
    levels = 10**np.arange(-6,0.1)
    levels = np.log(10)*np.arange(-6,0.1)
    # plt.contourf(xg, yg, np.exp(Lg - Lg.max()), levels=levels)
    plt.contourf(xg, yg, Lg - Lg.max(), levels=levels)
    plt.colorbar()
    ax = plt.gca()
    ax.set_xlim(p1_vals.min(), p1_vals.max())
    ax.set_ylim(p2_vals.min(), p2_vals.max())

def vet_slice(model, z1_idx, z2_idx, zdelt1, zdelt2, Nz):
    histpars = np.array(model.history.serialize())
    z1_0, z2_0 = histpars[z1_idx], histpars[z2_idx]
    z1_vals = np.linspace(z1_0-0.5*zdelt1, z1_0+0.5*zdelt1, Nz)
    z2_vals = np.linspace(z2_0-0.5*zdelt2, z2_0+0.5*zdelt2, Nz)
    xg, yg, Lg = profile_timer(run_grid, model, z1_vals, z2_vals, z1_idx, z2_idx)
    show_contours(xg, yg, Lg, z1_vals, z2_vals, z1_0, z2_0)


# ============================================================================
#                               Main routine(s)
# ============================================================================


def run_experiments():
    M, Nsamp, Nburn, Nthin = 4, 100000, 20000, 100
    model_chains_h0, model_chains_h1 = [ ], [ ]
    for pars in model_pars:
        # Initialize model and grab the mesh cell size
        model = initialize_geomodel(pars)
        h = model.h
        # Turn anti-aliasing off and sample
        model.h = 0.001 * h
        chains = np.array([run_mcmc(model, Nsamp, Nburn, Nthin) for i in range(M)])
        model_chains_h0.append(np.array(chains))
        print("chains_h0.shape =", chains.shape)
        print("gelman_rubin(): Rhat = {}".format(gelman_rubin(chains)))
        # Turn anti-aliasing back on and sample again
        model.h = h
        chains = np.array([run_mcmc(model, Nsamp, Nburn, Nthin) for i in range(M)])
        model_chains_h1.append(np.array(chains))
        print("chains_h1.shape =", chains.shape)
        print("gelman_rubin(): Rhat = {}".format(gelman_rubin(chains)))
    # Save to pickles for later; most likely faster than re-running!
    with open("model_chains_h0.pkl", 'wb') as pklfile:
        pickle.dump(np.array(model_chains_h0), pklfile)
    with open("model_chains_h1.pkl", 'wb') as pklfile:
        pickle.dump(np.array(model_chains_h1), pklfile)

if __name__ == "__main__":
    run_experiments()
