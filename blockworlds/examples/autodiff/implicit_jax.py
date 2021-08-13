#!/usr/bin/env python

"""
RS 2020/06/04:  Auto-Differentiating Implicit Modeling Package

This particular part of the package is meant to incorporate auto-derivatives
using Google Jax.  It doesn't look like it'll be that useful in itself,
given that it's incredibly slow and memory-intensive even for fairly modest
voxel sizes and histories.  But perhaps it can be a useful touchstone.
"""

import time
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from jax import grad, jacobian, jit
from jax.config import config
import jax.numpy as jnp
import jax.scipy.special
from blockworlds.gravity import profile_timer, DiscreteGravity
from blockworlds.gravity import baseline_tensor_mesh, survey_gridded_locations

config.update("jax_debug_nans", True)

# ============================================================================
#                               Helper functions
# ============================================================================


def soft_if_then_piecewise_ifthen(d, y0, y1, h):
    # linear (boxcar smoothing kernel) as if/then (for jax.grad)
    if d <= -0.5*h:
        return y0
    elif d <= 0.5*h:
        return 0.5*(y0+y1) - (y0-y1)*d/h
    else:
        return y1

def soft_if_then_piecewise(d, y0, y1, h):
    """
    :param y0: limiting value on negative side of d
    :param y1: limiting value on positive side of d
    :param h: transition scale
    :return: y0 if d << 0, y1 if d >> 0, with smooth transition over |d| < h
    """
    # linear (boxcar smoothing kernel) as subscript
    result = 0.5*(y0+y1) - (y0-y1)*d/h
    result[d < -0.5*h] = y0[d < -0.5*h]
    result[d > +0.5*h] = y1[d > +0.5*h]
    return result

def soft_if_then_logjit(d, y0, y1, h):
    # these arguments have been optimized to best approximate the mean
    # partial volume of a cube averaged over all slicing directions
    tanh_arg = 2.2*d/h + 3.3*(d/h)**3
    return (y1-y0) * (1.0 + jnp.tanh(tanh_arg))/2.0 + y0


# ============================================================================
#                 Initial implementation of events as GeoFuncs
# ----------------------------------------------------------------------------
# RS 2020/10/09:  This was the first implementation of kinematic histories
# before I refactored it into the GeoHistory and GeoEvent classes.  I'm
# keeping them here since Jax only works on pure Python functions and don't
# want to have to think at this stage about making GeoHistory Jax-compliant
# before I've assessed whether autodiff is useful for these models.
# ============================================================================


class GeoFunc:

    def __init__(self, npars, base_gfunc):
        self.npars = npars
        self.base_gfunc = base_gfunc

    def __call__(self, r, h, p):
        raise NotImplementedError


class Basement(GeoFunc):

    def __init__(self):
        super().__init__(1, None)

    def __call__(self, r, h, p):
        rho = p[0]
        return rho*jnp.ones(shape=r.shape[:-1])


class StratigraphicLayer(GeoFunc):

    def __init__(self, base_gfunc):
        super().__init__(2, base_gfunc)

    def __call__(self, r, h, p):
        dz, rho = p[-2:]
        rp = r + jnp.array([0.0, 0.0, dz]) # jnp.array([0.0, 0.0, dz])
        rho_up = rho*jnp.ones(shape=r.shape[:-1])
        rho_down = self.base_gfunc(rp, h, p[:-2])
        return soft_if_then_logjit(rp[:,2], rho_down, rho_up, h)


class PlanarFault(GeoFunc):

    def __init__(self, base_gfunc):
        super().__init__(7, base_gfunc)

    def __call__(self, r, h, p):
        r0, n, s = jnp.array(p[-7:-4]), jnp.array(p[-4:-1]), p[-1]
        nz = jnp.array([0.0, 0.0, 1.0])
        v = jnp.cross(jnp.cross(nz, n), n)
        rdelt = s * v/jnp.sqrt(jnp.dot(v, v))
        g0 = self.base_gfunc(r, h, p[:-7])
        g1 = self.base_gfunc(r + rdelt, h, p[:-7])
        return soft_if_then_logjit(jnp.dot(r-r0, n), g0, g1, h)


# ============================================================================
#                Testing construction of non-trivial subsurfaces
# ============================================================================


def gradtest():
    """
    Testing ground for adding automatic gradients to GeoFuncs
    :return: nothing (yet)
    """
    # All the same setup as before
    z0, L, NL = 0.0, 10000.0, 15
    h = L/NL
    print("z0, L, nL, h =", z0, L, NL, h)
    mesh = baseline_tensor_mesh(NL, h, centering='CCN')
    survey = survey_gridded_locations(L, L, 20, 20, z0)
    history = [Basement()]
    history.append(StratigraphicLayer(history[-1]))
    history.append(StratigraphicLayer(history[-1]))
    history.append(PlanarFault(history[-1]))
    history.append(PlanarFault(history[-1]))
    # Basic stratigraphy
    histpars = [3.0, 1900.0, 2.5, 2500.0, 2.0]
    # Fault #1
    histpars.extend([-4000.0, 0.0, 0.0, 0.940, 0.0, 0.342, -4200.0])
    # Fault #2
    histpars.extend([+4000.0, 0.0, 0.0, 0.940, 0.0, -0.342, 4200.0])
    # Forward model stuff
    fwdmodel = DiscreteGravity(mesh, survey, history[0])
    fwdmodel.gfunc = Basement()
    fwdmodel.edgemask = profile_timer(fwdmodel.calc_gravity, h, [1.0])

    # Test gradient of anti-aliasing function
    x = np.linspace(-1, 1, 4001)
    t0 = time.time()
    g = [soft_if_then_piecewise_ifthen(xi, 0.0, 1.0, 1.0) for xi in x]
    t1 = time.time()
    g = soft_if_then_piecewise(x, np.zeros(x.shape), np.ones(x.shape), 1.0)
    t2 = time.time()
    g = soft_if_then_logjit(x, np.zeros(x.shape), np.ones(x.shape), 1.0)
    t3 = time.time()
    print("looped:    {:.2e} sec".format(t1-t0))
    print("vector:    {:.2e} sec".format(t2-t1))
    print("logistic:  {:.2e} sec".format(t3-t2))
    # a vectorized way to take the 1-D derivative
    logistic_jac = jacobian(soft_if_then_logjit)
    dg = np.sum(logistic_jac(x, 0.0, 1.0, 1.0), axis=0)
    fig = plt.figure()
    plt.plot(x, g)
    plt.plot(x, dg, ls='--')
    plt.show()

    # Go through the history event by event
    for m, part_history in enumerate(history):
        fwdmodel.gfunc = part_history
        npars = np.sum([e.npars for e in history[:m+1]])
        profile_timer(fwdmodel.calc_gravity, h, histpars[:npars])
        fwdmodel.fwd_data -= fwdmodel.edgemask * fwdmodel.voxmodel.mean()
        fig = plt.figure(figsize=(12,4))
        ax1 = plt.subplot(121)
        fwdmodel.plot_model_slice(ax=ax1)
        ax2 = plt.subplot(122)
        fwdmodel.plot_gravity(ax=ax2)
        plt.show()

    # Finite differences, hrngh
    t0 = time.time()
    dgfd = [ ]
    for i in range(len(histpars)):
        hp1, hp2 = np.array(histpars), np.array(histpars)
        dth = 1e-7*max(1, histpars[i])
        hp1[i] -= dth
        vox1 = history[-1](fwdmodel.mesh.gridCC, h, hp1)
        hp2[i] += dth
        vox2 = history[-1](fwdmodel.mesh.gridCC, h, hp2)
        dgfd.append((vox2-vox1)/(2*dth))
    dgfd = np.array(dgfd).T
    t1 = time.time()
    print("dgfd ran in {:.3f} seconds".format(t1-t0))

    # Demonstrate an end-to-end derivative w/rt parameters
    gfuncjac = profile_timer(jacobian, history[-1], 2)
    gfuncjac = jit(gfuncjac)
    jhistpars = jnp.array(histpars)
    dg = profile_timer(gfuncjac, fwdmodel.mesh.gridCC, h, jhistpars)
    print("dg.shape =", dg.shape)
    for i in range(3):
        dg = profile_timer(gfuncjac, fwdmodel.mesh.gridCC, h, jhistpars)
    for dgi in dg.T:
        fwdmodel.voxmodel = np.array(dgi)
        print("voxmodel.mean, voxmodel.std =",
              fwdmodel.voxmodel.mean(), fwdmodel.voxmodel.std())
        print("voxmodel.isnan.sum() =", np.sum(np.isnan(fwdmodel.voxmodel)))
        # fwdmodel.plot_model_slice()
        # plt.show()

    idx = (np.abs(dgfd) > 1e-9)
    resids = 0.5*(dgfd[idx]-dg[idx])/(np.abs(dg[idx]+dgfd[idx]))
    print("mean fractional derivative error: {:.3g} +/- {:.3g}"
          .format(resids.mean(), resids.std()))


if __name__ == "__main__":
    gradtest()