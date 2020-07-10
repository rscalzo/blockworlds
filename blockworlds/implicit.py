#!/usr/bin/env python

"""
RS 2020/06/04:  Basic Implicit Modeling Package

This code is meant to build a package that can define complex geologies as
compositions of events that introduce implicit interfaces into a volume.
"""

import numpy as np
import matplotlib.pyplot as plt
from blockworlds import profile_timer, DiscreteGravity
from blockworlds import baseline_tensor_mesh, survey_gridded_locations


# ============================================================================
#                          Kinds of geological events
# ============================================================================

def l2norm(v):
    return np.sqrt(np.sum(np.atleast_2d(v**2), axis=1))

def soft_if_then(d, y0, y1, h):
    """
    :param y0: limiting value on negative side of d
    :param y1: limiting value on positive side of d
    :param h: transition scale
    :return: y0 if d << 0, y1 if d >> 0, with smooth transition over |d| < h
    """
    # linear (boxcar smoothing kernel)
    result = 0.5*(y0+y1) - (y0-y1)*d/h
    result[d < -0.5*h] = y0[d < -0.5*h]
    result[d > +0.5*h] = y1[d > +0.5*h]
    # error function (Gaussian kernel)
    # result = 0.5 * (1 + erf(2.15 * d))      # goes from 0 to 1
    # result2 = result1*(y1-y0) + y0          # goes from y0 to y1
    return result


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
        return rho*np.ones(shape=r.shape[:-1])


class StratigraphicLayer(GeoFunc):

    def __init__(self, base_gfunc):
        super().__init__(2, base_gfunc)

    def __call__(self, r, h, p):
        dz, rho = p[-2:]
        rp = r + np.array([0, 0, dz])
        rho_up = rho*np.ones(shape=r.shape[:-1])
        rho_down = self.base_gfunc(rp, h, p[:-2])
        return soft_if_then(rp[:,2], rho_down, rho_up, h)


class PlanarFault(GeoFunc):

    def __init__(self, base_gfunc):
        super().__init__(7, base_gfunc)

    def __call__(self, r, h, p):
        r0, n, s = p[-7:-4], p[-4:-1], p[-1]
        v = np.cross(np.cross([0, 0, 1], n), n)
        rdelt = s * v/l2norm(v)
        g0 = self.base_gfunc(r, h, p[:-7])
        g1 = self.base_gfunc(r + rdelt, h, p[:-7])
        return soft_if_then(np.dot(r-r0, n), g0, g1, h)

# ============================================================================
#                Testing construction of non-trivial subsurfaces
# ============================================================================

def plot_soft_if_then():
    x = np.linspace(-10,10,41)
    y = np.ones(shape=x.shape)
    y0 = soft_if_then(x, 0.0*y, 1.0*y, 0.001)
    y1 = soft_if_then(x, 0.0*y, 1.0*y, 2.0)
    y2 = soft_if_then(x, 0.8*y, 0.2*y, 10.0)
    y3 = soft_if_then(x-3.0, y1, y2, 3.0)
    plt.plot(x, y0)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3, ls='--')
    plt.show()

def plot_subsurface_01():
    """
    Create something with a single stratigraphic layer
    :return: nothing (but plot the result)
    """
    z0, L, NL = 0.0, 10000.0, 30
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
    histpars.extend([-4000.0, 0.0, 0.0, 0.950, 0.0, 0.312, -4200.0])
    # Fault #2
    histpars.extend([+4000.0, 0.0, 0.0, 0.950, 0.0, -0.312, 4200.0])
    fwdmodel = DiscreteGravity(mesh, survey, history[0])
    fwdmodel.gfunc = Basement()
    fwdmodel.edgemask = profile_timer(fwdmodel.calc_gravity, h, [1.0])
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

if __name__ == "__main__":
    # plot_soft_if_then()
    plot_subsurface_01()
