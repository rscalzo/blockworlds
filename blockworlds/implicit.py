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
    # result = 0.5 * (1 + erf(2.15 * d/h))          # goes from 0 to 1
    # result = result*(y1-y0) + y0                  # goes from y0 to y1
    # tanh function (some other smooth kernel)
    # result = 0.5 * (1 + np.tanh(2.5*d/h))         # goes from 0 to 1
    # result = result*(y1-y0) + y0                  # goes from y0 to y1
    return result

# ============================================================================
#                 Initial implementation of events as GeoFuncs
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
#                More sophisticated implementation of GeoEvents
# ============================================================================


class GeoEvent:

    _attrs = [ ]

    def __init__(self, *args, **kwargs):
        self.deserialize(*args)
        self.set_kw_attrs(**kwargs)
        self.previous_event = None
        self.Npars = len(self._attrs)

    def serialize(self):
        return np.array([getattr(self, attr) for attr in self._attrs])

    def deserialize(self, *args):
        for i, arg in enumerate(args):
            setattr(self, self._attrs[i], args[i])

    def set_kw_attrs(self, **kwargs):
        for key, val in kwargs:
            if key in self._attrs:
                setattr(self, key, val)

    def get_kw_attrs(self):
        return { attr: getattr(self, attr) for attr in self._attrs }

    def set_previous_event(self, event):
        self.previous_event = event

    def rockprops(self, r, h):
        raise NotImplementedError

    def log_prior(self):
        return 0.0

    def __str__(self):
        np = zip(self._attrs, self.serialize())
        parstr = ', '.join(["{}={}".format(n, p) for n, p in np])
        return "{}({})".format(self.__class__.__name__, parstr)


class BasementEvent(GeoEvent):

    _attrs = ['density']

    def rockprops(self, r, h):
        return self.density * np.ones(shape=r.shape[:-1])

    def log_prior(self):
        # density: lognormal with default mean and variance
        lp = -0.5*((np.log10(self.density) - 0.5)/0.1)**2
        return lp


class StratLayerEvent(GeoEvent):

    _attrs = ['thickness', 'density']

    def rockprops(self, r, h):
        assert(isinstance(self.previous_event, GeoEvent))
        rp = r + np.array([0, 0, self.thickness])
        rho_up = self.density*np.ones(shape=r.shape[:-1])
        rho_down = self.previous_event.rockprops(rp, h)
        return soft_if_then(rp[:,2], rho_down, rho_up, h)

    def log_prior(self):
        # density: lognormal with default mean and variance
        # thickness: exponential with default scale length (100 km)
        lp = -0.5*((np.log10(self.density) - 0.5)/0.2)**2
        lp = lp - (np.inf if self.thickness < 0 else (self.thickness/1e+5))
        return lp


class PlanarFaultEvent(GeoEvent):

    _attrs = ['x0', 'y0', 'nth', 'nph', 's']

    def rockprops(self, r, h):
        assert(isinstance(self.previous_event, GeoEvent))
        # Point on fault specified in Cartesian coordinates; assume z0 = 0
        # since we're probably just including geologically observed faults
        r0 = np.array([self.x0, self.y0, 0.0])
        # Unit normal to fault ("polar vector") specified with
        # nth = elevation angle (+90 = +z, -90 = -z)
        # nph = azimuthal angle (runs counterclockwise, zero in +x direction)
        th, ph = np.radians(self.nth), np.radians(self.nph)
        n = [np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), np.sin(th)]
        # Geology in +n direction slips relative to the background
        # Slip is vertical (+z direction) in units of meters along the fault
        v = np.cross(np.cross([0, 0, 1], n), n)
        rdelt = self.s * v/l2norm(v)
        g0 = self.previous_event.rockprops(r, h)
        g1 = self.previous_event.rockprops(r + rdelt, h)
        return soft_if_then(np.dot(r-r0, n), g0, g1, h)

    def log_prior(self):
        # nth, nph: uniform in solid angle
        # s: exponential with default scale length
        lp = np.cos(np.radians(self.nth))
        lp = lp - 0.5*(self.s/1e+5)**2
        return lp


class GeoHistory:

    def __init__(self):
        self.event_list = [ ]

    def add_event(self, event):
        if len(self.event_list) == 0:
            assert(isinstance(event, BasementEvent))
        else:
            assert(isinstance(event, GeoEvent))
            event.set_previous_event(self.event_list[-1])
        self.event_list.append(event)

    def serialize(self):
        return np.concatenate([event.serialize() for event in self.event_list])

    def deserialize(self, pvec):
        for event in self.event_list:
            psub, pvec = pvec[:event.Npars], pvec[event.Npars:]
            event.deserialize(*psub)

    def rockprops(self, r, h):
        return self.event_list[-1].rockprops(r, h)

    def logprior(self):
        return np.sum([event.log_prior() for event in self.event_list])


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
    Create a basic graben geology using recursive procedural API
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
    histpars.extend([-4000.0, 0.0, 0.0, 0.940, 0.0, 0.342, -4200.0])
    # Fault #2
    histpars.extend([+4000.0, 0.0, 0.0, 0.940, 0.0, -0.342, 4200.0])
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

def plot_subsurface_02():
    """
    Create a basic graben geology using object-oriented API
    :return: nothing (but plot the result)
    """
    # Initialize basic grid parameters
    z0, L, NL = 0.0, 10000.0, 30
    h = L/NL
    print("z0, L, nL, h =", z0, L, NL, h)
    mesh = baseline_tensor_mesh(NL, h, centering='CCN')
    survey = survey_gridded_locations(L, L, 20, 20, z0)
    # Create the history
    history = GeoHistory()
    history.add_event(BasementEvent(3.0))
    history.add_event(StratLayerEvent(1900.0, 2.5))
    history.add_event(StratLayerEvent(2500.0, 2.0))
    history.add_event(PlanarFaultEvent(-4000.0, 0.0, +20.0, 0.0, -4200.0))
    history.add_event(PlanarFaultEvent(+4000.0, 0.0, -20.0, 0.0, +4200.0))
    print("history.pars =", history.serialize())
    # Can also set parameters all at once -- good for running MCMC
    history = GeoHistory()
    history.add_event(BasementEvent())
    history.add_event(StratLayerEvent())
    history.add_event(StratLayerEvent())
    history.add_event(PlanarFaultEvent())
    history.add_event(PlanarFaultEvent())
    history.deserialize([3.0, 1900.0, 2.5, 2500.0, 2.0,
                         -4000.0, 0.0, +20.0, 0.0, -4200.0,
                         +4000.0, 0.0, -20.0, 0.0, +4200.0])
    print("history.pars =", history.serialize())
    # Plot a cross-section
    fwdmodel = DiscreteGravity(mesh, survey, history.event_list[0])
    fwdmodel.gfunc = BasementEvent(1.0).rockprops
    fwdmodel.edgemask = profile_timer(fwdmodel.calc_gravity, h)
    for m, event in enumerate(history.event_list):
        print("current event:", event)
        fwdmodel.gfunc = lambda r, h: np.array(event.rockprops(r, h))
        profile_timer(fwdmodel.calc_gravity, h)
        fwdmodel.fwd_data -= fwdmodel.edgemask * fwdmodel.voxmodel.mean()
        fig = plt.figure(figsize=(12,4))
        ax1 = plt.subplot(121)
        fwdmodel.plot_model_slice(ax=ax1)
        ax2 = plt.subplot(122)
        fwdmodel.plot_gravity(ax=ax2)
        plt.show()

if __name__ == "__main__":
    # plot_soft_if_then()
    # plot_subsurface_01()
    plot_subsurface_02()
