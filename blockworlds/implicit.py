#!/usr/bin/env python

"""
RS 2020/06/04:  Basic Implicit Modeling Package

This code is meant to build a package that can define complex geologies as
compositions of events that introduce implicit interfaces into a volume.
"""

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from blockworlds import profile_timer, DiscreteGravity
from blockworlds import baseline_tensor_mesh, survey_gridded_locations


# ============================================================================
#                               Helper functions
# ============================================================================


def sph2xyz(th, ph):
    thr, phr = np.radians(th), np.radians(ph)
    n = [np.cos(thr)*np.cos(phr), np.cos(thr)*np.sin(phr), np.sin(thr)]
    return np.array(n)

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

def soft_if_then_logistic(d, y0, y1, h):
    return (y1-y0) * (1.0 + np.tanh(2.2*d/h + 3.3*(d/h)**3))/2.0 + y0


soft_if_then = soft_if_then_logistic


# ============================================================================
#   Some machinery around probability distributions (not to reinvent pymc3!)
# ============================================================================


class LogPrior:

    _pars = [ ]
    Ndim = None

    def __init__(self, **kwargs):
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    def __call__(self):
        pass

    def sample(self, size=1):
        pass


class UniGaussianDist(LogPrior):
    """
    1-D Gaussian distribution
    """

    _pars = ['mean', 'std']
    Ndim = 1

    def __call__(self, x):
        lognorm = -0.5*np.log(2*np.pi*self.std**2)
        return -0.5*((x-self.mean)/self.std)**2 + lognorm

    def sample(self, size=1):
        return np.random.normal(self.mean, self.std, size=1)


class UniLognormDist(LogPrior):
    """
    1-D lognormal distribution
    """
    _pars = ['mean', 'std']
    Ndim = 1

    def __init__(self, **kwargs):
        super(UniLognormDist, self).__init__(**kwargs)
        self.logmu = np.log(self.mean**2 / np.sqrt(self.mean**2 + self.std**2))
        self.logsig = np.sqrt(np.log(1 + self.std**2/self.mean**2))

    def __call__(self, x):
        # Quick range check to avoid NaN logs
        u = np.array(np.atleast_1d(x))
        iok = (u > 0)
        u[iok] = np.log(u[iok])
        u[~iok] = -np.inf
        lognorm = -0.5 * np.log(2 * np.pi * self.logsig**2) - u
        r = np.zeros(u.shape)
        r[iok] = -0.5*((u[iok] - self.logmu)/self.logsig)**2 + lognorm[iok]
        r[~iok] = -np.inf
        return r

    def sample(self, size=1):
        return np.exp(np.random.normal(self.logmu, self.logsig, size=size))


class UniformDist(LogPrior):
    """
    1-D uniform distribution
    """

    _pars = ['mean', 'width']
    Ndim = 1

    def __call__(self, x):
        lognorm = -np.log(self.width)
        return -np.inf if np.abs(x-self.mean) > 0.5*self.width else lognorm

    def sample(self, size=1):
        hw = 0.5*self.width
        return np.random.uniform(self.mean - hw, self.mean + hw, size=size)


class vMFDist(LogPrior):
    """
    2-D von Mises-Fisher distribution (prior on unit normals / directions)
    """

    _pars = ['th0', 'ph0', 'kappa']
    Ndim = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Convert spherical coordinates of mode into a Cartesian basis
        self.gamma = sph2xyz(self.th0, self.ph0)
        self.v0 = np.cross(self.gamma, (0, 0, 1))
        self.v0 /= np.sqrt(np.dot(self.v0, self.v0))
        self.v1 = np.cross(self.v0, self.gamma)
        self.v1 /= np.sqrt(np.dot(self.v1, self.v1))

    def __call__(self, th, ph):
        pm = 0.5*3 - 1
        x = sph2xyz(th, ph)
        # Use exponentially scaled Bessel function to avoid divide by zero
        logbess = np.log(scipy.special.ive(pm, self.kappa)) + self.kappa
        lognorm = pm*np.log(self.kappa) - (pm+1)*np.log(2*np.pi) - logbess
        return lognorm + self.kappa*np.dot(self.gamma, x)

    def sample(self, size=1):
        """
        Follows Appendix A of
            Pakyuz-Charrier, E., et al., Solid Earth 9, 385–402 (2018)
        """
        # "For mu = (0, (.), 1) the pseudo-random vector is given by..."
        # W = dot product of random deviate with mean unit direction
        # V = random aximuthal angle to rotate around it
        xi = np.random.uniform(size=size)
        W = 1 + (np.log(xi) + np.log(1 - (xi-1)/xi
                            * np.exp(-2*self.kappa)))/self.kappa
        V = 2 * np.pi * np.random.uniform(size=size)
        # Construct spherical coordinates of deviate
        U = np.sqrt(1 - W*W)
        vrand = W*self.gamma + U*np.cos(V)*self.v0 + U*np.sin(V)*self.v1
        thrand = np.degrees(np.arcsin(vrand[2]))
        phrand = np.degrees(np.arctan2(vrand[1], vrand[0]))
        return thrand, phrand


# ============================================================================
#                More sophisticated implementation of GeoEvents
# ============================================================================


class GeoEvent:

    _pars = [ ]
    _priors = [ ]

    def __init__(self, priors, **kwargs):
        # Check whether the priors are correctly specified
        self._priors = priors
        ppars = [ ]
        for p in self._priors:
            parnames, dist = p[:-1], p[-1]
            for parname in parnames:
                if parname not in self._pars:
                    raise ValueError("{} is not an attribute of class {}"
                                     .format(parname, self.__class__.__name__))
            ppars.extend(parnames)
            if len(parnames) != dist.Ndim:
                raise ValueError("distribution {} is {}-dimensional"
                                 .format(dist.__class__.__name__, dist.Ndim))
        if sorted(ppars) != sorted(self._pars):
            raise ValueError("every variable of event {} must have exactly "
                             "one prior".format(self.__class__.__name__))
        # Other housekeeping
        self.set_to_prior_draw()
        self.set_kw_attrs(**kwargs)
        self.previous_event = None
        self.Npars = len(self._pars)

    def serialize(self):
        return np.array([getattr(self, attr) for attr in self._pars])

    def deserialize(self, *args):
        for i in range(len(args)):
            setattr(self, self._pars[i], args[i])

    def set_kw_attrs(self, **kwargs):
        for key, val in kwargs:
            if key in self._pars or key in self._hypars:
                setattr(self, key, val)

    def get_kw_attrs(self):
        return {attr: getattr(self, attr) for attr in self._pars}

    def set_previous_event(self, event):
        self.previous_event = event

    def rockprops(self, r, h):
        raise NotImplementedError

    def log_prior(self):
        lP = 0.0
        for p in self._priors:
            parnames, dist = p[:-1], p[-1]
            pars = [getattr(self, parname) for parname in parnames]
            lP += dist(*pars)
        return lP

    def set_to_prior_draw(self):
        for p in self._priors:
            parnames, dist = p[:-1], p[-1]
            vals = dist.sample(size=1)
            for parname, val in zip(parnames, vals):
                setattr(self, parname, val)

    def __str__(self):
        np = zip(self._pars, self.serialize())
        parstr = ', '.join(["{}={}".format(n, p) for n, p in np])
        return "{}({})".format(self.__class__.__name__, parstr)


class BasementEvent(GeoEvent):

    _pars = ['density']

    def rockprops(self, r, h):
        return self.density * np.ones(shape=r.shape[:-1])


class StratLayerEvent(GeoEvent):

    _pars = ['thickness', 'density']

    def rockprops(self, r, h):
        assert(isinstance(self.previous_event, GeoEvent))
        rp = r + np.array([0, 0, self.thickness])
        rho_up = self.density*np.ones(shape=r.shape[:-1])
        rho_down = self.previous_event.rockprops(rp, h)
        return soft_if_then(rp[:,2], rho_down, rho_up, h)


class PlanarFaultEvent(GeoEvent):

    _pars = ['x0', 'y0', 'nth', 'nph', 's']

    def rockprops(self, r, h):
        assert(isinstance(self.previous_event, GeoEvent))
        # Point on fault specified in Cartesian coordinates; assume z0 = 0
        # since we're probably just including geologically observed faults
        r0 = np.array([self.x0, self.y0, 0.0])
        # Unit normal to fault ("polar vector") specified with
        # nth = elevation angle (+90 = +z, -90 = -z)
        # nph = azimuthal angle (runs counterclockwise, zero in +x direction)
        n = sph2xyz(self.nth, self.nph)
        # Geology in +n direction slips relative to the background
        # Slip is vertical (+z direction) in units of meters along the fault
        v = np.cross(np.cross([0, 0, 1], n), n)
        vnorm = np.sqrt(np.sum(np.atleast_2d(v ** 2), axis=1))
        rdelt = self.s * v/vnorm
        g0 = self.previous_event.rockprops(r, h)
        g1 = self.previous_event.rockprops(r + rdelt, h)
        return soft_if_then(np.dot(r-r0, n), g0, g1, h)


class FoldEvent(GeoEvent):

    _pars = ['nth', 'nph', 'pitch', 'phase', 'wavelength', 'amplitude']

    def rockprops(self, r, h):
        assert(isinstance(self.previous_event, GeoEvent))
        # nth, nph define compression axis of fold
        # psi defines pitch, relative to an axis aligned with +z
        n = sph2xyz(self.nth, self.nph)
        rpsi = np.radians(self.pitch)
        rphs = np.radians(self.phase)
        # Define an orthonormal frame for the fold
        # n = fold axis, v0 = horizontal, v1 = vertical
        v0 = np.cross(n, [0, 0, 1])
        v0 /= np.sqrt(np.dot(v0, v0))
        v1 = np.cross(v0, n)
        v1 /= np.sqrt(np.dot(v1, v1))
        # Define perturbation of positions
        v = np.sin(rpsi)*v0 + np.cos(rpsi)*v1
        sinarg = 2*np.pi*np.dot(r, n)/self.wavelength + rphs
        rdelt = self.amplitude*np.sin(sinarg)[:,np.newaxis]*v
        return self.previous_event.rockprops(r + rdelt, h)


class SphereIntrusionEvent(GeoEvent):

    _pars = ['x0', 'y0', 'z0', 'R', 'density']

    def rockprops(self, r, h):
        assert(isinstance(self.previous_event, GeoEvent))
        # Sphere center in Cartesian coordinates
        r0 = np.array([self.x0, self.y0, self.z0])
        # Uniform density contrast inside sphere
        # Whatever background geology outside sphere
        g0 = self.previous_event.rockprops(r, h)
        g1 = self.density * np.ones(shape=r.shape[:-1])
        fval = self.R - np.sqrt(np.sum((r-r0)**2, axis=1))
        return soft_if_then(fval, g0, g1, h)


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

    def set_to_prior_draw(self):
        for event in self.event_list:
            event.set_to_prior_draw()


# ============================================================================
#                Testing construction of non-trivial subsurfaces
# ============================================================================


def gen_two_fault_model_demo(pars):
    """
    Generate a demo model with three stratigraphic layers and two faults
    (15 parameters), based on some starting parameter values as provided by
    Mark Lindsay.
    :param pars: np.array of shape (15,)
    :return: GeoHistory instance
    """
    (rho_0, dz_1, rho_1, dz_2, rho_2,       # stratigraphy
     x0_3, y0_3, nth_3, nph_3, s_3,         # 1st fault
     x0_4, y0_4, nth_4, nph_4, s_4) = pars  # 2nd fault

    # Initialize a new history and set the prior means at the values of the
    # true parameters.  This implicitly makes us assume we're within the
    # support of the prior throughout, so if we want to investigate possible
    # prior misspecification we'll have to modify or supplant this.
    history = GeoHistory()
    history.add_event(
        BasementEvent(
            [('density', UniLognormDist(mean=rho_0, std=0.5))]
        )
    )
    history.add_event(
        StratLayerEvent(
            [('thickness', UniLognormDist(mean=dz_1, std=50.0)),
             ('density', UniLognormDist(mean=rho_1, std=0.1))]
        )
    )
    history.add_event(
        StratLayerEvent(
            [('thickness', UniLognormDist(mean=dz_2, std=50.0)),
             ('density', UniLognormDist(mean=rho_2, std=0.1))]
        )
    )
    history.add_event(
        PlanarFaultEvent(
            [('x0', UniGaussianDist(mean=x0_3, std=1.0)),
             ('y0', UniGaussianDist(mean=y0_3, std=1.0)),
             ('nth', 'nph', vMFDist(th0=nth_3, ph0=nph_3, kappa=25)),
             ('s', UniGaussianDist(mean=s_3, std=150.0))]
        )
    )
    history.add_event(
        PlanarFaultEvent(
            [('x0', UniGaussianDist(mean=x0_4, std=1.0)),
             ('y0', UniGaussianDist(mean=y0_4, std=1.0)),
             ('nth', 'nph', vMFDist(th0=nth_4, ph0=nph_4, kappa=25)),
             ('s', UniGaussianDist(mean=s_4, std=150.0))]
        )
    )
    return history


def gen_fold_model_demo(pars):
    """
    Generate a demo model with three stratigraphic layers, a fold and a fault
    (16 parameters), based on some starting parameter values as provided by
    Mark Lindsay.
    :param pars: np.array of shape (16,)
    :return: GeoHistory instance
    """
    (rho_0, dz_1, rho_1, dz_2, rho_2,       # stratigraphy
     nth_3, nph_3, psi_3, phi_3, L_3, A_3,  # fold
     x0_4, y0_4, nth_4, nph_4, s_4) = pars  # fault

    # Initialize a new history and set the prior means at the values of the
    # true parameters.  This implicitly makes us assume we're within the
    # support of the prior throughout, so if we want to investigate possible
    # prior misspecification we'll have to modify or supplant this.
    history = GeoHistory()
    history.add_event(
        BasementEvent(
            [('density', UniLognormDist(mean=rho_0, std=0.5))]
        )
    )
    history.add_event(
        StratLayerEvent(
            [('thickness', UniLognormDist(mean=dz_1, std=50.0)),
             ('density', UniLognormDist(mean=rho_1, std=0.1))]
        )
    )
    history.add_event(
        StratLayerEvent(
            [('thickness', UniLognormDist(mean=dz_2, std=50.0)),
             ('density', UniLognormDist(mean=rho_2, std=0.1))]
        )
    )
    history.add_event(
        FoldEvent(
            [('nth', 'nph', vMFDist(th0=nth_3, ph0=nph_3, kappa=25)),
             ('pitch', UniGaussianDist(mean=psi_3, std=1.0)),
             ('phase', UniGaussianDist(mean=phi_3, std=1.0)),
             ('wavelength', UniLognormDist(mean=L_3, std=50.0)),
             ('amplitude', UniLognormDist(mean=A_3, std=5.0))]
        )
    )
    history.add_event(
        PlanarFaultEvent(
            [('x0', UniGaussianDist(mean=x0_4, std=1.0)),
             ('y0', UniGaussianDist(mean=y0_4, std=1.0)),
             ('nth', 'nph', vMFDist(th0=nth_4, ph0=nph_4, kappa=100)),
             ('s', UniGaussianDist(mean=s_4, std=15.0))]
        )
    )
    return history


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
    history = gen_two_fault_model_demo(
        [3.0, 1900.0, 2.5, 2500.0, 2.0,
         -4000.0, 0.0, +20.0, 0.0, -4200.0,
         +4000.0, 0.0, -20.0, 0.0, +4200.0]
    )
    # Add a final Fold event just for laughs
    history.add_event(
        FoldEvent(
            [('nth', 'nph', vMFDist(th0=+0.0, ph0=0.0, kappa=100)),
             ('pitch', UniGaussianDist(mean=0.0, std=30.0)),
             ('phase', UniformDist(mean=0.0, width=360.0)),
             ('wavelength', UniGaussianDist(mean=3000.0, std=300.0)),
             ('amplitude', UniGaussianDist(mean=300.0, std=50.0))]
        )
    )
    print("history.pars =", history.serialize())
    # Can also set parameters all at once -- good for running MCMC
    history.set_to_prior_draw()
    history.deserialize([3.0, 1900.0, 2.5, 2500.0, 2.0,
                         -4000.0, 0.0, +20.0, 0.0, -4200.0,
                         +4000.0, 0.0, -20.0, 0.0, +4200.0,
                         -0.0, 0.0, 0.0, 0.0, 3000.0, 300.0])
    print("history.pars =", history.serialize())
    print("history.prior =", history.logprior())
    # Plot a cross-section
    fwdmodel = DiscreteGravity(mesh, survey, history.event_list[0])
    fwdmodel.gfunc = history.event_list[0].rockprops
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
    plot_subsurface_02()
