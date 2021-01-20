#!/usr/bin/env python

"""
RS 2020/06/04:  Block World Discretization Tests

This code examines in more detail the kinds of discontinuities induced on
posterior distributions of interpretable Bayesian inversions by the
discretization needed to solve the sensor forward model problem.
"""

# Imports after one of the SimPEG tensor mesh gravity forward model examples

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

from discretize import TensorMesh, TreeMesh
from discretize.utils import mkvc
from discretize.utils.meshutils import refine_tree_xyz

from SimPEG.utils import plot2Ddata, model_builder
from SimPEG import maps
from SimPEG.potential_fields import gravity
# import SimPEG.dask


def profile_timer(f, *args, **kwargs):
    """
    A wrapper to run functions and tell us how long they took
    :param f: function to run
    :param args: ordered parameters
    :param kwargs: keyword parameters
    :return: value of f for those parameters
    """
    t0 = time.time()
    result = f(*args, **kwargs)
    t1 = time.time()
    print("{} ran in {:.3f} sec".format(f.__name__, t1-t0))
    return result

# ============================================================================
#    Procedures to build baseline meshes and refine them around interfaces
# ============================================================================

def baseline_tensor_mesh(N, delta, centering="CCC"):
    """
    Set up a basic regular Cartesian tensor mesh other packages would use
    :param N: length of one edge of a cubical volume in cells
    :param delta: length of one edge of a mesh cube
    :param centering: a three-letter code specifying whether each axis is
        positive ('P'), negative ('N'), or centered ('C')
    :return: TensorMesh instance
    """
    hx = hy = hz = [(delta, N),]
    return TensorMesh([hx, hy, hz], centering)

def baseline_octree_mesh(N, delta):
    """
    Set up a basic regular Cartesian octree mesh as a default; this can then
    be refined once we specify model features or voxelizations, so it is NOT
    currently finalized
    :param N: length of one edge of a cubical volume in cells
    :param delta: length of one edge of a mesh cube
    :return: TreeMesh instance
    """
    h = delta * np.ones(N)
    mesh = TreeMesh([h, h, h], x0="CCC")
    mesh.refine(3, finalize=False)
    return mesh

def refine_octree_surface(mesh, f):
    """
    Refine an octree mesh around the given surface
    :param mesh: TreeMesh instance to refine
    :param f: function giving z(x,y) in physical units
    :return: TreeMesh instance
    """
    xx, yy = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
    zz = f(xx, yy)
    idx_valid = ~np.isnan(zz)
    xx, yy, zz = xx[idx_valid], yy[idx_valid], zz[idx_valid]
    surf = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]
    # Play with different octree_levels=[lx,ly,lz] settings below
    return refine_tree_xyz(
        mesh, surf, octree_levels=[1,1,1], method="surface", finalize=False
    )

# ============================================================================
#        Procedures to construct and manipulate gravity survey objects
# ============================================================================

def construct_survey(locations, components=['gz']):
    """
    Just calls the survey constructor; shorthand
    :param locations: list of (x, y, z) tuples with sensor locations
    :param components: list of strings from among the following:
        'gz' = vertical component of vector gravity anomaly
    :return: survey instance
    """
    components = ["gz"]
    receiver_list = [gravity.receivers.Point(locations, components=components)]
    source_field = gravity.sources.SourceField(receiver_list=receiver_list)
    return gravity.survey.Survey(source_field)

def survey_gridded_locations(Lx, Ly, Nx, Ny, z0, components=['gz']):
    """
    Set up a gravity survey on a regular grid
    :param Lx: length of a side of the discrete box along x-axis (m)
    :param Ly: length of a side of the discrete box along y-axis (m)
    :param Nx: number of grid points in x
    :param Ny: number of grid points in y
    :param z0: z location (m)
    :param components: see "components" argument of construct_survey()
    :return: np.array of sensor locations of shape [N, 3]
    """
    # Tile a square area centered at (0, 0) at height z0 (flat topography)
    dx, dy = Lx/Nx, Ly/Ny
    x = (np.arange(Nx) - 0.5*Nx + 0.5)*dx
    y = (np.arange(Ny) - 0.5*Ny + 0.5)*dy
    x, y = np.meshgrid(x, y)
    x, y = mkvc(x.T), mkvc(y.T)
    z = np.ones(x.shape) * z0
    locations = np.c_[x, y, z]
    return construct_survey(locations, components)

def survey_random_locations(Lx, Ly, Nr, z0, components=['gz']):
    """
    Set up a gravity survey with uniformly distributed locations
    :param Lx: length of a side of the discrete box along x-axis (m)
    :param Ly: length of a side of the discrete box along y-axis (m)
    :param Nr: number of random (x, y) sensor locations
    :param z0: z location (m)
    :param components: see "components" argument of construct_survey()
    :return: np.array of sensor locations of shape [N, 3]
    """
    x = Lx*(np.random.uniform(size=Nr) - 0.5)
    y = Ly*(np.random.uniform(size=Nr) - 0.5)
    z = np.ones(Nr)*z0
    locations = np.c_[x, y, z]
    return construct_survey(locations, components)

# ============================================================================
#                     Plotting and visualization functions
# ============================================================================

def plot_model_slice(mesh, model, ax=None,
                     grid=True, axlabels=True, colorbar=True):
    """
    Plot a vertical slice of a model so we can see what we're doing; this is
    completely ripped off one of the SimPEG notebooks, so if we decide we want
    different views of the subsurface we'll need to tweak it
    :param mesh: discretize.mesh instance
    :param model: np.array of rock properties conforming to mesh
        (usually evaluated as gfunc(mesh.gridCC) or similar)
    :param ax: optional matplotlib.axes.Axes instance (into subplot);
        if None, create new set of axes and hit matplotlib.show() at the end
    :return: nothing (yet)
    """
    show = (ax is None)
    if show:
        fig = plt.figure(figsize=(9, 4))
        ax = plt.gca()
    # ind_active = (model == model)
    # plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    plot_objects = mesh.plotSlice(
        model, # plotting_map*model,
        normal="Y",
        ax=ax,
        ind=int(mesh.hy.size / 2),
        grid=grid,
        clim=(np.min(model), np.max(model)),
        pcolorOpts={"cmap": "viridis"},
    )
    quadmeshimg = plot_objects[0]
    if axlabels:
        ax.set_title("Model slice at y = 0 m")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        colorbar_label = "Density (g/cm$^3$)"
    else:
        colorbar_label = ""
    if colorbar:
        plt.colorbar(quadmeshimg, aspect=10, pad=0.02, label=colorbar_label)
    if show:
        plt.show()

def plot_gravity(survey, data, ax=None, axlabels=True, colorbar=True):
    """
    Shows a 2-D overhead map of a gravity survey
    :param survey: survey instance
    :param data: measurements
    :param ax: optional matplotlib.axes.Axes instance (into subplot);
        if None, create new set of axes and hit matplotlib.show() at the end
    :return: nothing (yet)
    """
    show = (ax is None)
    if show:
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
    locations = survey.receiver_locations
    quadcont, axsub = plot2Ddata(
        survey.receiver_locations, data, ax=ax,
        contourOpts={"cmap": "bwr"}
    )
    if axlabels:
        ax.set_title("Gravity Anomaly (Z-component)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        colorbar_label = "Anomaly (mgal)"
    else:
        colorbar_label = ""
    if colorbar:
        plt.colorbar(quadcont, format="%.0g", pad=0.03, label=colorbar_label)
    if show:
        plt.show()

# ============================================================================
#               Procedures for instantiating discretized worlds
# ============================================================================

def gfunc_uniform_sphere(r, R, rho):
    """
    Point density field for a round sphere in the middle of space.
    :param r: np.array of shape (N, 3) representing N (x,y,z) locations
    :param R: radius of sphere (m)
    :param rho: density contrast inside sphere (kg/m^3)
    :return: np.array of shape (N, ) for evaluated densities
    """
    return rho * (np.sum(r**2, axis=1) < R**2)

def analytic_forward_gravity_sphere(survey, R, rho, r0=(0, 0, 0)):
    """
    Analytic forward gravity for a sphere (gz component)
    :param R: radius of sphere (m)
    :param rho: density of sphere (g/cm^3)
    :return: data in same geometry as survey passed in
    """
    r = survey.receiver_locations.T
    grav = gravity.analytics.GravSphereFreeSpace(r[0], r[1], r[2], R,
                                                 r0[0], r0[1], r0[2], rho)
    return grav


class DiscreteGravity:
    """
    Run regular gravity model on a single mesh
    """

    def __init__(self, mesh, survey, gfunc):
        """
        Initialize the problem
        :param mesh: discretize.mesh instance
        :param survey: SimPEG.gravity.Gravity.Survey instance
        :param gfunc: geology function mapping a np.array of (x,y,z) positions
            (shape = (N, 3)) to a set of rock properties (density contrast)
        """
        # Set all the initial stuff up
        self.survey = survey
        self.mesh = mesh
        self.gfunc = gfunc
        # Initialize a gravity simulation object to cache sensitivities and
        # make MCMC that much faster
        self.model_map = maps.IdentityMap(mesh=mesh, nP=mesh.nC)
        self.ind_active = np.array([True for i in range(mesh.nC)])
        self.fwd = gravity.simulation.Simulation3DIntegral(
            survey=self.survey,
            mesh=self.mesh,
            rhoMap=self.model_map,
            actInd=self.ind_active,
            store_sensitivities="ram",
        )
        self.voxmodel = None
        self.fwd_data = None

    def calc_voxmodel(self, *args):
        """
        Calculate voxelized rock properties
        :param *args: arguments to pass to gfunc
        :return: np.array of voxelized rock properties
        """
        self.voxmodel = np.array(self.gfunc(self.mesh.gridCC, *args))
        return self.voxmodel

    def calc_gravity(self, *args):
        """
        :param *args: arguments to pass to gfunc
        :return: np.array of gravity readings
        """
        # The baseline action is to just evaluate the rock properties directly
        # at the centers of the mesh, which will almost certainly not give
        # very good convergence behavior; if/when we sort out anti-aliasing
        # for rectilinear meshes, we should include it here
        self.calc_voxmodel(*args)
        self.fwd_data = self.fwd.dpred(self.voxmodel)
        return self.fwd_data

    def plot_model_slice(self, **kwargs):
        plot_model_slice(self.mesh, self.voxmodel, **kwargs)

    def plot_gravity(self, **kwargs):
        plot_gravity(self.survey, self.fwd_data, **kwargs)


class RichardsonGravity:
    """
    Run gravity on different meshes, then solve for the infinite resolution
    limit with appropriate uncertainty attached
    """

    def __init__(self, L, dL, survey, gfunc):
        """
        :param L: lateral extent of square survey area in meters
        :param dL: list of mesh block sizes in meters
        :param survey: gravity survey geometry
        :param gfunc: geology function mapping a np.array of (x,y,z) positions
            (shape = (N, 3)) to a set of rock properties (density contrast)
        """
        # Set all the initial stuff up
        self.survey = survey
        self.L = L
        self.dL = list(sorted(dL)[::-1])
        self.survey = survey
        self.gfunc = gfunc
        # Make a TensorMesh and forward model pair for each set of parameters
        self.meshxfwd = [ ]
        for dLi in self.dL:
            NL = 2*int(L/dLi)
            mesh = baseline_tensor_mesh(NL, dLi)
            model_map = maps.IdentityMap(mesh=mesh, nP=mesh.nC)
            ind_active = np.array([True for i in range(mesh.nC)])
            fwd = gravity.simulation.Simulation3DIntegral(
                survey=survey,
                mesh=mesh,
                rhoMap=model_map,
                actInd=ind_active,
                store_sensitivities="ram",
            )
            self.meshxfwd.append((mesh, fwd))

    def _setup_calc_gravity(self, *args):
        """
        Calculate possible gravity signals at multiple grid scales
        :param *args: arguments to pass to gfunc
        :return: (np.array of gravity readings, np.array of h values)
        """
        # Set up the regression matrices
        f, H = [ ], [ ]
        for i in range(len(self.dL)):
            dLi = self.dL[i]
            mesh, sim = self.meshxfwd[i]
            model = self.gfunc(mesh.gridCC, *args)
            dpred = sim.dpred(model)
            plot_gravity(self.survey, dpred)
            f.append(dpred)
            H.append(dLi)
        return f, H

    def calc_gravity_powerlaw(self, *args):
        """
        Calculate final gravity signal using Richardson extrapolation
        :param *args: arguments to pass to gfunc
        :return: np.array of gravity readings
        """
        f, H = self._setup_calc_gravity(*args)
        # The problem is a linear regression in the form
        # f = f0 + c*h^alpha = c0*1 + c1*h^alpha = H*C
        self.f = f = np.array(f)
        H = np.array([np.ones(len(H)), H]).T
        # Step through a bunch of alphas and find the lowest residuals
        # This is basically maximum likelihood
        alpha_min, C_min, rms_min = None, None, np.inf
        for alpha in 10**np.linspace(-0.5, 0.5, 11):
            C, res, rank, s = np.linalg.lstsq(H**alpha, f, rcond=None)
            rms = np.sqrt(np.sum(res/C[0]**2)/np.prod(f.shape))
            print("alpha, rms = {:.2f}, {:.2g}".format(alpha, rms))
            if rms < rms_min:
                alpha_min, C_min = alpha, C

        # Extract f0 from C
        f0, ch = C_min
        return f0

    def calc_gravity(self, *args):
        return self.calc_gravity_powerlaw(*args)


def main():
    """
    The main routine
    :return: nothing
    """
    h, t = 2.0, 2.0
    L, z0, R, rho, Ng = 16.0, 16.0, 10.0, 1000.0, 10
    components = ['gz']
    survey = survey_gridded_locations(L, L, Ng, Ng, z0, components)
    # survey = survey_random_locations(L, L, Ng*Ng, z0, components)

    # new Richardson setup
    RG = RichardsonGravity(L, [2.8, 2.0, 1.4], survey, gfunc_uniform_sphere)
    gravR = RG.calc_gravity(R, rho)
    grav0 = analytic_forward_gravity_sphere(survey, R, rho)[2]
    for i in range(len(RG.dL)):
        dL, res = RG.dL[i], (RG.f[i] - grav0)/RG.f[i]
        print("mu, std resids (dL = {}) = {:.3g} {:.3g}"
              .format(dL, np.mean(res), np.std(res)))
    resR = (gravR-grav0)/gravR
    print("mu, std resids (dL ~ 0.0) = {:.3g} {:.3g}"
          .format(np.mean(resR), np.std(resR)))

    # plot results
    plot_gravity(survey, gravR)
    plot_gravity(survey, grav0)
    plot_gravity(survey, res)
    plot_gravity(survey, resR)


if __name__ == '__main__':
    main()
    # notebook_test_scratch()