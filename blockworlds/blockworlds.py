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

from sklearn.decomposition import PCA


def profile_timer(f, *args, **kwargs):
    """
    A wrapper to run functions and tell us how long they took
    :param f: function to run
    :param args: ordered parameters
    :param kwargs: keyword parameters
    :return: value of f for those parameters
    """
    t0 = time.time()
    result = f(*args, *kwargs)
    t1 = time.time()
    print("{} ran in {:.3f} sec".format(f.__name__, t1-t0))
    return result

# ============================================================================
#    Procedures to build baseline meshes and refine them around interfaces
# ============================================================================

def baseline_tensor_mesh(N, delta):
    """
    Set up a basic regular Cartesian tensor mesh other packages would use
    :param N: length of one edge of a cubical volume in cells
    :param delta: length of one edge of a mesh cube
    :return: TensorMesh instance
    """
    hx = hy = hz = [(delta, N),]
    return TensorMesh([hx, hy, hz], "CCC")

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
#               Procedures for instantiating discretized worlds
# ============================================================================

def plot_model_slice(mesh, model):
    """
    Plot a vertical slice of a model so we can see what we're doing; this is
    completely ripped off one of the SimPEG notebooks, so if we decide we want
    different views of the subsurface we'll need to tweak it
    :param mesh: discretize.mesh instance
    :param model: np.array of rock properties conforming to mesh
        (usually evaluated as gfunc(mesh.gridCC) or similar)
    :return: nothing (yet)
    """
    fig = plt.figure(figsize=(9, 4))
    ind_active = (model == model)
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])
    mesh.plotSlice(
        plotting_map * model,
        normal="Y",
        ax=ax1,
        # ind=int(mesh.nCy / 2),
        ind=int(mesh.hy.size / 2),
        grid=True,
        clim=(np.min(model), np.max(model)),
        pcolorOpts={"cmap": "viridis"},
    )
    ax1.set_title("Model slice at y = 0 m")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
    norm = mpl.colors.Normalize(vmin=np.min(model), vmax=np.max(model))
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis
    )
    cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=12)
    plt.show()

def setup_world(mesh, gfunc, *args):
    """
    Shorthand; just evaluates gfunc at the centers of the grid cells
    :param mesh: discretize.mesh instance
    :param gfunc: callable accepting np.array of 3-D locations of shape (N, 3)
        and returning an np.array of rock properties of shape (N, )
    :param *args: ordered parameters to gfunc
    :return: np.array of shape (N, ) -- a geophysical "model" for SimPEG
        representing a 3-D array of rock properties
    """
    return gfunc(mesh.gridCC, *args)

def gfunc_uniform_sphere(r, R, rho):
    """
    Point density field for a round sphere in the middle of space.
    :param r: np.array of shape (N, 3) representing N (x,y,z) locations
    :param R: radius of sphere (m)
    :param rho: density contrast inside sphere (kg/m^3)
    :return: np.array of shape (N, ) for evaluated densities
    """
    return rho * (np.sum(r**2, axis=1) < R**2)

def setup_exp01_sphere_world(R, rho, mesh):
    """
    Set up experiment 1, a round sphere in the middle of space.  This is the
    simplest possible distribution, free of boundary value issues; the field
    should just be an inverse square law centered at the mesh center.
    :param R: radius of sphere (m)
    :param rho: density contrast of sphere (kg/m^3)
    :param mesh: discretize.mesh instance
    :return: nothing (yet)
    """
    gfunc = lambda r: rho * (np.sum(r**2, axis=1) < R**2)
    return setup_world(mesh, gfunc)

def calculate_forward_gravity(survey, mesh, model, plot=False):
    """
    Calculates forward gravity end to end.  A shorthand -- if you're going to
    calculate forward gravity repeatedly using this mesh, you'll want to cache
    the sensitivities inside the Simulation3DIntegral instance we're using.
    :param survey: gravity.survey.Survey instance
    :param mesh: discretize.mesh instance
    :param model:
    :param plot:
    :return:
    """
    # Shamelessly ganked from the SimPEG docs
    # Use the entire mesh
    model_map = maps.IdentityMap(mesh=mesh, nP=mesh.nC)
    ind_active = (model == model)
    # Simulation instance
    simulation = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        actInd=ind_active,
        store_sensitivities="ram", # "forward_only",
    )
    # Compute predicted data for the model passed in
    dpred = simulation.dpred(model)

    if plot:
        plot_gravity(survey, dpred)
    return dpred

def analytic_forward_gravity_sphere(survey, R, rho):
    """
    Analytic forward gravity for a sphere (gz component)
    :param R: radius of sphere (m)
    :param rho: density of sphere (g/cm^3)
    :return: data in same geometry as survey passed in
    """
    r = survey.receiver_locations.T
    grav = gravity.analytics.GravSphereFreeSpace(r[0], r[1], r[2], R,
                                                 0.0, 0.0, 0.0, rho)
    return grav

def plot_gravity(survey, data):
    """
    Shows a 2-D overhead map of a gravity survey
    :param survey: survey instance
    :param data: measurements
    :return: nothing
    """
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
    locations = survey.receiver_locations
    plot2Ddata(survey.receiver_locations, data, ax=ax1,
               contourOpts={"cmap": "bwr"})
    ax1.set_title("Gravity Anomaly (Z-component)")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    ax2 = fig.add_axes([0.82, 0.1, 0.03, 0.85])
    norm = mpl.colors.Normalize(vmin=-np.max(np.abs(data)),
                                vmax=np.max(np.abs(data)))
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr, format="%.1e"
    )
    cbar.set_label("$mgal$", rotation=270, labelpad=15, size=12)
    plt.show()

def run_grav_survey_sphere(survey, dL=1.0, run_treemesh=True):
    """
    Run forward gravity for the given survey and block resolution
    :param survey: gravity survey geometry
    :param dL: block size in meters
    :return: np.array of gravity measurements
    """
    L, z0, R, rho, Ng = 16.0, 8.0, 10.0, 1000.0, 10
    NL = 2*int(L/dL)
    print ("L, dL, N =", L, dL, NL)

    t0 = time.time()
    if run_treemesh:
        # Refine along sphere border
        mesh = baseline_octree_mesh(NL, dL)
        f = lambda x, y: np.sqrt(R**2 - x**2 - y**2)
        mesh = refine_octree_surface(mesh, f)
        f = lambda x, y: -np.sqrt(R**2 - x**2 - y**2)
        mesh = refine_octree_surface(mesh, f)
        mesh.finalize()
    else:
        mesh = baseline_tensor_mesh(NL, dL)
    t1 = time.time()
    print("setup_mesh ran in {:.3f} sec".format(t1-t0))

    # try a Richardson limit approach
    model = setup_world(mesh, gfunc_uniform_sphere, R, rho)
    plot_model_slice(mesh, model)
    grav = profile_timer(calculate_forward_gravity, survey, mesh, model)
    return grav

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
            model = setup_world(mesh, self.gfunc, *args)
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
    # old Richardson setup
    """
    grav2 = run_grav_survey_sphere(survey, h, run_treemesh=False)
    grav1 = run_grav_survey_sphere(survey, h/t, run_treemesh=False)
    gravR = (t**2 * grav1 - grav2)/(t**2 - 1)
    grav0 = analytic_forward_gravity_sphere(survey, R, rho)[2]
    res = grav2-grav0
    print("mu, std resids t=2 =", np.mean(res/gravR), np.std(res/gravR))
    res = grav1-grav0
    print("mu, std resids t=1 =", np.mean(res/gravR), np.std(res/gravR))
    res = gravR-grav0
    print("mu, std resids t=R =", np.mean(res/gravR), np.std(res/gravR))
    """
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