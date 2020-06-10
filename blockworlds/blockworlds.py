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
    z = np.ones(Nr)
    locations = np.c_[x, y, z]
    return construct_survey(locations, components)

# ============================================================================

def setup_exp01_sphere_world(R, rho, mesh, plot_slice=False):
    """
    Set up experiment 1, a round sphere in the middle of space.  This is the
    simplest possible distribution, free of boundary value issues; the field
    should just be an inverse square law centered at the mesh center.
    :param R: radius of sphere (m)
    :param rho: density contrast of sphere (kg/m^3)
    :param mesh: discretize.mesh instance
    :return: nothing (yet)
    """
    # Use the SimPEG block utilities to build this; it should just evaluate
    # the density at the center of each cell without anti-aliasing
    model = np.zeros(mesh.nC)
    idx_sphere = model_builder.getIndicesSphere(np.r_[0.0, 0.0, 0.0], R,
                                                mesh.gridCC)
    model[idx_sphere] = rho
    ind_active = (model == model)
    # Plot a slice to verify we made this correctly
    if plot_slice:
        # Pulled this incantation straight from the docs
        fig = plt.figure(figsize=(9, 4))
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
    return model

def calculate_forward_gravity(survey, mesh, model, plot=False):
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

def run_grav_survey_sphere(survey, dL=1.0):
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
    run_treemesh = True
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
    model = setup_exp01_sphere_world(R, rho, mesh, plot_slice=True)
    grav = profile_timer(calculate_forward_gravity, survey, mesh, model)
    return grav

def main():
    """
    The main routine
    :return: nothing
    """
    h, t = 0.5, 2.0
    L, z0, R, rho, Ng = 16.0, 8.0, 10.0, 1000.0, 10
    components = ['gz']
    survey = survey_gridded_locations(L, L, Ng, Ng, z0, components)
    grav2 = run_grav_survey_sphere(survey, h)
    grav1 = run_grav_survey_sphere(survey, h/t)
    gravR = (t * grav1 - grav2)/(t - 1)
    grav0 = analytic_forward_gravity_sphere(survey, R, rho)[2]
    res = grav2-grav0
    print("mu, std resids t=2 =", np.mean(res/gravR), np.std(res/gravR))
    res = grav1-grav0
    print("mu, std resids t=1 =", np.mean(res/gravR), np.std(res/gravR))
    res = gravR-grav0
    print("mu, std resids t=R =", np.mean(res/gravR), np.std(res/gravR))
    plot_gravity(survey, gravR)
    plot_gravity(survey, gravR-grav0)


if __name__ == '__main__':
    main()