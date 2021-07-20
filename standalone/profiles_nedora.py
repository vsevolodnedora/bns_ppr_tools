import gc
import argparse
import h5py
from math import pi, log10
import numpy as np
import scidata.carpet.grid as grid
import scidata.units as ut
from scidata.carpet.interp import Interpolator
import time
import os

"""
    Compute the baryonic mass within the simulation for a given cutoffs.
    The mass is computed by evaluating the proper density D = rho * W * vol,
    where 'rho' is the rest-mass density, 'W' is the lorentz factor and 
    'vol' is the square-root of the three-metric.
    
    In order to avoid double-counting the data in overlapping refinement 
    levels, the the smaller refinment is always 'masked out' from the 
    current one.
    No interpolation is performed.
"""


NLEVELS = 7
multiplier = 2. # multiply the result by this number to account for 2 hemispheres

# cut-off density for the disk (rho < 10^13 g/cm^3)
density_cutoff = ut.conv_dens(ut.cgs, ut.cactus, 1.0E13)
# cut-off lapse for the data inside the horizon
lapse_cutoff = 0.3

def read_carpet_grid(dfile):
    """
    Generate carpet grid object from a profile
    """
    L = []
    for il in range(NLEVELS):
        gname = "reflevel={}".format(il)
        group = dfile[gname]
        level = grid.basegrid()
        level.delta = np.array(group.attrs["delta"])
        level.dim = 3
        level.directions = range(3)
        level.iorigin = np.array([0, 0, 0], dtype=np.int32)
        level.origin = np.array(group.attrs["extent"][0::2])
        level.n = np.array(group["rho"].shape, dtype=np.int32)
        level.rlevel = il
        L.append(level)
    return grid.grid(sorted(L, key=lambda x: x.rlevel))


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sim_path", dest="sim_path", help="Simulation path to analyze.")
args = parser.parse_args()

## Simulation must containd folder: profiles/
out_dir = "{}/postprocess/disk/".format(args.sim_path)
in_dir = "{}/profiles/3d/".format(args.sim_path)

try:
    iterations = os.listdir(in_dir)
    if len(iterations) < 1:
        print("No iterations found in {}".format(in_dir))
        exit()
    os.system("mkdir -p {}".format(out_dir))
except:
    print("{}: no directory found.".format(in_dir))
    exit()

# loop over all the iterations
its, mss, deltats = [], [], []

iterations = np.sort(np.array([it.split('.')[0] for it in iterations], dtype=np.int))

for _iteration in iterations:

    # print("it:{} ".format(int(_iteration.split('.')[0]))),

    # load data
    fname = "{}/{}".format(in_dir, str(_iteration)+'.h5')
    dfile = h5py.File(fname, "r")

    # Read carpet grid
    grid_simu = read_carpet_grid(dfile)

    D, X, Y, Z, deltas = [], [], [], [], []

    # for each iteration level extract the data and compute density
    for idx, rlevel in enumerate(grid_simu):
        #print("idx: {}".format(idx))
        start_t = time.time()
        #print("Processing reflevel {}...".format(idx)),
        group = dfile["reflevel={}".format(rlevel.rlevel)]

        # generate mesh in form of 3D arrays for each direction
        x, y, z = rlevel.mesh()
        X.append(x)
        Y.append(y)
        Z.append(z)
        deltas.append(rlevel.delta)

        # extract data in form of 3D array
        rho = np.nan_to_num(np.array(group["rho"]))
        eps = np.array(group["eps"])
        W = np.nan_to_num(np.array(group["w_lorentz"]))
        vol = np.nan_to_num(np.array(group["vol"]))
        lapse = np.nan_to_num(np.array(group["lapse"]))

        # set nan to zero and cutoff the disk density
        dens = np.nan_to_num(rho * W * vol)
        # apply the cut-off for density and lapse
        mask = np.where(np.logical_or(dens >= density_cutoff, lapse <= lapse_cutoff))
        # apply the mask to density
        dens[mask] = 0

        D.append(dens)

        # clear the memory
        del rho
        del eps
        del W
        del dens
        #print("done! ({:.2f} sec)".format(time.time() - start_t))
    # clear the stack
    gc.collect()

    start_t = time.time()

    # create "Frames" of masks to mask previous rl from the next
    # to avoid double counting the data
    nlevelist = np.arange(NLEVELS, 0, -1) - 1
    x = [] # for reversed data
    y = []
    z = []
    Dm = []
    # go in reverse, from the smallest rl to the largest, prgressivly
    # masking-out the previous one from the next (leaving rl=0 intact)
    mass = 0.
    for ii, rl in enumerate(nlevelist):
        # igrid = NLEVELS - ii - 1
        # print(rl, igrid)
        x.append(X[rl][3:-3, 3:-3, 3:-3])
        y.append(Y[rl][3:-3, 3:-3, 3:-3])
        z.append(Z[rl][3:-3, 3:-3, 3:-3])
        mask = np.ones(x[ii].shape, dtype=bool)

        # create a mask of a previous refinement level (the smaller one than the current, ii )
        if ii > 0:
            x_ = (x[ii][:, :, :] <= x[ii - 1][:, 0, 0].max()) & (x[ii][:, :, :] >= x[ii - 1][:, 0, 0].min())
            y_ = (y[ii][:, :, :] <= y[ii - 1][0, :, 0].max()) & (y[ii][:, :, :] >= y[ii - 1][0, :, 0].min())
            z_ = (z[ii][:, :, :] <= z[ii - 1][0, 0, :].max()) & (z[ii][:, :, :] >= z[ii - 1][0, 0, :].min())
            mask = mask & np.invert((x_ & y_ & z_))

        Dm = np.copy(D[rl][3:-3, 3:-3, 3:-3])[mask] # Mask-out the previous rl and remove ghosts from the data
        rl_mass = float(multiplier * np.sum(Dm) * np.prod(deltas[rl])) # compute the mass in the domain
        mass = mass + rl_mass
        #print("rl:{} mass:{}".format(rl, rl_mass))
    print("{} [Msun]".format(mass))

    its.append(int(_iteration))
    mss.append(float(mass))

    deltats.append(time.time() - start_t)

    del X
    del Y
    del Z
    del x
    del y
    del z
    del Dm

np.savetxt("./mdisk_mask.txt", np.vstack((np.array(its),
                                          np.array(mss),
                                          #np.array(deltats)
                                          )).T, header='it mass[Mo]')

