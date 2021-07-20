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

NLEVELS = 7
# python old_mjenclosed.py -s /data/numrel/francesco.zappa/SLy_LRZ/SLy_130130_LK_LR/

def read_carpet_grid(dfile):
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


def make_stretched_grid(x0, x1, x2, nlin, nlog):
    assert x1 > 0
    assert x2 > 0
    x_lin_f = np.linspace(x0, x1, nlin)
    x_log_f = 10.0 ** np.linspace(log10(x1), log10(x2), nlog)
    return np.concatenate((x_lin_f, x_log_f))


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

print("\n")

iterations = np.sort(np.array([it.split('.')[0] for it in iterations], dtype=np.int))


its, masses, deltats = [], [], []
for _iteration in iterations:
    fname = "{}/{}".format(in_dir, str(_iteration) + ".h5")
    # print("Working on {}..".format(fname))

    dfile = h5py.File(fname, "r")

    # Read carpet grid
    start_t = time.time()
    # print("Reading grid..."),
    grid_simu = read_carpet_grid(dfile)
    # print("done! ({:.2f} sec)".format(time.time() - start_t))

    # cut-off density for the disk (rho < 10^13 g/cm^3)
    density_cutoff = ut.conv_dens(ut.cgs, ut.cactus, 1.0E13)
    # cut-off lapse for the data inside the horizon
    lapse_cutoff = 0.3

    # Rest mass and angular momentum densities
    D, J = [], []
    for idx, rlevel in enumerate(grid_simu):
        start_t = time.time()
        # print("Processing reflevel {}...".format(idx)),
        group = dfile["reflevel={}".format(rlevel.rlevel)]

        x, y, z = rlevel.mesh()

        rho = np.nan_to_num(np.array(group["rho"]))
        eps = np.array(group["eps"])
        press = np.array(group["press"])
        velx = np.array(group["velx"])
        vely = np.array(group["vely"])
        velz = np.array(group["velz"])
        W = np.nan_to_num(np.array(group["w_lorentz"]))
        gxx = np.array(group["gxx"])
        gxy = np.array(group["gxy"])
        gxz = np.array(group["gxz"])
        gyy = np.array(group["gyy"])
        gyz = np.array(group["gyz"])
        gzz = np.array(group["gzz"])
        vol = np.nan_to_num(np.array(group["vol"]))
        lapse = np.nan_to_num(np.array(group["lapse"]))

        vup = [velx, vely, velz]
        vlow = [np.zeros_like(vv) for vv in [velx, vely]]
        metric = [[gxx, gxy, gxz], [gxy, gyy, gyz], [gxz, gyz, gzz]]
        for i in range(2):
            for j in range(3):
                vlow[i][:] += metric[i][j][:] * vup[j][:]
        vphi = -y * vlow[0] + x * vlow[1]

        # set nan to zero and cutoff the disk density
        dens = np.nan_to_num(rho * W * vol)
        mask = np.where(np.logical_or(dens >= density_cutoff, lapse <= lapse_cutoff))
        # mask = np.where(np.logical_or(rho >= density_cutoff, lapse <= lapse_cutoff))
        dens[mask] = 0

        D.append(dens)
        J.append((rho * (1 + eps) + press) * W * W * vol * vphi)

        # print("rl: %d = %.6f" % (rlevel.rlevel, np.sum(dens) * np.prod(rlevel.delta)))

        del vup
        del vlow
        del metric
        del vphi
        del rho
        del eps
        del press
        del velx
        del vely
        del velz
        del W
        del gxx
        del gxy
        del gxz
        del gyy
        del gyz
        del gzz
        del dens
        # print("done! ({:.2f} sec)".format(time.time() - start_t))
    gc.collect()

    start_t = time.time()

    # Make cylindrical grid
    r_cyl_f = make_stretched_grid(0., 15., 512., 75, 64)
    z_cyl_f = make_stretched_grid(0., 15., 512., 75, 64)
    phi_cyl_f = np.linspace(0, 2 * pi, 64)

    r_cyl = 0.5 * (r_cyl_f[1:] + r_cyl_f[:-1])
    z_cyl = 0.5 * (z_cyl_f[1:] + z_cyl_f[:-1])
    phi_cyl = 0.5 * (phi_cyl_f[1:] + phi_cyl_f[:-1])

    r_cyl_3d, phi_cyl_3d, z_cyl_3d = np.meshgrid(r_cyl, phi_cyl, z_cyl, indexing='ij')
    x_cyl_3d = r_cyl_3d * np.cos(phi_cyl_3d)
    y_cyl_3d = r_cyl_3d * np.sin(phi_cyl_3d)

    # Interpolate D and J to the cylindrical grid
    start_t = time.time()
    # print("Interpolating to cylindrical grid..."),

    iD = Interpolator(grid_simu, D, interp=1)
    iJ = Interpolator(grid_simu, J, interp=1)

    xi = np.column_stack([x_cyl_3d.flatten(), y_cyl_3d.flatten(), z_cyl_3d.flatten()])
    D_cyl = iD(xi).reshape(r_cyl_3d.shape)
    J_cyl = iJ(xi).reshape(r_cyl_3d.shape)

    # print("done! ({:.2f} sec)".format(time.time() - start_t))

    # Reduce data (assume symmetry across xy-plane)
    #start_t = time.time()
    # print("Reduce data..."),
    dphi_cyl = np.diff(phi_cyl_f)[np.newaxis, :, np.newaxis]
    dz_cyl = np.diff(z_cyl_f)[np.newaxis, np.newaxis, :]
    dr_cyl = np.diff(r_cyl_f)
    D_rc = 2 * np.sum(D_cyl * dz_cyl * dphi_cyl, axis=(1, 2))
    J_rc = 2 * np.sum(J_cyl * dz_cyl * dphi_cyl, axis=(1, 2))
    # print("done! ({:.2f} sec)".format(time.time() - start_t))

    # Write to disk
    #start_t = time.time()
    # print("Output data..."),

    # ofile = open(out_dir + "MJ_encl_" + fname.split('/')[-1].replace(".h5", ".txt"), "w")
    # # ofile = open("out/MJ_encl_" + fname.split('/')[-1].replace(".h5", ".txt"), "w")
    # ofile.write("# 1:rcyl 2:drcyl 3:M 4:J\n")
    # for i in range(r_cyl.shape[0]):
    #     ofile.write("{} {} {} {}\n".format(r_cyl[i], dr_cyl[i], D_rc[i], J_rc[i]))
    # ofile.close()

    print("{} {}".format(_iteration, np.sum(r_cyl * dr_cyl * D_rc)))
    # print("{} %.6f Mo" % (np.sum(r_cyl * dr_cyl * D_rc)))
    # print("done! ({:.2f} sec)".format(time.time() - start_t))
    its.append(_iteration)
    masses.append(np.sum(r_cyl * dr_cyl * D_rc))

    deltats.append(time.time() - start_t)

out = np.vstack((
    np.array(its),
    np.array(masses),
    #np.array(deltats)
)).T

np.savetxt('./mdisk_int.txt', out, header='it mdisk[Mo]')