"""

"""

import numpy as np

class FORMULAS:

    def __init__(self):
        pass

    @staticmethod
    def r(x, y):
        return np.sqrt(x ** 2 + y ** 2)# + z ** 2)

    @staticmethod
    def density(rho, w_lorentz, vol):
        return rho * w_lorentz * vol

    @staticmethod
    def vup(velx, vely, velz):
        return [velx, vely, velz]

    @staticmethod
    def metric(gxx, gxy, gxz, gyy, gyz, gzz):
        return [[gxx, gxy, gxz], [gxy, gyy, gyz], [gxz, gyz, gzz]]

    @staticmethod
    def enthalpy(eps, press, rho):
        return 1 + eps + (press / rho)

    @staticmethod
    def shift(betax, betay, betaz):
        return [betax, betay, betaz]

    @staticmethod
    def shvel(shift, vlow):
        shvel = np.zeros(shift[0].shape)
        for i in range(len(shift)):
            shvel += shift[i] * vlow[i]
        return shvel

    @staticmethod
    def u_0(w_lorentz, shvel, lapse):
        return w_lorentz * (shvel - lapse)

    @staticmethod
    def hu_0(h, u_0):
        return h * u_0

    @staticmethod
    def vlow(metric, vup):
        vlow = [np.zeros_like(vv) for vv in [vup[0], vup[1], vup[2]]]
        for i in range(3):  # for x, y
            for j in range(3):
                vlow[i][:] += metric[i][j][:] * vup[j][:]  # v_i = g_ij * v^j (lowering index) for x y
        return vlow

    @staticmethod
    def vphi(x, y, vlow):
        return -y * vlow[0] + x * vlow[1]

    @staticmethod
    def vr(x, y, r, vup):
        # r = np.sqrt(x ** 2 + y ** 2)
        # print("x: {}".format(x.shape))
        # print("y: {}".format(y.shape))
        # print("r: {}".format(y.shape))
        # print("vup[0]: {}".format(vup[0].shape))

        return (x / r) * vup[0] + (y / r) * vup[1]

    @staticmethod
    def theta(r, z):
        # r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return np.arccos(z/r)

    @staticmethod
    def phi(x, y):
        return np.arctan2(y, x)

    @staticmethod
    def dens_unb_geo(u_0, rho, w_lorentz, vol):

        c_geo = -u_0 - 1.0
        mask_geo = (c_geo > 0).astype(int)  # 1 or 0
        rho_unbnd_geo = rho * mask_geo  # if 1 -> same, if 0 -> masked
        dens_unbnd_geo = rho_unbnd_geo * w_lorentz * vol

        return dens_unbnd_geo

    @staticmethod
    def dens_unb_bern(enthalpy, u_0, rho, w_lorentz, vol):

        density = rho * w_lorentz * vol

        c_ber = -enthalpy * u_0 - 1.0
        mask_ber = (c_ber > 0).astype(int)
        # rho_unbnd_bernoulli = rho * mask_ber
        density_unbnd_bernoulli = density * mask_ber
        # print(density_unbnd_bernoulli); exit(1)
        # print(np.sum(density_unbnd_bernoulli / density))
        # print(np.unique(density_unbnd_bernoulli / density)); exit(1)
        return density_unbnd_bernoulli

    @staticmethod
    def dens_unb_garch(enthalpy, u_0, lapse, press, rho, w_lorentz, vol):

        c_ber = -enthalpy * u_0 - 1.0
        c_gar = c_ber - (lapse / w_lorentz) * (press / rho)
        mask_gar = (c_gar > 0).astype(int)
        rho_unbnd_garching = rho * mask_gar
        density_unbnd_garching = rho_unbnd_garching * w_lorentz * vol

        return density_unbnd_garching

    @staticmethod
    def ang_mom(rho, eps, press, w_lorentz, vol, vphi):
        return (rho * (1 + eps) + press) * w_lorentz * w_lorentz * vol * vphi

    @staticmethod
    def ang_mom_flux(ang_mom, lapse, vr):
        return ang_mom * lapse * vr

    # data manipulation methods
    @staticmethod
    def get_slice(x3d, y3d, z3d, data3d, slice='xy'):

        if slice == 'yz':
            ix0 = np.argmin(np.abs(x3d[:, 0, 0]))
            if abs(x3d[ix0, 0, 0]) < 1e-15:
                res = data3d[ix0, :, :]
            else:
                if x3d[ix0, 0, 0] > 0:
                    ix0 -= 1
                res = 0.5 * (data3d[ix0, :, :] + data3d[ix0 + 1, :, :])
        elif slice == 'xz':
            iy0 = np.argmin(np.abs(y3d[0, :, 0]))
            if abs(y3d[0, iy0, 0]) < 1e-15:
                res = data3d[:, iy0, :]
            else:
                if y3d[0, iy0, 0] > 0:
                    iy0 -= 1
                res = 0.5 * (data3d[:, iy0, :] + data3d[:, iy0 + 1, :])
        elif slice == 'xy':
            iz0 = np.argmin(np.abs(z3d[0, 0, :]))
            if abs(z3d[0, 0, iz0]) < 1e-15:
                res = data3d[:, :, iz0]
            else:
                if z3d[0, 0, iz0] > 0 and iz0 > 0:
                    iz0 -= 1
                res = 0.5 * (data3d[:, :, iz0] + data3d[:, :, iz0 + 1])
        else:
            raise ValueError("slice:{} not recognized. Use 'xy', 'yz' or 'xz' to get a slice")
        return res


    @staticmethod
    def q_eff_nua_over_density(q_eff_nea, density):
        return q_eff_nea / density

    @staticmethod
    def abs_energy_over_density(abs_energy, density):
        return abs_energy / density

    # --------- OUTFLOWED -----

    @staticmethod
    def vinf(eninf):
        return np.sqrt(2 * eninf)

    @staticmethod
    def vinf_bern(eninf, enthalpy):
        return np.sqrt(2*(enthalpy*(eninf + 1) - 1))

    @staticmethod
    def vel(w_lorentz):
        return np.sqrt(1 - 1 / (w_lorentz**2))

    @staticmethod
    def get_tau(rho, vel, radius, lrho_b):

        rho_b = 10 ** lrho_b
        tau_0 = 0.5 * 2.71828182845904523536 * (radius / vel) * (0.004925794970773136) # in ms
        tau_b = tau_0 * ((rho/rho_b) ** (1.0 / 3.0))
        return tau_b # ms